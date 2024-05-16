from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, nvidia

from dataclasses import dataclass
import functools
from typing import Any, Tuple
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path


@functools.lru_cache()
def _path_to_binary(binary: str):
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
    ]

    for bin in paths:
        if os.path.exists(bin) and os.path.isfile(bin):
            result = subprocess.check_output([bin, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                if version is not None:
                    return bin, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_ptxas_version():
    version = subprocess.check_output([_path_to_binary("ptxas")[0], "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        return 80 + minor
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor
    raise RuntimeError("Triton only support CUDA 10.0 or higher")


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    enable_fp_fusion: bool = True
    allow_fp8e4nv: bool = False
    allow_fp8e4b15: bool = False
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = os.getenv("TRITON_LIBDEVICE_PATH", str(default_libdir / 'libdevice.10.bc'))
        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CUDABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cuda'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        assert isinstance(self.capability, int)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        args = {k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts}
        args["allow_fp8e4nv"] = self.capability >= 89
        args["allow_fp8e4b15"] = self.capability < 90
        args["max_num_imprecise_acc_default"] = 2**30 if self.capability == 90 else 0
        return CUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self):
        import triton.language.extra.cuda as cuda
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if self.capability >= 80 else cuda.convert_custom_float8_sm70
        }
        return codegen_fns

    def load_dialects(self, ctx):
        nvidia.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        cluster_info = nvidia.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        if capability // 10 >= 8:
            passes.ttgpuir.add_f32_dot_tc(pm)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm, capability)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.common.add_cse(pm)
        if capability // 10 >= 8:
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, opt.num_warps, opt.num_ctas, capability)
        if capability // 10 <= 8:
            passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if capability // 10 >= 9:
            nvidia.passes.ttnvgpuir.add_fence_insertion(pm)
            nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        passes.common.add_canonicalizer(pm)
        pm.run(mod)
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # add an perf collection pass
        passes.ttgpuir.add_perf_collection(pm)
        nvidia.passes.ttgpuir.add_decompose_unsupported_conversions(pm)
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)
        nvidia.passes.ttgpuir.add_to_llvmir(pm, capability)
        nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)
        pm.run(mod)
        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        nvidia.set_nvvm_reflect_ftz(llvm_mod)
        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        # Set kernel attributes
        # kernels = [fn for fn in llvm_mod.get_functions() if fn.has_public_visibility() and not fn.is_declaration()]
        # assert len(kernels) == 1
        # kernels[0].add_fn_attr("nvvm.maxntid", f"1, {options.num_warps*32}")
        # kernels[0].add_fn_attr("nvvm.kernel", "1")

        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_ptx(src, metadata, opt, capability):
        ptx_version = opt.ptx_version
        if ptx_version is None:
            _, cuda_version = _path_to_binary("ptxas")
            ptx_version = ptx_get_version(cuda_version)

        # PTX 8.3 is the max version supported by llvm 3a83162168.
        #
        # To check if a newer PTX version is supported, increase this value
        # and run a test.  If it's not supported, LLVM will print a warning
        # like "+ptx8.4 is not a recognized feature for this target".
        llvm_ptx_version = min(83, ptx_version)

        triple = 'nvptx64-nvidia-cuda'
        proc = 'sm_90a' if capability == 90 else f'sm_{capability}'
        features = f'+ptx{llvm_ptx_version}'
        ret = llvm.translate_to_asm(src, triple, proc, features, ['nvptx-short-ptr'], opt.enable_fp_fusion, False)
        # Find kernel names (there should only be one)
        names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]
        # post-process
        ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        # Remove the debug flag that prevents ptxas from optimizing the code
        ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        if os.environ.get("NVPTX_ENABLE_DUMP", "0") == "1":
            print("// -----// NVPTX Dump //----- //")
            print(ret)
        return ret

    @staticmethod
    def make_cubin(src, metadata, opt, capability):
        ptxas, _ = _path_to_binary("ptxas")
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            line_info = '' if os.environ.get('TRITON_DISABLE_LINE_INFO') else ' -lineinfo'
            fmad = '' if opt.enable_fp_fusion else ' --fmad=false'
            suffix = 'a ' if capability == 90 else ' '
            if os.environ.get("DISABLE_PTXAS_OPT", "0") == "1":
                cmd = f'{ptxas}{line_info}{fmad} -v --opt-level 0 --gpu-name=sm_{capability}{suffix}{fsrc.name} -o {fbin} 2> {flog.name}'
            else:
                cmd = f'{ptxas}{line_info}{fmad} -v --gpu-name=sm_{capability}{suffix}{fsrc.name} -o {fbin} 2> {flog.name}'

            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if e.returncode == 255:
                    raise RuntimeError(f'Internal Triton PTX codegen error: \n{log}')
                elif e.returncode == 128 + signal.SIGSEGV:
                    raise RuntimeError(
                        f'Please run `ptxas {fsrc.name}` to confirm that this is a bug in `ptxas`\n{log}')
                else:
                    raise RuntimeError(f'`ptxas` failed with error code {e.returncode}: \n{log}')
            finally:
                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.capability)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.capability)

    @functools.lru_cache()
    def hash(self):
        version = get_ptxas_version()
        return f'{version}-{self.capability}'
