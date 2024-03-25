from __future__ import annotations
import hashlib
import json
from .._C.libtriton import get_env_vars, ir
from ..backends import backends
from .. import __version__
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager
from ..runtime.driver import driver
# TODO: this shouldn't be here
from dataclasses import dataclass
from .code_generator import ast_to_ttir
from pathlib import Path
import re
import functools
import os


@dataclass
class AttrsDescriptor:
    divisible_by_16: set = None
    equal_to_1: set = None

    def __post_init__(self):
        if self.divisible_by_16 is None:
            self.divisible_by_16 = set()
        if self.equal_to_1 is None:
            self.equal_to_1 = set()

    def to_dict(self):
        return {'divisible_by_16': list(self.divisible_by_16), 'equal_to_1': list(self.equal_to_1)}

    @staticmethod
    def from_dict(data):
        return AttrsDescriptor(divisible_by_16=set(data.get('divisible_by_16', [])),
                               equal_to_1=set(data.get('equal_to_1', [])))

    def hash(self):
        key = str([sorted(x) for x in self.__dict__.values()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


# - ^\s*tt\.func\s+ : match the start of the string, any leading whitespace, the keyword func,
#    and any following whitespace
# - (public\s+)? : optionally match the keyword public and any following whitespace
# - (@\w+) : match an @ symbol followed by one or more word characters
#   (letters, digits, or underscores), and capture it as group 1 (the function name)
# - (\((?:%\w+: \S+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\)) : match a pair of parentheses enclosing
#   zero or more arguments separated by commas, and capture it as group 2 (the argument list)
# - (attributes \{[\S\s]+\})? : optionally match attributes enclosed in braces and capture it as group 3
mlir_prototype_pattern = r"^\s*tt\.func\s+(?:public\s+)?(@\w+)(\((?:%\w+: [\S\s]+(?: \{\S+ = \S+ : \S+\})?(?:, )?)*\))\s*(attributes \{[\S\s]+\})?\s+\{\s*$"
ptx_prototype_pattern = r"\.(?:visible|extern)\s+\.(?:entry|func)\s+(\w+)\s*\(([^)]*)\)"
prototype_pattern = {
    "ttir": mlir_prototype_pattern,
    "ttgir": mlir_prototype_pattern,
    "ptx": ptx_prototype_pattern,
}

mlir_arg_type_pattern = r'%\w+: ((?:[^,\s<]+|<[^>]+>)+),?'
ptx_arg_type_pattern = r"\.param\s+\.(\w+)"
arg_type_pattern = {
    "ttir": mlir_arg_type_pattern,
    "ttgir": mlir_arg_type_pattern,
    "ptx": ptx_arg_type_pattern,
}


def convert_type_repr(x):
    # Currently we only capture the pointer type and assume the pointer is on global memory.
    # TODO: Capture and support shared memory space
    match = re.search(r'!tt\.ptr<([^,]+)', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


def _get_num_warps_from_ir_str(src: str):
    ttgir_num_warps_pattern = r'"triton_gpu.num-warps"\s?=\s?(\d+)\s?:'
    # TODO(jlebar): Using a regex to get num-warps is a hack, and will break if
    # e.g. someone has an instruction (not module) attribute named "num-warps".
    num_warps_matches = re.findall(ttgir_num_warps_pattern, src)
    assert len(num_warps_matches) == 1, "Expected exactly one match for num_warps"
    num_warps = int(num_warps_matches[0])
    return num_warps


class ASTSource:

    def __init__(self, fn, signature, constants=None, attrs=None) -> None:
        self.fn = fn
        self.ext = "ttir"
        self.name = fn.__name__
        self.signature = signature
        self.constants = constants
        self.attrs = attrs
        if isinstance(self.signature, str):
            self.signature = {k: v.strip() for k, v in enumerate(self.signature.split(","))}
        if self.constants is None:
            self.constants = dict()
        if self.attrs is None:
            self.attrs = AttrsDescriptor()

    def hash(self):
        sorted_sig = [v for k, v in sorted(self.signature.items())]
        sorted_constants = [(k, v) for k, v in sorted(self.constants.items())]
        key = f"{self.fn.cache_key}-{self.attrs.hash()}-{sorted_sig}-{sorted_constants}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def make_ir(self, options, context):
        return ast_to_ttir(self.fn, self, context=context, options=options)

    def parse_options(self):
        return dict()


class IRSource:

    def __init__(self, path):
        self.path = path
        path = Path(path)
        self.ext = path.suffix[1:]
        self.src = path.read_text()
        match = re.search(prototype_pattern[self.ext], self.src, re.MULTILINE)
        self.name = match.group(1)
        signature = match.group(2)
        types = re.findall(arg_type_pattern[self.ext], signature)
        self.signature = {k: convert_type_repr(ty) for k, ty in enumerate(types)}

    def hash(self):
        return hashlib.sha256(self.src.encode("utf-8")).hexdigest()

    def make_ir(self, options, context):
        module = ir.parse_mlir_module(self.path, context)
        module.context = context
        return module

    def parse_options(self):
        if self.ext == "ttgir":
            return {'num_warps': _get_num_warps_from_ir_str(self.src)}
        return dict()


@functools.lru_cache()
def triton_key():
    import pkgutil
    TRITON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    contents = []
    # frontend
    with open(__file__, "rb") as f:
        contents += [hashlib.sha256(f.read()).hexdigest()]
    # compiler
    compiler_path = os.path.join(TRITON_PATH, 'compiler')
    backends_path = os.path.join(TRITON_PATH, 'compiler', 'backends')
    for lib in pkgutil.iter_modules([compiler_path, backends_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.sha256(f.read()).hexdigest()]
    # backend
    libtriton_hash = hashlib.sha256()
    with open(os.path.join(TRITON_PATH, "_C/libtriton.so"), "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())
    # language
    language_path = os.path.join(TRITON_PATH, 'language')
    for lib in pkgutil.iter_modules([language_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, "rb") as f:
            contents += [hashlib.sha256(f.read()).hexdigest()]
    return f'{__version__}' + '-'.join(contents)


def parse(full_name, ext, context):
    if ext == "ttir" or ext == "ttgir":
        module = ir.parse_mlir_module(full_name, context)
        module.context = context
        return module
    if ext == "llir" or ext == "ptx":
        return Path(full_name).read_text()
    if ext == "cubin":
        return Path(full_name).read_bytes()


def filter_traceback(e: BaseException):
    """
    Removes code_generator.py and related files from tracebacks.

    These are uninteresting to the user -- "just show me *my* code!"
    """
    if e.__cause__ is not None:
        filter_traceback(e.__cause__)
    if e.__context__ is not None:
        filter_traceback(e.__context__)

    # If a user has a file that matches one of these, they're out of luck.
    BAD_FILES = [
        "/triton/compiler/code_generator.py",
        "/ast.py",
    ]

    tb = e.__traceback__
    frames = []
    while tb is not None:
        if not any(f for f in BAD_FILES if tb.tb_frame.f_code.co_filename.endswith(f)):
            frames.append(tb)
        tb = tb.tb_next

    for (cur_frame, next_frame) in zip(frames, frames[1:]):
        cur_frame.tb_next = next_frame

    if not frames:
        e.__traceback__ = None
    else:
        frames[-1].tb_next = None
        e.__traceback__ = frames[0]


def compile(src, target=None, options=None):
    if target is None:
        target = driver.active.get_current_target()
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
    # create backend
    if ir_source:
        assert isinstance(src, str), "source must be either AST or a filepath"
        src = IRSource(src)
    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))
    # create cache manager
    key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(get_env_vars().items()))}"
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    # For dumping/overriding only hash the source as we want it to be independent of triton
    # core changes to make it easier to track kernels by hash.
    enable_override = os.environ.get("TRITON_KERNEL_OVERRIDE", "0") == "1"
    enable_ir_dump = os.environ.get("TRITON_KERNEL_DUMP", "0") == "1"
    fn_override_manager = get_override_manager(src.hash()) if enable_override else None
    fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
    metadata_filename = f"{src.name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    if metadata_path is not None:
        # cache hit!
        metadata = json.loads(Path(metadata_path).read_text())
        return CompiledKernel(src, metadata_group, hash)
    # initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **get_env_vars(),
    }
    # run compilation pipeline  and populate metadata
    stages = dict()
    backend.add_stages(stages, options)
    first_stage = list(stages.keys()).index(src.ext)
    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    if ir_source:
        first_stage += 1
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    try:
        module = src.make_ir(options, context)
    except Exception as e:
        filter_traceback(e)
        raise
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{src.name}.{ext}"
        metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)
        if fn_dump_manager is not None:
            fn_dump_manager.put(next_module, ir_filename)
        if (fn_override_manager is not None and fn_override_manager.has_file(ir_filename)):
            print(f"\nOverriding kernel with file {ir_filename}")
            full_name = fn_override_manager.get_file(ir_filename)
            next_module = parse(full_name, ext, context)
        # use an env variable to parse ttgir from file
        use_ttgir_loc = os.environ.get("USE_TTGIR_LOC", "0") == "1"
        if use_ttgir_loc and ext == "ttgir":
            ttgir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module = parse(ttgir_full_name, ext, context)
            print(f"re-parse ttgir with {ttgir_full_name}")
        module = next_module
    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)
    # return handle to compiled kernel
    return CompiledKernel(src, metadata_group, hash)


def make_backend(target):
    actives = [x.compiler for x in backends.values() if x.compiler.supports_target(target)]
    if len(actives) != 1:
        raise RuntimeError(
            f"{len(actives)} compatible backends for target ({target[0]}) ({actives}). There should only be one.")
    return actives[0](target)


class CompiledKernel:

    # Hooks for external tools to monitor the execution of triton kernels
    # TODO: move out of this namespace since it's a runtime thing
    launch_enter_hook = None
    launch_exit_hook = None

    def __init__(self, src, metadata_group, hash):
        from collections import namedtuple
        metadata_path = next((Path(p) for c, p in metadata_group.items() if c.endswith(".json")))
        self.metadata = json.loads(metadata_path.read_text())
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(self.metadata.keys())))
        self.metadata = KernelMetadata(**self.metadata)
        self.src = src
        self.hash = hash

        self.name = self.metadata.name
        # stores the text of each level of IR that was generated during compilation
        asm_files = [Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
        binary_ext = make_backend(self.metadata.target).binary_ext
        self.asm = {
            file.suffix[1:]: file.read_bytes() if file.suffix[1:] == binary_ext else file.read_text()
            for file in asm_files
        }
        self.kernel = self.asm[binary_ext]
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.module = None
        self.function = None

    def _init_handles(self):
        if self.module is not None:
            return
        device = driver.active.get_current_device()
        # create launcher
        self.run = driver.active.launcher_cls(self.src, self.metadata)
        # not enough shared memory to run the kernel
        max_shared = driver.active.utils.get_device_properties(device)["max_shared_mem"]
        if self.metadata.shared > max_shared:
            raise OutOfResources(self.metadata.shared, max_shared, "shared memory")
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills = driver.active.utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device)

    def __getattribute__(self, name):
        if name == 'run':
            self._init_handles()
        return super().__getattribute__(name)

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                device = driver.active.get_current_device()
                stream = driver.active.get_current_stream(device)
            md = self.metadata
            self.run(grid[0], grid[1], grid[2], md.num_warps, md.num_ctas, md.cluster_dims[0], md.cluster_dims[1],
                     md.cluster_dims[2], md.shared, stream, self.function, CompiledKernel.launch_enter_hook,
                     CompiledKernel.launch_exit_hook, md, *args)

        return runner
