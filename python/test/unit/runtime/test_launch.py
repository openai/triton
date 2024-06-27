import gc
# import importlib
# import os
# import sys
# import tempfile
# import textwrap
# import time
import tracemalloc

import torch

import triton
import triton.language as tl

# from typing import Tuple


def test_grid() -> None:

    def grid(META):
        assert "num_warps" in META
        assert "n_regs" in META
        assert "size_smem" in META
        return (1, 1)

    @triton.jit
    def kernel(x):
        pass

    # launch kernel
    kernel[grid](6)


def test_metadata() -> None:

    used_hook = False

    def _launch_metadata(grid, kernel, args):
        ret = dict()
        ret["grid"] = grid
        ret["value"] = args["x"]
        return ret

    def hook(launch_metadata):
        nonlocal used_hook
        metadata = launch_metadata.get()
        assert metadata["grid"] == (1, 3, 2)
        assert metadata["value"] == 6
        used_hook = True

    @triton.jit(launch_metadata=_launch_metadata)
    def kernel(x):
        pass

    # launch kernel
    triton.compiler.CompiledKernel.launch_enter_hook = hook
    kernel[(1, 3, 2)](6)
    triton.compiler.CompiledKernel.launch_enter_hook = None
    assert used_hook


def test_memory_leak() -> None:

    @triton.jit
    def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
        xnumel = 10
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)

    tracemalloc.start()
    try:
        inp = torch.randn(10, device='cuda')
        out = torch.randn(10, device='cuda')
        kernel[(10, )](inp, out, 10, XBLOCK=16)
        gc.collect()
        begin, _ = tracemalloc.get_traced_memory()
        for _ in range(100):
            kernel[(10, )](inp, out, 10, XBLOCK=16)
        gc.collect()
        end, _ = tracemalloc.get_traced_memory()
        assert end - begin < 30000
    finally:
        tracemalloc.stop()


# LATENCY_THRESHOLD_US = 46

# def test_kernel_launch_latency() -> None:
#     def define_kernel(kernel_name: str, num_tensor_args: int) -> str:
#         arg_str = ",".join([f"arg{i}: torch.Tensor" for i in range(num_tensor_args)])
#         arg_str += ", n_elements: int, BLOCK_SIZE: tl.constexpr"
#         func_str = f"""
#         import torch

#         import triton
#         import triton.language as tl

#         @triton.jit
#         def {kernel_name}({arg_str}):
#             pass
#         """
#         with tempfile.NamedTemporaryFile(mode="w+t", suffix=".py", delete=False) as temp_file:
#             temp_file.write(textwrap.dedent(func_str))
#             temp_file_path = temp_file.name

#         return temp_file_path

#     def import_kernel(file_path, kernel_name):
#         directory, filename = os.path.split(file_path)
#         module_name, _ = os.path.splitext(filename)
#         sys.path.insert(0, directory)

#         module = importlib.import_module(module_name)
#         kernel = getattr(module, kernel_name)
#         return kernel

#     def empty(*kernel_args: Tuple[torch.Tensor]):
#         first_arg = kernel_args[0]
#         n_elements = first_arg.numel()
#         grid = (triton.cdiv(n_elements, 1024),)
#         device = torch.cuda.current_device()
#         # Warmup
#         empty_kernel[grid](*kernel_args, n_elements, BLOCK_SIZE=1024, device=device)
#         torch.cuda.synchronize()
#         # Measure launch overhead at steady state
#         num_runs = 1000
#         start_time = time.time()
#         for i in range(num_runs):
#             empty_kernel[grid](*kernel_args, n_elements, BLOCK_SIZE=1024, device=device)
#         end_time = time.time()
#         latency_us = (end_time - start_time) / num_runs * 1e6

#         assert latency_us < LATENCY_THRESHOLD_US, "Kernel launch time has increased!"

#     num_tensor_args = 40
#     kernel_name = 'empty_kernel'
#     file_path = define_kernel(kernel_name, num_tensor_args)
#     empty_kernel = import_kernel(file_path, kernel_name)

#     # Initialize random tensors for the empty_kernel
#     torch.manual_seed(0)
#     size = 1024
#     kernel_args = (torch.rand(size, device='cuda') for i in range(num_tensor_args))

#     # Run empty, which would run empty_kernel internally
#     empty(*kernel_args)
