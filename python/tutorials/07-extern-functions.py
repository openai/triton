"""
Libdevice (`tl.extra.libdevice`) function
==============================
Triton can invoke a custom function from an external library.
In this example, we will use the `libdevice` library to apply `asin` on a tensor.
Please refer to https://docs.nvidia.com/cuda/libdevice-users-guide/index.html (CUDA) and/or https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs/ocml/src (HIP) regarding the semantics of all available libdevice functions.
In `libdevice.py`, we try to aggregate functions with the same computation but different data types together.
For example, both `__nv_asin` and `__nv_asinf` calculate the principal value of the arc sine of the input, but `__nv_asin` operates on `double` and `__nv_asinf` operates on `float`.
Triton automatically selects the correct underlying device function to invoke based on input and output types.
"""

# %%
#  asin Kernel
# ------------

import os
import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice

from pathlib import Path


@triton.jit
def asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


# %%
#  Using the default libdevice library path
# -----------------------------------------
# We can use the default libdevice library path encoded in `triton/language/math.py`

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
output_triton = torch.zeros(size, device='cuda')
output_torch = torch.asin(x)
assert x.is_cuda and output_triton.is_cuda
n_elements = output_torch.numel()
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

# %%
#  Customize the libdevice library path
# -------------------------------------
# We can also customize the libdevice library path by passing the path to the `libdevice` library to the `asin` kernel.
# This example uses the path from the `third_party/nvidia/backend/lib` directory (the default path).

libdir = Path(__file__).parent.parent.parent / 'third_party/nvidia/backend/lib'
extern_libs = {}
extern_libs['libdevice'] = os.getenv("TRITON_LIBDEVICE_PATH", str(libdir / 'libdevice.10.bc'))

output_triton = torch.empty_like(x)
asin_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=1024, extern_libs=extern_libs)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
