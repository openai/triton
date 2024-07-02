import itertools
import pytest
import torch

import triton
import triton.language as tl


def test_pre_call_hooks(device):

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    class MyTensor(torch.Tensor):
        pass

    def my_hook(*args, **kwargs):
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, MyTensor):
                raise Exception("MyTensor is not allowed")

    add_kernel.add_pre_run_hook(my_hook)

    x = torch.randn(4, device=device)
    y = MyTensor(x)
    out = torch.zeros_like(x)
    with pytest.raises(Exception):
        add_kernel[(4, )](x, y, out, 4, 4)


def test_pre_call_hooks_with_additional_args(device):

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    def my_hook(*args, **kwargs):
        if kwargs["grid"] == (4, ):
            raise RuntimeError("grid (4,) is not allowed")

    add_kernel.add_pre_run_hook(my_hook)

    x = torch.randn(4, device=device)
    y = torch.Tensor(x)
    out = torch.zeros_like(x)
    with pytest.raises(RuntimeError):
        add_kernel[(4, )](x, y, out, 4, 4)
