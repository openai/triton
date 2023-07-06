import pytest
import torch

# import triton
# import triton.ops


@pytest.mark.parametrize("M, N, dtype, mode",
                         [
                             (M, N, dtype, mode) for M in [1024]
                             for N in [512]
                             for dtype in ['float16']
                             for mode in ['backward']
                         ]
                         )
def test_op(M, N, dtype, mode):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8 and dtype == "bfloat16":
        pytest.skip("Only test bfloat16 on devices with sm >= 80")
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[dtype]
    # create inputs
    x = torch.randn(M, N, dtype=dtype, device='cuda', requires_grad=True)
    idx = 4 + torch.ones(M, dtype=torch.int64, device='cuda')
    # forward pass
    # tt_y = triton.ops.cross_entropy(x, idx)
    th_y = torch.nn.CrossEntropyLoss(reduction="none")(x, idx)
    if mode == 'forward':
        torch.testing.assert_allclose(th_y, th_y)
    # backward pass
    elif mode == 'backward':
        dy = torch.randn_like(th_y)
        # triton backward
        th_y.backward(dy)
        tt_dx = x.grad.clone()
        # torch backward
        x.grad.zero_()
        th_y.backward(dy)
        th_dx = x.grad.clone()
        torch.testing.assert_allclose(th_dx, tt_dx)
