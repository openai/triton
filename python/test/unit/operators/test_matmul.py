import itertools

import pytest
import torch

import triton
import triton.language as tl
import triton.ops


@pytest.mark.parametrize(
    "BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, NSTAGE, M, N, K, AT, BT, ADTYPE, BDTYPE, INPUT_PRECISION, F8_FASTACCUM, ACC_DTYPE, OUTPUT_DTYPE",
    itertools.chain(
        *[[
            # 1 warp
            (16, 16, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (32, 16, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (16, 32, 16, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (16, 16, 32, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (32, 16, 32, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (16, 32, 32, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (16, 16, 64, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (64, 16, 64, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (16, 64, 64, 1, 1, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            # 2 warp
            (64, 32, 64, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (32, 64, 64, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (64, 32, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (32, 64, 16, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (128, 32, 32, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (32, 128, 32, 1, 2, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            # 4 warp
            (128, 64, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (64, 128, 16, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (128, 32, 32, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (32, 128, 32, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (128, 32, 64, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (32, 128, 64, 1, 4, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            # 8 warp
            (128, 256, 16, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (256, 128, 16, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (256, 128, 32, 1, 8, 2, None, None, None, AT, BT, DTYPE, DTYPE, None, True, None, None),
            # variable input
            (128, 128, 32, 1, 4, 2, 256, 384, 160, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (128, 128, 32, 1, 4, 2, 107, 233, 128, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (128, 128, 32, 1, 4, 2, 107, 233, 83, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (128, 256, 64, 1, 8, 3, 256, 512, 160, AT, BT, DTYPE, DTYPE, None, True, None, None),
        ] for DTYPE in ["float16", "bfloat16", "float32"] for AT in [False, True] for BT in [False, True]],
        # n-stage
        *[[
            (16, 16, 16, 1, 1, STAGES, 32, 32, 80, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (64, 32, 64, 1, 2, STAGES, 128, 64, 128, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (128, 64, 16, 1, 4, STAGES, 256, 128, 80, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (256, 128, 32, 1, 8, STAGES, 512, 256, 160, AT, BT, DTYPE, DTYPE, None, True, None, None),
            (128, 128, 32, 1, 4, STAGES, 256, 256, 160, AT, BT, DTYPE, DTYPE, None, True, None, None),
        ]
          for DTYPE in ["float16", "bfloat16", "float32"]
          for AT in [False, True]
          for BT in [False, True]
          for STAGES in [4]],
        # tf32x3
        *[[
            (16, 16, 16, 1, 1, 2, 32, 32, 80, AT, BT, "float32", "float32", "tf32x3", True, None, None),
            (64, 32, 64, 1, 2, 2, 128, 64, 128, AT, BT, "float32", "float32", "tf32x3", True, None, None),
            (128, 64, 16, 1, 4, 2, 256, 128, 80, AT, BT, "float32", "float32", "tf32x3", True, None, None),
            (256, 128, 32, 1, 8, 2, 512, 256, 160, AT, BT, "float32", "float32", "tf32x3", True, None, None),
            (128, 128, 32, 1, 4, 2, 256, 256, 160, AT, BT, "float32", "float32", "tf32x3", True, None, None),
        ] for AT in [False, True] for BT in [False, True]],
        # mixed-precision
        *[[
            (32, 32, 32, 1, 1, 2, None, None, None, AT, BT, ADTYPE, BDTYPE, None, FASTACCUM, None, None),
            (128, 256, 32, 1, 8, 2, None, None, None, AT, BT, ADTYPE, BDTYPE, None, FASTACCUM, None, None),
            (32, 64, 32, 1, 1, 2, 64, 128, 32, AT, BT, ADTYPE, BDTYPE, None, FASTACCUM, None, None),
        ] for ADTYPE, BDTYPE in [
            ("float8e4nv", "float8e5"),
            ("float8e4nv", "float8e4nv"),
            ("float8e5", "float8e4nv"),
            ("float8e5", "float8e5"),
            ("float8e4b15", "float8e4b15"),
            ("float8e4nv", "float16"),
            ("float16", "float8e5"),
            ("int8", "bfloat16"),
            ("float16", "int8"),
            ("float16", "float32"),
            ("float32", "float16"),
            ("bfloat16", "float32"),
            ("float32", "bfloat16"),
        ] for AT in [False, True] for BT in [False, True] for FASTACCUM in [True, False]],
        # mixed-precision block layout
        *[[
            (32, 32, 32, 1, 1, 2, None, None, None, AT, BT, ADTYPE, BDTYPE, None, True, None, None),
            (128, 256, 32, 1, 8, 2, None, None, None, AT, BT, ADTYPE, BDTYPE, None, True, None, None),
            (32, 64, 32, 1, 1, 2, 64, 128, 32, AT, BT, ADTYPE, BDTYPE, None, True, None, None),
        ] for ADTYPE, BDTYPE in [
            ("float8e4nv", "float16"),
            ("float16", "float8e5"),
            ("float16", "float32"),
            ("float32", "float16"),
            ("bfloat16", "float32"),
            ("float32", "bfloat16"),
        ] for AT in [False, True] for BT in [False, True]],
        # acc-out-dtype and output_dtype
        *[[
            (32, 32, 32, 1, 1, 2, None, None, None, False, False, "float16", "float16", None, True, ACC_DTYPE,
             OUTPUT_DTYPE),
            (128, 256, 32, 1, 8, 2, None, None, None, False, False, "float16", "float16", None, True, ACC_DTYPE,
             OUTPUT_DTYPE),
        ] for ACC_DTYPE in [None, "float16", "float32"] for OUTPUT_DTYPE in [None, "float16", "float32"]],
    ),
)
def test_op(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, NWARP, NSTAGE, M, N, K, AT, BT, ADTYPE, BDTYPE, INPUT_PRECISION,
            F8_FASTACCUM, ACC_DTYPE, OUTPUT_DTYPE):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:
        pytest.skip("Only test tl.dot() on devices with sm >= 70")
    if capability[0] < 8 and (ADTYPE == "bfloat16" or BDTYPE == "bfloat16"):
        pytest.skip("Only test bfloat16 on devices with sm >= 80")
    if capability[0] < 9 and (ADTYPE == "float8e4nv" or BDTYPE == "float8e4nv"):
        pytest.skip("Only test float8e4nv on devices with sm >= 90")
    if (ADTYPE == "bfloat16" or BDTYPE == "bfloat16") and SPLIT_K != 1:
        pytest.skip("bfloat16 matmuls don't allow split_k for now")
    torch.manual_seed(0)
    # nuke kernel decorators -- will set meta-parameters manually
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'BLOCK_K': BLOCK_K, 'SPLIT_K': SPLIT_K}
    pre_hook = None if SPLIT_K == 1 else lambda nargs: nargs['C'].zero_()
    configs = [triton.Config(kwargs=kwargs, num_warps=NWARP, num_stages=NSTAGE, pre_hook=pre_hook)]
    kernel = triton.ops._matmul.kernel
    kernel.configs = configs
    # kernel.run = kernel.run.run.run

    # get matrix shape
    M = BLOCK_M if M is None else M
    N = BLOCK_N if N is None else N
    K = BLOCK_K * SPLIT_K if K is None else K

    def is_fp8(dtype):
        return "float8" in dtype

    def f8_to_f16(x, dtype):

        @triton.jit
        def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(X + offs, mask=mask)
            tl.store(Y + offs, x, mask=mask)

        ret = torch.empty_strided(x.shape, x.stride(), dtype=torch.float16, device=x.device)
        grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
        dtype = getattr(tl, dtype)
        kernel[grid](ret, triton.reinterpret(x, dtype), ret.numel(), BLOCK_SIZE=1024)
        return ret

    def upcast_if_fp8(x, dtype):
        if is_fp8(dtype):
            return f8_to_f16(x, dtype)
        return x

    def init_input(m, n, dtype, acc_dtype):
        if 'float8' in dtype:
            ewidth = {'float8e4b15': 4, 'float8e4nv': 4, 'float8e5': 5}[dtype]
            sign = torch.randint(2, size=(m, n), device="cuda", dtype=torch.int8) * 128
            val = torch.randint(2**3 - 1, size=(m, n), device="cuda", dtype=torch.int8) << 7 - ewidth
            return sign | val
        if dtype == "int8":
            return torch.randint(-128, 127, (m, n), device="cuda", dtype=torch.int8)
        # Use small range of values to prevent numerical issues.
        min_exp = -4 if acc_dtype == "float16" else -10
        exponents = torch.randint(min_exp, 0, size=(m, n))
        ret = (2.**exponents).to(getattr(torch, dtype)).to("cuda")
        return ret

    # allocate/transpose inputs
    a = init_input(M, K, ADTYPE, ACC_DTYPE)
    b = init_input(K, N, BDTYPE, ACC_DTYPE)
    a = a if not AT else a.T.contiguous().T
    b = b if not BT else b.T.contiguous().T
    # run test
    th_a = upcast_if_fp8(a, ADTYPE)
    th_b = upcast_if_fp8(b, BDTYPE)
    ab_dtype = triton.ops.get_higher_dtype(th_a.dtype, th_b.dtype)
    acc_dtype = getattr(torch, ACC_DTYPE) if ACC_DTYPE else ab_dtype
    output_dtype = getattr(torch, OUTPUT_DTYPE) if OUTPUT_DTYPE else ab_dtype
    th_c = torch.matmul(th_a.to(output_dtype), th_b.to(output_dtype))
    try:
        if is_fp8(ADTYPE):
            a = triton.reinterpret(a, getattr(tl, ADTYPE))
        if is_fp8(BDTYPE):
            b = triton.reinterpret(b, getattr(tl, BDTYPE))
        tt_c = triton.ops.matmul(a, b, acc_dtype if ACC_DTYPE else None, INPUT_PRECISION, F8_FASTACCUM, output_dtype)
        torch.testing.assert_close(th_c, tt_c)
    except triton.OutOfResources as e:
        pytest.skip(str(e))
