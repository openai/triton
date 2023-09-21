# flake8: noqa: F821,F841
import itertools
import os
import re
from typing import Optional, Union

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton._C.libtriton.triton as _triton
import triton.language as tl
from triton.common.build import is_hip
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']

# TODO: enable multiple cta cluster testing.
# num_ctas_list = [1, 4] if torch.cuda.get_device_capability()[0] == 9 else [1]
num_ctas_list = [1]

if is_hip():
    GPU_DIALECT = "triton_gpu_rocm"
    THREADS_PER_WARP = 64
else:
    GPU_DIALECT = "triton_gpu"
    THREADS_PER_WARP = 32


def _bitwidth(dtype: str) -> int:
    # ex.: "int64" -> 64
    return int(re.search(r'(\d+)$', dtype).group(1))


def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Workaround. Never return zero so tests of division don't error out.
        return x
    elif dtype_str and 'float8' in dtype_str:
        x = rs.randint(20, 40, shape, dtype=np.int8)
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32')
                & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')


def to_triton(x: np.ndarray, device='cuda', dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if dst_type and 'float8' in dst_type:
            return reinterpret(torch.tensor(x, device=device), getattr(tl, dst_type))
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)


def torch_dtype_name(dtype) -> str:
    if isinstance(dtype, triton.language.dtype):
        return dtype.name
    elif isinstance(dtype, torch.dtype):
        # 'torch.int64' -> 'int64'
        m = re.match(r'^torch\.(\w+)$', str(dtype))
        return m.group(1)
    else:
        raise TypeError(f'not a triton or torch dtype: {type(dtype)}')


def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel


def check_cuda_only(device):
    if device not in ['cuda']:
        pytest.skip("Only for cuda")


def check_type_supported(dtype, device):
    '''
    skip test if dtype is not supported on the current device
    '''
    if device in ['cuda']:
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
            pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")
        if cc[0] < 9 and (dtype is tl.float8e4nv or dtype == "float8e4nv"):
            pytest.skip("float8e4nv is only supported on NVGPU with cc >= 90")


class MmaLayout:
    def __init__(self, version, warps_per_cta, ctas_per_cga, cta_split_num, cta_order, instr_shape):
        self.version = version
        self.warps_per_cta = str(warps_per_cta)
        self.ctas_per_cga = str(ctas_per_cga)
        self.cta_split_num = str(cta_split_num)
        self.cta_order = str(cta_order)
        self.instr_shape = str(instr_shape)

    def __str__(self):
        return f"#{GPU_DIALECT}.mma<{{versionMajor={self.version[0]}, versionMinor={self.version[1]}, warpsPerCTA={self.warps_per_cta}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}, instrShape={self.instr_shape}}}>"


class BlockedLayout:
    def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order, ctas_per_cga, cta_split_num, cta_order):
        self.sz_per_thread = str(size_per_thread)
        self.threads_per_warp = str(threads_per_warp)
        self.warps_per_cta = str(warps_per_cta)
        self.order = str(order)
        self.ctas_per_cga = str(ctas_per_cga)
        self.cta_split_num = str(cta_split_num)
        self.cta_order = str(cta_order)

    def __str__(self):
        return f"#{GPU_DIALECT}.blocked<{{sizePerThread={self.sz_per_thread}, threadsPerWarp={self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


class SharedLayout:
    def __init__(self, vec, per_phase, max_phase, order, ctas_per_cga, cta_split_num, cta_order):
        self.vec = str(vec)
        self.per_phase = str(per_phase)
        self.max_phase = str(max_phase)
        self.order = str(order)
        self.ctas_per_cga = str(ctas_per_cga)
        self.cta_split_num = str(cta_split_num)
        self.cta_order = str(cta_order)

    def __str__(self):
        return f"#{GPU_DIALECT}.shared<{{vec={self.vec}, perPhase={self.per_phase}, maxPhase={self.max_phase}, order={self.order}, CTAsPerCGA={self.ctas_per_cga}, CTASplitNum={self.cta_split_num}, CTAOrder={self.cta_order}}}>"


@pytest.mark.parametrize("dtype_x", list(dtypes) + ["bfloat16"])
def test_empty_kernel(dtype_x, device):
    SIZE = 128

    @triton.jit
    def kernel(X, SIZE: tl.constexpr):
        pass
    check_type_supported(dtype_x, device)
    x = to_triton(numpy_random(SIZE, dtype_str=dtype_x), device=device, dst_type=dtype_x)
    kernel[(1, )](x, SIZE=SIZE, num_warps=4)


# generic test functions
def _test_unary(dtype_x, expr, numpy_expr=None, device='cuda', num_ctas=1):
    check_type_supported(dtype_x, device)  # early return if dtype_x is not supported
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    x = numpy_random(SIZE, dtype_str=dtype_x)
    if 'log' in expr:
        x = np.abs(x) + 0.01
    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    z_tri = to_triton(np.empty_like(z_ref), device=device, dst_type=dtype_x)
    kernel[(1, )](z_tri, x_tri, SIZE=SIZE, num_warps=4, num_ctas=num_ctas)
    # compare
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)


def _binary_op_dtype_override(a: str, b: str) -> Optional[np.dtype]:
    """
    Given two dtype strings, returns the numpy dtype Triton thinks binary
    operations on the two types should return. Returns None if the return value
    matches numpy. This is generally needed because Triton and pytorch return
    narrower floating point types than numpy in mixed operations, and because
    Triton follows C/C++ semantics around mixed signed/unsigned operations, and
    numpy/pytorch do not.
    """
    overrides = {
        ('float16', 'int16'): np.float16,
        ('float16', 'int32'): np.float16,
        ('float16', 'int64'): np.float16,
        ('float16', 'uint16'): np.float16,
        ('float16', 'uint32'): np.float16,
        ('float16', 'uint64'): np.float16,
        ('int8', 'uint8'): np.uint8,
        ('int8', 'uint16'): np.uint16,
        ('int8', 'uint32'): np.uint32,
        ('int8', 'uint64'): np.uint64,
        ('int16', 'uint16'): np.uint16,
        ('int16', 'uint32'): np.uint32,
        ('int16', 'uint64'): np.uint64,
        ('int32', 'uint32'): np.uint32,
        ('int32', 'uint64'): np.uint64,
        ('int64', 'uint64'): np.uint64,
    }
    key = (a, b) if a < b else (b, a)
    return overrides.get(key)


def _test_binary(dtype_x, dtype_y, expr, numpy_expr=None, mode_x='real', mode_y='real', device='cuda', num_ctas=1, y_low=None, y_high=None):
    check_type_supported(dtype_x, device)  # early return if dtype_x is not supported
    check_type_supported(dtype_y, device)
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(Z, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        z = GENERATE_TEST_HERE
        tl.store(Z + off, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': expr})
    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs, low=y_low, high=y_high)
    if mode_x == 'nan':
        x[:] = float('nan')
    if mode_y == 'nan':
        y[:] = float('nan')
    # reference result
    z_ref = eval(expr if numpy_expr is None else numpy_expr)
    dtype_z = _binary_op_dtype_override(dtype_x, dtype_y)
    if dtype_z is not None:
        z_ref = z_ref.astype(dtype_z)
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    y_tri = to_triton(y, device=device, dst_type=dtype_y)
    z_tri = to_triton(np.empty(SIZE, dtype=z_ref.dtype), device=device)
    kernel[(1, )](z_tri, x_tri, y_tri, SIZE=SIZE,
                  num_warps=4, num_ctas=num_ctas)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), err_msg=expr, rtol=0.01)


def _mod_operation_ill_conditioned(dtype_x, dtype_y) -> bool:
    # The result of x % y is ill-conditioned if x % y is much smaller than x.
    # pytorch/CUDA has slightly different (probably better) rounding on
    # remainders than stock LLVM. We currently don't expect to match it
    # bit-for-bit.
    return (dtype_x, dtype_y) in [
        ('int32', 'bfloat16'),
        ('int32', 'float16'),
        ('int32', 'float32'),
        ('int64', 'bfloat16'),
        ('int64', 'float16'),
        ('int64', 'float32'),
        ('int64', 'float64'),
        ('uint16', 'bfloat16'),
        ('uint16', 'float16'),
        ('uint16', 'float32'),
        ('uint32', 'bfloat16'),
        ('uint32', 'float16'),
        ('uint32', 'float32'),
        ('uint64', 'bfloat16'),
        ('uint64', 'float16'),
        ('uint64', 'float32'),
        ('uint64', 'float64'),
    ]

# ---------------
# test binary ops
# ---------------


@pytest.mark.parametrize("dtype_x, dtype_y, op", [
    (dtype_x, dtype_y, op)
    for op in ['+', '-', '*', '/', '%']
    for dtype_x in dtypes_with_bfloat16
    for dtype_y in dtypes_with_bfloat16
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_bin_op(dtype_x, dtype_y, op, num_ctas, device):
    expr = f' x {op} y'
    if op == '%' and dtype_x in int_dtypes + uint_dtypes and dtype_y in int_dtypes + uint_dtypes:
        # LLVM has 'numpy.fmod', not 'numpy.remainder', semantics on integer remainders.
        numpy_expr = 'np.fmod(x, y)'
    elif op in ('/', '%') and dtype_x in ('int16', 'float16', 'bfloat16') and dtype_y in ('int16', 'float16', 'bfloat16'):
        # Triton promotes 16-bit floating-point / and % to 32-bit because there
        # are no native div or FRem operations on float16. Since we have to
        # convert anyway, we may as well take the accuracy bump.
        numpy_expr = f'x.astype(np.float32) {op} y.astype(np.float32)'
    elif (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    if op == '%' and _mod_operation_ill_conditioned(dtype_x, dtype_y):
        with pytest.raises(AssertionError, match='Not equal to tolerance'):
            _test_binary(
                dtype_x,
                dtype_y,
                expr,
                numpy_expr,
                device=device,
                num_ctas=num_ctas)
    elif (op in ('%', '/') and
          ((dtype_x in int_dtypes and dtype_y in uint_dtypes) or
           (dtype_x in uint_dtypes and dtype_y in int_dtypes))):
        with pytest.raises(triton.CompilationError) as exc_info:
            _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas)
        assert re.match('Cannot use .* because they have different signedness', str(exc_info.value.__cause__))
    else:
        _test_binary(
            dtype_x,
            dtype_y,
            expr,
            numpy_expr,
            device=device,
            num_ctas=num_ctas)


@pytest.mark.parametrize("dtype, order", [(dtype, order) for dtype in dtypes_with_bfloat16 for order in [0, 1]])
def test_addptr(dtype, order, device):
    check_type_supported(dtype, device)

    @triton.jit
    def kernel(x, y, ORDER: tl.constexpr, SIZE: tl.constexpr):
        offs = tl.arange(0, SIZE)
        if ORDER == 0:
            tl.store(y + offs, tl.load(x + offs))
        else:
            tl.store(offs + y, tl.load(offs + x))

    SIZE = 1024
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    x_tri = to_triton(x, dst_type=dtype, device=device)
    y_tri = to_triton(y, dst_type=dtype, device=device)
    y = x
    kernel[1,](x_tri, y_tri, order, SIZE)
    np.testing.assert_allclose(y, to_numpy(y_tri))


@pytest.mark.parametrize("dtype_x, dtype_y",
                         [(dtype_x, dtype_y) for dtype_x in int_dtypes for dtype_y in int_dtypes] +
                         [(dtype_x, dtype_y) for dtype_x in uint_dtypes for dtype_y in uint_dtypes]
                         )
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_floordiv(dtype_x, dtype_y, num_ctas, device):
    # Triton has IEEE, not numpy/torch, semantics for %, and those carry
    # through to //, so we have to use a nonstandard expression to get a
    # reference result for //.
    expr = 'x // y'
    numpy_expr = '((x - np.fmod(x, y)) / y)'
    _test_binary(
        dtype_x,
        dtype_y,
        expr,
        numpy_expr,
        device=device,
        num_ctas=num_ctas)


def test_unsigned_name_mangling(device='cuda'):
    # Test that uint32 and int32 are mangled differently by the compiler
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(O1, O2, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        out1 = tl.abs(x)  # uint32 -> nop
        out2 = tl.abs(-y)  # int32 -> should have an effect
        tl.store(O1 + off, out1)
        tl.store(O2 + off, out2)

    dtype_x = 'uint32'
    dtype_y = 'int32'
    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs)
    # reference result
    expect = (np.abs(x), np.abs(-y))
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    y_tri = to_triton(y, device=device, dst_type=dtype_y)
    actual = tuple(
        to_triton(np.empty_like(e), device=device)
        for e in expect
    )
    kernel[(1, )](actual[0], actual[1], x_tri, y_tri, SIZE=SIZE, num_warps=4)

    # Bitwise op, so expect exact equality
    assert (expect[0] == to_numpy(actual[0])).all()
    assert (expect[1] == to_numpy(actual[1])).all()


def test_unsigned_name_mangling(device):
    # Test that uint32 and int32 are mangled differently by the compiler
    SIZE = 128
    # define the kernel / launch-grid

    @triton.jit
    def kernel(O1, O2, X, Y, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        y = tl.load(Y + off)
        out1 = tl.abs(x)  # uint32 -> nop
        out2 = tl.abs(-y)  # int32 -> should have an effect
        tl.store(O1 + off, out1)
        tl.store(O2 + off, out2)

    dtype_x = 'uint32'
    dtype_y = 'int32'
    # inputs
    rs = RandomState(17)
    x = numpy_random(SIZE, dtype_str=dtype_x, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype_y, rs=rs)
    # reference result
    expect = (np.abs(x), np.abs(-y))
    # triton result
    x_tri = to_triton(x, device=device, dst_type=dtype_x)
    y_tri = to_triton(y, device=device, dst_type=dtype_y)
    actual = tuple(
        to_triton(np.empty_like(e), device=device)
        for e in expect
    )
    kernel[(1, )](actual[0], actual[1], x_tri, y_tri, SIZE=SIZE, num_warps=4)

    # Bitwise op, so expect exact equality
    assert (expect[0] == to_numpy(actual[0])).all()
    assert (expect[1] == to_numpy(actual[1])).all()


# ---------------
# test bitwise ops
# ---------------
@pytest.mark.parametrize("dtype_x, dtype_y, op", [
    (dtype_x, dtype_y, op)
    for op in ['&', '|', '^']
    for dtype_x in dtypes + dtypes_with_bfloat16
    for dtype_y in dtypes + dtypes_with_bfloat16
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_bitwise_op(dtype_x, dtype_y, op, num_ctas, device):
    expr = f'x {op} y'
    if (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    if 'float' in dtype_x + dtype_y:
        with pytest.raises(triton.CompilationError) as exc_info:
            _test_binary(dtype_x, dtype_y, expr, numpy_expr='np.array([])', device=device, num_ctas=num_ctas)
        # The CompilationError must have been caused by a C++ exception with this text.
        assert re.match('invalid operands of type', str(exc_info.value.__cause__))
    else:
        _test_binary(
            dtype_x,
            dtype_y,
            expr,
            numpy_expr,
            device=device,
            num_ctas=num_ctas)


@pytest.mark.parametrize("dtype_x, dtype_y, op", [
    (dtype_x, dtype_y, op)
    for op in ['<<', '>>']
    for dtype_x in int_dtypes + uint_dtypes
    for dtype_y in int_dtypes + uint_dtypes
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_shift_op(dtype_x, dtype_y, op, num_ctas, device):
    expr = f'x {op} y'
    bw = max(_bitwidth(dtype_x), _bitwidth(dtype_y))
    if dtype_x.startswith('int'):
        dtype_z = f'int{bw}'
    else:
        dtype_z = f'uint{bw}'
    numpy_expr = f'x.astype(np.{dtype_z}) {op} y.astype(np.{dtype_z})'
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, device=device, num_ctas=num_ctas, y_low=0, y_high=65)


# ---------------
# test compare ops
# ---------------
ops = ['==', '!=', '>', '<', '>=', '<=']


@pytest.mark.parametrize("dtype_x, dtype_y, op, mode_x, mode_y",
                         # real
                         [
                             (dtype_x, dtype_y, op, 'real', 'real')
                             for op in ops
                             for dtype_x in dtypes
                             for dtype_y in dtypes
                         ] +
                         # NaNs
                         [('float32', 'float32', op, mode_x, mode_y)
                             for op in ops
                             for mode_x, mode_y in [('nan', 'real'),
                                                    ('real', 'nan'),
                                                    ('nan', 'nan')]

                          ])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_compare_op(dtype_x, dtype_y, op, mode_x, mode_y, num_ctas, device):
    expr = f'x {op} y'
    if (dtype_x in uint_dtypes and dtype_y in int_dtypes and _bitwidth(dtype_x) >= _bitwidth(dtype_y)):
        numpy_expr = f'x.astype(np.{dtype_x}) {op} y.astype(np.{dtype_x})'
    elif (dtype_y in uint_dtypes and dtype_x in int_dtypes and _bitwidth(dtype_y) >= _bitwidth(dtype_x)):
        numpy_expr = f'x.astype(np.{dtype_y}) {op} y.astype(np.{dtype_y})'
    else:
        numpy_expr = None
    _test_binary(dtype_x, dtype_y, expr, numpy_expr, mode_x=mode_x, mode_y=mode_y, device=device, num_ctas=num_ctas)


# ---------------
# test broadcast
# ---------------
@pytest.mark.parametrize("dtype", dtypes_with_bfloat16)
def test_broadcast(dtype, device):
    @triton.jit
    def broadcast_kernel(x_ptr, y_ptr, y_broadcasted_ptr, M: tl.constexpr, N: tl.constexpr):
        offset1 = tl.arange(0, M)
        offset2 = tl.arange(0, N)
        x = tl.load(x_ptr + N * offset1[:, None] + offset2[None, :])
        y = tl.load(y_ptr + offset2)
        _, y_broadcasted = tl.broadcast(x, y)
        tl.store(y_broadcasted_ptr + N * offset1[:, None] + offset2[None, :], y_broadcasted)

    M = 32
    N = 64
    rs = RandomState(17)
    x = numpy_random((M, N), dtype_str=dtype, rs=rs)
    y = numpy_random(N, dtype_str=dtype, rs=rs)
    _, y_broadcasted_np = np.broadcast_arrays(x, y)

    x_tri = to_triton(x, device=device, dst_type=dtype)
    y_tri = to_triton(y, device=device, dst_type=dtype)
    y_broadcasted_tri = to_triton(np.empty((M, N), dtype=y_broadcasted_np.dtype), device=device, dst_type=dtype)

    broadcast_kernel[(1,)](x_tri, y_tri, y_broadcasted_tri, M=M, N=N)
    assert (y_broadcasted_np == to_numpy(y_broadcasted_tri)).all()

# ----------
# test slice
# ----------


def test_slice(device):

    @triton.jit
    def slice_kernel(XBLOCK: tl.constexpr):
        data = tl.arange(0, XBLOCK)
        tl.static_assert(data.shape == [XBLOCK])

        t = data[None, :]
        tl.static_assert(t.shape == [1, XBLOCK])

        t = data[None, :, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        scalar = tl.full([], 1, tl.int32)
        tl.static_assert(scalar.shape == [])

        t = scalar[None]
        tl.static_assert(t.shape == [1])

        t = scalar[None, None]
        tl.static_assert(t.shape == [1, 1])

    slice_kernel[(1,)](XBLOCK=32)


# ------------------
# test invalid slice
# ------------------


def test_invalid_slice(device):
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst):
        dst[10:]

    with pytest.raises(triton.CompilationError, match='unsupported tensor index'):
        _kernel[(1,)](dst=dst)


# ----------------
# test expand_dims
# ----------------
def test_expand_dims(device):
    @triton.jit
    def expand_dims_kernel(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, 0)
        tl.static_assert(t.shape == [1, N])

        t = tl.expand_dims(offset1, 1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.expand_dims(offset1, -1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.expand_dims(offset1, -2)
        tl.static_assert(t.shape == [1, N])

        t = tl.expand_dims(offset1, (0, -1))
        tl.static_assert(t.shape == [1, N, 1])

        t = tl.expand_dims(offset1, (0, 1, 3))
        tl.static_assert(t.shape == [1, 1, N, 1])

        t = tl.expand_dims(offset1, (-4, 2, -1))
        tl.static_assert(t.shape == [1, N, 1, 1])

        t = tl.expand_dims(offset1, (3, 1, 2))
        tl.static_assert(t.shape == [N, 1, 1, 1])

        scalar = tl.sum(offset1)
        tl.static_assert(scalar.shape == [])
        t = tl.expand_dims(scalar, 0)
        tl.static_assert(t.shape == [1])

        t = tl.expand_dims(scalar, -1)
        tl.static_assert(t.shape == [1])

    N = 32
    dummy_tensor = torch.empty((), device=device)
    expand_dims_kernel[(1,)](dummy_tensor, N)


def test_expand_dims_error_cases(device):
    @triton.jit
    def dim_out_of_range1(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, -2)
        t = tl.expand_dims(offset1, -3)

    @triton.jit
    def dim_out_of_range2(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, 1)
        t = tl.expand_dims(offset1, 2)

    @triton.jit
    def dim_out_of_range3(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, 1)
        scalar = tl.sum(offset1)

        t = tl.expand_dims(scalar, 1)

    @triton.jit
    def duplicate_dim1(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, (0, 0))

    @triton.jit
    def duplicate_dim2(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, (0, -3))

    N = 32
    dummy_tensor = torch.empty((), device=device)

    with pytest.raises(triton.CompilationError, match="invalid axis -3"):
        dim_out_of_range1[(1,)](dummy_tensor, N)

    with pytest.raises(triton.CompilationError, match="invalid axis 2"):
        dim_out_of_range2[(1,)](dummy_tensor, N)

    with pytest.raises(triton.CompilationError, match="invalid axis 1"):
        dim_out_of_range3[(1,)](dummy_tensor, N)

    with pytest.raises(triton.CompilationError, match=r"duplicate axes, normalized axes = \[0, 0\]"):
        duplicate_dim1[(1,)](dummy_tensor, N)

    with pytest.raises(triton.CompilationError, match=r"duplicate axes, normalized axes = \[0, 0\]"):
        duplicate_dim2[(1,)](dummy_tensor, N)


# ----------------------------
# test invalid program id axis
# ----------------------------
def test_invalid_pid_axis(device):
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst):
        pid = tl.program_id(20)

    with pytest.raises(triton.CompilationError, match=r"program_id must be in \[0,3\]"):
        _kernel[(1,)](dst)


# ---------------
# test where
# ---------------
@pytest.mark.parametrize("dtype", dtypes_with_bfloat16 + ["*int32"])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_where(dtype, num_ctas, device):
    select_ptrs = False
    if dtype == "*int32":
        dtype = "int64"
        select_ptrs = True
    check_type_supported(dtype, device)

    @triton.jit
    def where_kernel(cond_ptr, a_ptr, b_ptr, output_ptr, n_elements,
                     BLOCK_SIZE: tl.constexpr,
                     TEST_POINTERS: tl.constexpr,
                     TEST_SCALAR_POINTERS: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        decide = tl.load(cond_ptr + offsets, mask=mask)
        if TEST_SCALAR_POINTERS:
            ptr = tl.where(tl.load(cond_ptr), a_ptr, b_ptr)
            output = tl.load(ptr + offsets, mask=mask)
        else:
            if TEST_POINTERS:
                a = tl.load(a_ptr + offsets, mask=mask).to(tl.pi32_t)
                b = tl.load(b_ptr + offsets, mask=mask).to(tl.pi32_t)
            else:
                a = tl.load(a_ptr + offsets, mask=mask)
                b = tl.load(b_ptr + offsets, mask=mask)
            output = tl.where(decide, a, b)
        tl.store(output_ptr + offsets, output, mask=mask)

    SIZE = 1_000
    rs = RandomState(17)
    cond = numpy_random(SIZE, 'bool', rs)
    x = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    y = numpy_random(SIZE, dtype_str=dtype, rs=rs)
    z = np.where(cond, x, y)

    cond_tri = to_triton(cond, device=device)
    x_tri = to_triton(x, device=device, dst_type=dtype)
    y_tri = to_triton(y, device=device, dst_type=dtype)
    z_tri = to_triton(np.empty(SIZE, dtype=z.dtype), device=device, dst_type=dtype)

    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']),)
    where_kernel[grid](cond_tri, x_tri, y_tri, z_tri, SIZE, BLOCK_SIZE=1024, TEST_POINTERS=select_ptrs, TEST_SCALAR_POINTERS=False, num_ctas=num_ctas)
    assert (z == to_numpy(z_tri)).all()
    if select_ptrs:
        where_kernel[grid](cond_tri, x_tri, y_tri, z_tri, SIZE, BLOCK_SIZE=1024, TEST_POINTERS=select_ptrs, TEST_SCALAR_POINTERS=True)
        z = np.where(cond[0], x, y)
        assert (z == to_numpy(z_tri)).all()


@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_where_broadcast(num_ctas, device):
    @triton.jit
    def where_kernel(cond_ptr, a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
        yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]

        mask = tl.load(cond_ptr + yoffsets)
        vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
        res = tl.where(mask, vals, 0.)
        tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)

    @triton.jit
    def where_scalar_condition(a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
        yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]
        mask = 0
        vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
        res = tl.where(mask, vals, 0.)
        tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)

    SIZE = 32
    dtype = 'float32'
    rs = RandomState(17)
    x = numpy_random((SIZE, SIZE), dtype_str=dtype, rs=rs)
    mask = numpy_random(SIZE, 'bool', rs=rs)
    z = np.where(mask, x, 0)
    cond_tri = to_triton(mask, device=device)
    x_tri = to_triton(x, device=device, dst_type=dtype)
    z_tri = to_triton(np.empty((SIZE, SIZE), dtype=z.dtype), device=device, dst_type=dtype)
    where_kernel[(1,)](cond_tri, x_tri, z_tri, SIZE)
    assert (z == to_numpy(z_tri)).all()
    where_scalar_condition[(1,)](x_tri, z_tri, SIZE, num_ctas=num_ctas)
    z = np.where(0, x, 0)
    assert (z == to_numpy(z_tri)).all()

# ---------------
# test unary ops
# ---------------


@pytest.mark.parametrize("dtype_x, expr", [
    (dtype_x, ' -x') for dtype_x in dtypes_with_bfloat16
] + [
    (dtype_x, ' ~x') for dtype_x in int_dtypes
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_unary_op(dtype_x, expr, num_ctas, device):
    _test_unary(dtype_x, expr, device=device, num_ctas=num_ctas)

# ----------------
# test math ops
# ----------------


@pytest.mark.parametrize("dtype_x, expr", [(dtype_x, expr) for dtype_x in ["float32", "float64"] for expr in ['exp', 'log', 'cos', 'sin']])
def test_math_op(dtype_x, expr, device):
    _test_unary(dtype_x, f'tl.{expr}(x)', f'np.{expr}(x) ', device=device)

# ----------------
# test abs
# ----------------


@pytest.mark.parametrize("dtype_x", [
    (dtype_x)
    for dtype_x in dtypes_with_bfloat16
])
def test_abs(dtype_x, device):
    _test_unary(dtype_x, 'tl.abs(x)', 'np.abs(x) ', device=device)


@pytest.mark.parametrize("in_dtype", [tl.float8e4b15, tl.float8e4nv, tl.float8e5])
def test_abs_fp8(in_dtype, device):
    if is_hip():
        pytest.skip('test_abs_fp8 not supported on HIP.')

    @triton.jit
    def abs_kernel(X, Z, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = tl.abs(x)
        tl.store(Z + off, z)

    f8_tensor = torch.tensor(range(-128, 128), dtype=torch.int8, device=device)
    # f32_to_f8 doesn't handle nan, so we make sure f8_tensor doesn't contain any nan
    all_exp_ones = (f8_tensor & 0b01111100) == 128 - 2**in_dtype.fp_mantissa_width
    f8_tensor[all_exp_ones] = 0
    f8 = triton.reinterpret(f8_tensor, in_dtype)
    n_elements = f8_tensor.numel()
    out_f8 = torch.empty_like(f8_tensor)
    abs_kernel[(1,)](f8, triton.reinterpret(out_f8, in_dtype), n_elements)

    f32_tensor = convert_float_to_float32(f8_tensor, in_dtype)
    expect = f32_tensor.abs()
    actual_f8 = convert_float_to_float32(out_f8, in_dtype)
    torch.testing.assert_close(actual_f8, expect, equal_nan=True)


# ----------------
# test indexing
# ----------------


def make_ptr_str(name, shape):
    rank = len(shape)
    offsets = []
    stride = 1
    for i in reversed(range(rank)):
        idx = ', '.join([':' if ii == i else 'None' for ii in range(rank)])
        offsets += [f'tl.arange(0, {shape[i]})[{idx}]*{stride}']
        stride *= shape[i]
    return f"{name} + {' + '.join(offsets)}"


# TODO: handle `%4 = triton_gpu.convert_layout %3 : (tensor<32xi32, #blocked0>) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>``
@pytest.mark.parametrize("expr, dtype_str", [
    (f'x[{s}]', d)
    for s in ['None, :', ':, None',
              'None, :, :',
              ':, :, None']
    for d in ['int32', 'uint32', 'uint16']
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_index1d(expr, dtype_str, num_ctas, device):
    rank_x = expr.count(':')
    rank_y = expr.count(',') + 1
    shape_x = [32 for _ in range(rank_x)]
    shape_z = [32 for _ in range(rank_y)]
    shape_z_rank_mismatch = [32 for _ in range(rank_y + 1)]
    shape_z_dim_mismatch = [64 for _ in range(rank_y)]

    # Triton kernel
    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        m = tl.arange(0, SIZE)
        n = tl.arange(0, SIZE)
        x = tl.load(X_PTR_EXPR)
        z = GENERATE_TEST_HERE
        tl.store(Z_PTR_EXPR, z)

    def generate_kernel(shape_x, shape_z):
        to_replace = {
            'X_PTR_EXPR': make_ptr_str('X', shape_x),
            'Z_PTR_EXPR': make_ptr_str('Z', shape_z),
            'GENERATE_TEST_HERE': expr,
        }
        return patch_kernel(kernel, to_replace)

    kernel_match = generate_kernel(shape_x, shape_z)
    kernel_dim_mismatch = generate_kernel(shape_x, shape_z_dim_mismatch)
    kernel_rank_mismatch = generate_kernel(shape_x, shape_z_rank_mismatch)

    # torch result
    x = numpy_random(shape_x, dtype_str=dtype_str)
    y = np.zeros(shape_z, dtype=getattr(np, dtype_str))
    z_ref = eval(expr) + y
    # triton result
    z_tri = to_triton(np.empty_like(z_ref), device=device)
    x_tri = to_triton(x, device=device)
    kernel_match[(1, )](z_tri, x_tri, num_warps=1, SIZE=shape_x[0])
    # compare
    assert (z_ref == to_numpy(z_tri)).all()

    def catch_compilation_error(kernel):
        try:
            kernel[(1, )](z_tri, x_tri, num_warps=1,
                          SIZE=shape_x[0], num_ctas=num_ctas)
        except triton.CompilationError as e:
            np.testing.assert_(True)
        except BaseException:
            np.testing.assert_(False)

    catch_compilation_error(kernel_dim_mismatch)
    catch_compilation_error(kernel_rank_mismatch)


# ---------------
# test tuples
# ---------------


@triton.jit
def tuples_fn(a, b):
    return a + b, \
        a - b, \
        a * b


def test_tuples(device):
    @triton.jit
    def with_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = tuples_fn(x, y)
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    @triton.jit
    def without_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = x + y, x - y, x * y
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    x = torch.tensor([1.3], device=device, dtype=torch.float32)
    y = torch.tensor([1.9], device=device, dtype=torch.float32)
    a_tri = torch.tensor([0], device=device, dtype=torch.float32)
    b_tri = torch.tensor([0], device=device, dtype=torch.float32)
    c_tri = torch.tensor([0], device=device, dtype=torch.float32)
    for kernel in [with_fn, without_fn]:
        kernel[(1, )](x, y, a_tri, b_tri, c_tri, num_warps=1)
        a_ref, b_ref, c_ref = x + y, x - y, x * y
        assert a_tri == a_ref
        assert b_tri == b_ref
        assert c_tri == c_ref


@triton.jit(noinline=True)
def noinline_simple_fn(x, y, Z):
    z = x + y
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_call_graph_fn1(x):
    return x + 1


@triton.jit(noinline=True)
def noinline_call_graph_fn2(y):
    return y + 2


@triton.jit(noinline=True)
def noinline_call_graph_fn(x, y, Z):
    t0 = noinline_call_graph_fn1(x)
    t1 = noinline_call_graph_fn2(y)
    z = t0 + t1
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_shared_fn(x, y, Z):
    offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    z = tl.load(Z + offs)
    z = tl.dot(z, z) + x + y
    tl.store(Z + offs, z)


@triton.jit(noinline=True)
def noinline_dynamic_fn(x, y, Z):
    if x >= 1:
        x = noinline_call_graph_fn1(x)
    else:
        x = noinline_call_graph_fn2(x)
    if y >= 2:
        y = noinline_call_graph_fn2(y)
    else:
        y = noinline_call_graph_fn1(y)
    z = x + y
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_call_multi_values_fn(x, y):
    return x + 1, y + 2


@triton.jit(noinline=True)
def noinline_multi_values_fn(x, y, Z):
    x, y = noinline_call_multi_values_fn(x, y)
    z = x + y
    tl.store(Z, z)


@pytest.mark.parametrize("mode", ["simple", "call_graph", "shared", "dynamic", "multi_values"])
def test_noinline(mode, device):
    if is_hip() and mode == "shared":
        pytest.skip('test_noinline["shared"] not supported on HIP.')

    @triton.jit
    def kernel(X, Y, Z):
        x = tl.load(X)
        y = tl.load(Y)
        GENERATE_TEST_HERE(x, y, Z)

    func_name = f'noinline_{mode}_fn'
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': func_name})
    x = torch.tensor([1.0], device=device, dtype=torch.float32)
    y = torch.tensor([2.0], device=device, dtype=torch.float32)
    if mode == "shared":
        z = torch.ones((16, 16), device=device, dtype=torch.float32)
    else:
        z = torch.tensor([0.0], device=device, dtype=torch.float32)
    kernel[(1,)](x, y, z, num_warps=1)
    if mode == "simple":
        assert torch.equal(z, x + y)
    elif mode == "call_graph" or mode == "dynamic" or mode == "multi_values":
        assert torch.equal(z, x + 1 + y + 2)
    elif mode == "shared":
        ref = torch.full((16, 16), 16, device=device, dtype=torch.float32)
        assert torch.equal(z, ref + x + y)


# ---------------
# test atomics
# ---------------
@pytest.mark.parametrize("op, dtype_x_str, mode, sem", itertools.chain.from_iterable([
    [
        ('add', 'float16', mode, sem),
        ('add', 'uint32', mode, sem), ('add', 'int32', mode, sem), ('add', 'float32', mode, sem),
        ('add', 'uint64', mode, sem), ('add', 'int64', mode, sem), ('add', 'float64', mode, sem),
        ('max', 'uint32', mode, sem), ('max', 'int32', mode, sem), ('max', 'float32', mode, sem),
        ('max', 'uint64', mode, sem), ('max', 'int64', mode, sem), ('max', 'float64', mode, sem),
        ('min', 'uint32', mode, sem), ('min', 'int32', mode, sem), ('min', 'float32', mode, sem),
        ('min', 'uint64', mode, sem), ('min', 'int64', mode, sem), ('min', 'float64', mode, sem),
    ]
    for mode in ['all_neg', 'all_pos', 'min_neg', 'max_pos']
    for sem in [None, 'acquire', 'release', 'acq_rel', 'relaxed']]))
def test_atomic_rmw(op, dtype_x_str, mode, sem, device):
    check_cuda_only(device)

    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:
        if dtype_x_str == 'float16':
            pytest.skip("Only test atomic float16 ops on devices with sm >= 70")
    n_programs = 5

    # triton kernel
    @triton.jit
    def kernel(X, Z):
        pid = tl.program_id(0)
        x = tl.load(X + pid)
        old = GENERATE_TEST_HERE
        tl.static_assert(old.dtype == x.dtype)

    sem_arg = sem if sem is None else f'"{sem}"'
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.atomic_{op}(Z, x, sem={sem_arg})'})
    numpy_op = {'add': np.sum, 'max': np.max, 'min': np.min}[op]
    max_neutral = float('-inf') if dtype_x_str in float_dtypes else np.iinfo(getattr(np, dtype_x_str)).min
    min_neutral = float('inf') if dtype_x_str in float_dtypes else np.iinfo(getattr(np, dtype_x_str)).max
    neutral = {'add': 0, 'max': max_neutral, 'min': min_neutral}[op]

    # triton result
    rs = RandomState(17)
    x = np.array([2**i for i in range(n_programs)], dtype=getattr(np, dtype_x_str))
    if mode == 'all_neg':
        x = -np.abs(x)
    if mode == 'all_pos':
        x = np.abs(x)
    if mode == 'min_neg':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = -np.max(np.abs(x)) - 1
    if mode == 'max_pos':
        idx = rs.randint(n_programs, size=(1, )).item()
        x[idx] = np.max(np.abs(x)) + 1
    x_tri = to_triton(x, device=device)

    z_tri = to_triton(np.array([neutral], dtype=getattr(np, dtype_x_str)), device=device)
    h = kernel[(n_programs, )](x_tri, z_tri)
    # torch result
    z_ref = numpy_op(x).astype(getattr(np, dtype_x_str))
    # compare
    exact = op not in ['add']
    if exact:
        assert z_ref.item() == to_numpy(z_tri).item()
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)
    sem_str = "acq_rel" if sem is None else sem
    if is_hip():
        return

    assert f"atom.global.gpu.{sem_str}" in h.asm["ptx"]


@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_atomic_rmw_predicate(num_ctas, device):
    @triton.jit
    def kernel(X):
        val = tl.program_id(0)
        if val < 64:
            tl.atomic_max(X, val)
    x = torch.zeros((1,), device=device, dtype=torch.int32)
    kernel[(4096,)](x, num_ctas=num_ctas)
    assert x.item() == 63


@pytest.mark.parametrize("shape, axis, num_ctas",
                         [(shape, axis, num_ctas) for shape in [(2, 2), (2, 8), (8, 2), (8, 8), (32, 32), (64, 64)] for axis in [0, 1] for num_ctas in num_ctas_list])
def test_tensor_atomic_rmw(shape, axis, num_ctas, device):
    shape0, shape1 = shape
    # triton kernel

    @triton.jit
    def kernel(Z, X, AXIS: tl.constexpr, SHAPE0: tl.constexpr, SHAPE1: tl.constexpr):
        off0 = tl.arange(0, SHAPE0)
        off1 = tl.arange(0, SHAPE1)
        x = tl.load(X + off0[:, None] * SHAPE1 + off1[None, :])
        z = tl.sum(x, axis=AXIS)
        if AXIS == 1:
            tl.atomic_add(Z + off0, z)
        else:
            tl.atomic_add(Z + off1, z)
    rs = RandomState(17)
    x = numpy_random((shape0, shape1), dtype_str="float32", rs=rs)
    # reference result
    z_ref = np.sum(x, axis=axis, keepdims=False)
    # triton result
    x_tri = to_triton(x, device=device)
    z_shape = (shape0, ) if axis == 1 else (shape1, )
    z_tri = to_triton(np.zeros(z_shape, dtype="float32"), device=device)
    kernel[(1,)](z_tri, x_tri, axis, shape0, shape1, num_ctas=num_ctas)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=1e-4)


@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_tensor_atomic_rmw_block(num_ctas, device):
    shape = (8, 8)

    @triton.jit
    def kernel(X, SHAPE0: tl.constexpr, SHAPE1: tl.constexpr):
        off0 = tl.arange(0, SHAPE0)
        off1 = tl.arange(0, SHAPE1)
        offs = off0[:, None] * SHAPE1 + off1[None, :]
        val = offs.to(tl.float32)
        x = X + offs
        tl.atomic_min(x, val)
    x = torch.ones((8, 8), device=device, dtype=torch.float32)
    kernel[(2,)](x, shape[0], shape[1], num_ctas=num_ctas)
    assert torch.min(x).item() == 0.0


@pytest.mark.parametrize("sem", [None, 'acquire', 'release', 'acq_rel', 'relaxed'])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_atomic_cas(sem, num_ctas, device):
    # 1. make sure that atomic_cas changes the original value (Lock)
    @triton.jit
    def change_value(Lock):
        tl.atomic_cas(Lock, 0, 1)

    Lock = torch.zeros((1,), device=device, dtype=torch.int32)
    change_value[(1,)](Lock)

    assert (Lock[0] == 1)

    # 2. only one block enters the critical section
    @triton.jit
    def serialized_add(data, Lock, SEM: tl.constexpr):
        ptrs = data + tl.arange(0, 128)
        while tl.atomic_cas(Lock, 0, 1, SEM) == 1:
            pass

        tl.store(ptrs, tl.load(ptrs) + 1.0)

        # release lock
        tl.atomic_xchg(Lock, 0)

    Lock = torch.zeros((1,), device=device, dtype=torch.int32)
    data = torch.zeros((128,), device=device, dtype=torch.float32)
    ref = torch.full((128,), 64.0)
    h = serialized_add[(64,)](data, Lock, SEM=sem, num_ctas=num_ctas)
    sem_str = "acq_rel" if sem is None else sem
    np.testing.assert_allclose(to_numpy(data), to_numpy(ref))
    if is_hip():
        return
    assert f"atom.global.{sem_str}" in h.asm["ptx"]


@pytest.mark.parametrize("num_ctas", num_ctas_list)
@pytest.mark.parametrize("dtype_str", int_dtypes)
def test_atomic_load_store_relaxed(num_ctas, device, dtype_str):

    @triton.jit
    def global_sum(in_ptr, out_ptr, workspace, XBLOCK: tl.constexpr):
        # Compute the global sum of an array of integers using relaxed atomics
        xid = tl.program_id(0)
        xindex = xid * XBLOCK + tl.arange(0, XBLOCK)

        block_data = tl.load(in_ptr + xindex)
        block_sum = tl.sum(block_data).to(block_data.dtype)

        if xid != tl.num_programs(0) - 1:
            tl.atomic_store(workspace + xid, block_sum, sem="relaxed")
            return

        # Last block does the final sum
        full_sum = block_sum.to(tl.int64)
        for target in range(tl.num_programs(0) - 1):
            cur_val = tl.full([], -1, workspace.dtype.element_ty)
            while cur_val == -1:
                # spin until value is available
                cur_val = tl.atomic_load(workspace + target, sem="relaxed")

            full_sum += cur_val.to(tl.int64)

        tl.store(out_ptr, full_sum)

    dtype = getattr(torch, dtype_str)
    XBLOCK = 16 if _bitwidth(dtype_str) > 8 else 1
    XGRID = 128
    # Make sure there is no overflow
    in_data = torch.randint(
        size=(XGRID, XBLOCK), low=0, high=16, device=device, dtype=dtype)
    actual = torch.full((), -1, device=device, dtype=torch.int64)
    workspace = torch.full((XGRID,), -1, device=device, dtype=dtype)
    expect = torch.sum(in_data)

    h = global_sum[(XGRID,)](in_data, actual, workspace,
                             XBLOCK=XBLOCK, num_ctas=num_ctas)

    torch.testing.assert_close(expect, actual)

    assert f"st.global.relaxed.gpu" in h.asm["ptx"]
    assert f"ld.global.relaxed.gpu" in h.asm["ptx"]


@pytest.mark.parametrize("num_ctas", num_ctas_list)
@pytest.mark.parametrize("dtype_str", ("bool", 'uint8', *int_dtypes,))
def test_atomic_load_store_acquire_release(num_ctas, device, dtype_str):

    @triton.jit
    def global_parallel_sum(in_ptr, out_ptr, workspace, flags,
                            XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
        xid = tl.program_id(0)
        xindex = xid * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        yindex = tl.arange(0, YBLOCK)

        block_data = tl.load(in_ptr + xindex * YBLOCK + yindex[None, :])
        block_sum = tl.sum(block_data, 0)

        if xid != tl.num_programs(0) - 1:
            # Publish block_sum in global memory, synchronized via the atomic flag
            tl.store(workspace + xid * YBLOCK + yindex, block_sum)
            tl.atomic_store(flags + xid, True, sem="release")
            return

        # Last block does the final sum
        full_sum = block_sum
        for target in range(tl.num_programs(0) - 1):
            ready_flag = tl.full([], 0, flags.dtype.element_ty)
            while not ready_flag.to(tl.int1):
                # spin until value is available
                ready_flag = tl.atomic_load(flags + target, sem="acquire")

            full_sum += tl.load(workspace + target * YBLOCK + yindex)

        tl.store(out_ptr + yindex, full_sum)

    dtype = getattr(torch, dtype_str)
    XBLOCK = 8
    YBLOCK = 16
    XGRID = 1024
    in_data = torch.testing.make_tensor(
        (XGRID, XBLOCK, YBLOCK), low=0, device=device, dtype=torch.float32)
    actual = torch.full((YBLOCK,), np.nan, device=device, dtype=torch.float32)
    workspace = torch.zeros((XGRID, YBLOCK), device=device, dtype=torch.float32)
    flags = torch.zeros((XGRID,), device=device, dtype=dtype)

    expect = in_data.sum(dim=(0, 1))

    h = global_parallel_sum[(XGRID,)](in_data, actual, workspace, flags,
                                      XBLOCK=XBLOCK, YBLOCK=YBLOCK,
                                      num_ctas=num_ctas)

    torch.testing.assert_close(expect, actual, rtol=1e-5, atol=1e-5)

    assert f"st.global.release.gpu" in h.asm["ptx"]
    assert f"ld.global.acquire.gpu" in h.asm["ptx"]


# ---------------
# test cast
# ---------------


@pytest.mark.parametrize("dtype_x, dtype_z, bitcast", [
    (dtype_x, dtype_z, False)
    for dtype_x in dtypes
    for dtype_z in dtypes
] + [
    ('float32', 'bfloat16', False),
    ('bfloat16', 'float32', False),
    ('float32', 'int32', True),
    ('float32', 'int1', False),
    ('int8', 'bfloat16', False),
] + [
    (f'uint{x}', f'int{x}', True) for x in [8, 16, 32, 64]
] + [
    (f'int{x}', f'uint{x}', True) for x in [8, 16, 32, 64]
])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_cast(dtype_x, dtype_z, bitcast, num_ctas, device):
    # bfloat16 on cc < 80 will not be tested
    check_type_supported(dtype_x, device)
    check_type_supported(dtype_z, device)

    if is_hip() and (dtype_z == "bfloat16"):
        pytest.skip(f'test_cast{(dtype_x, dtype_z)} cast to bfloat16 not supported on HIP.')

    size = 1024
    # This is tricky because numpy doesn't have bfloat, and torch doesn't have uints.
    if dtype_x.startswith('bfloat'):
        x_tri = torch.randn(size, dtype=getattr(torch, dtype_x), device=device)
    else:
        x = numpy_random(size, dtype_str=dtype_x, low=-10, high=10) * 10
        # Triton clamps negative values to zero, while numpy wraps around
        # intmax, so avoid negatives for now.
        # TODO: figure out which one should actually be happening, and test it
        if dtype_z in uint_dtypes:
            x = np.absolute(x)
        x_tri = to_triton(x, device=device)

    # triton kernel
    @triton.jit
    def kernel(X, Z, BITCAST: tl.constexpr, SIZE: tl.constexpr):
        x_ptr = X + tl.arange(0, SIZE)
        z_ptr = Z + tl.arange(0, SIZE)
        x = tl.load(x_ptr)
        z = x.to(Z.dtype.element_ty, bitcast=BITCAST)
        tl.store(z_ptr, z)

    dtype_z_np = dtype_z if dtype_z != 'int1' else 'bool_'
    # triton result
    if dtype_z.startswith('bfloat'):
        z_tri = torch.empty((size,), dtype=getattr(torch, dtype_z), device=device)
    else:
        z_tri = to_triton(np.empty((size, ), dtype=getattr(np, dtype_z_np)), device=device)
    kernel[(1, )](x_tri, z_tri, BITCAST=bitcast, SIZE=size, num_warps=1, num_ctas=num_ctas)
    # torch result
    if dtype_z.startswith('bfloat') or dtype_x.startswith('bfloat'):
        assert bitcast is False
        z_ref = x_tri.to(z_tri.dtype)
        torch.testing.assert_close(z_ref, z_tri, rtol=0, atol=0)
    else:
        if bitcast:
            z_ref = x.view(getattr(np, dtype_z_np))
        else:
            z_ref = x.astype(getattr(np, dtype_z_np))
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0, atol=0)


@pytest.mark.parametrize("dtype_str, num_warps", [(dtype_str, num_warps) for dtype_str in int_dtypes + float_dtypes for num_warps in [4, 8]])
def test_cat(dtype_str, num_warps, device):
    check_type_supported(dtype_str, device)

    @triton.jit
    def kernel(X, Y, Z, N: tl.constexpr):
        offs = tl.arange(0, N)
        x = tl.load(X + offs)
        y = tl.load(Y + offs)
        z = tl.cat(x, y, can_reorder=True)
        tl.store(Z + tl.arange(0, 2 * N), z)

    x = torch.arange(0, 128, device=device).to(getattr(torch, dtype_str))
    y = torch.arange(-128, 0, device=device).to(getattr(torch, dtype_str))
    z_ref = torch.cat([x, y], dim=0).sum()
    z = torch.zeros((256,), dtype=getattr(torch, dtype_str), device=device)
    kernel[(1, )](x, y, z, N=128, num_warps=num_warps)
    assert z.sum() == z_ref
    # check if there's no duplicate value in z
    assert z.unique().size(0) == z.size(0)


@pytest.mark.parametrize("dtype_str", list(torch_dtypes))
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_store_constant(dtype_str, num_ctas, device):
    check_type_supported(dtype_str, device)

    """Tests that boolean True is stored as 1"""
    @triton.jit
    def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        output = GENERATE_TEST_HERE
        tl.store(output_ptr + offsets, output, mask=mask)

    triton_dtype_str = 'uint8' if dtype_str == 'bool' else dtype_str
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.zeros([BLOCK_SIZE], dtype=tl.{triton_dtype_str}) + 1'})
    block_size = 128
    ref = torch.ones([block_size], dtype=getattr(torch, dtype_str), device=device)
    output = torch.zeros([block_size], dtype=getattr(torch, dtype_str), device=device)
    kernel[(1,)](output, block_size, BLOCK_SIZE=block_size, num_ctas=num_ctas)

    assert torch.all(output == ref)


def test_load_store_same_ptr(device):
    @triton.jit()
    def kernel(in_out_ptr):
        pid = tl.program_id(axis=0)
        x = tl.load(in_out_ptr + pid)
        out = x * 2
        tl.store(in_out_ptr + pid, out)

    for _ in range(1000):
        x = torch.ones((65536,), device=device, dtype=torch.float32)
        if is_hip():
            kernel[(65536,)](x, num_warps=16)  # threads per Warp for ROCM is 64
        else:
            kernel[(65536,)](x, num_warps=32)
        assert torch.all(x == 2)


def convert_float_to_float32(fp: torch.tensor, dtype=None):
    if not dtype:
        dtype = getattr(tl, torch_dtype_name(fp.dtype))

    fp = fp.view(getattr(torch, f"int{dtype.primitive_bitwidth}"))
    exp_width = dtype.primitive_bitwidth - dtype.fp_mantissa_width - 1
    exp_bias = dtype.exponent_bias
    sign = ((fp >> (dtype.primitive_bitwidth - 1)) & 0x01).int()
    exp = ((fp >> dtype.fp_mantissa_width) & ((1 << exp_width) - 1)).int()
    frac = (fp & ((1 << dtype.fp_mantissa_width) - 1)).int()

    output = torch.where(exp == 0,
                         # subnormal
                         ((-1.0) ** sign) * (2.0 ** (1 - exp_bias)) * (frac / (2.0 ** dtype.fp_mantissa_width)),
                         # normal
                         ((-1.0) ** sign) * (2.0 ** (exp - exp_bias)) * (1.0 + frac / (2.0 ** dtype.fp_mantissa_width))).float()

    extended_exp = ((1 << (tl.float32.primitive_bitwidth - tl.float32.fp_mantissa_width - 1)) - 1) << tl.float32.fp_mantissa_width
    # special cases, exp is 0b11..1
    if dtype in [tl.float8e4nv, tl.float8e4b15]:
        # float8e4m3nv does not have infinities
        output[fp == 0b01111111] = torch.nan
        output[fp == 0b11111111] = torch.nan
    else:
        output = torch.where(exp == (1 << exp_width) - 1,
                             ((sign << (tl.float32.primitive_bitwidth - 1)) | extended_exp | (frac << (tl.float32.fp_mantissa_width - dtype.fp_mantissa_width))).view(torch.float32),
                             output)
    return output


@pytest.mark.parametrize("in_dtype", [torch.float16, torch.bfloat16])
def test_convert_float16_to_float32(in_dtype, device):
    """Tests that check convert_float_to_float32 function"""
    check_type_supported(in_dtype, device)

    f16_input = torch.tensor(range(-int(2 ** (16 - 1)), int(2 ** (16 - 1))), dtype=torch.int16).view(in_dtype)
    f32_output = convert_float_to_float32(f16_input)

    nan = f16_input.isnan()
    assert torch.all(f32_output[nan].isnan())
    inf = f16_input.isinf()
    assert torch.all(f32_output[inf].isinf())
    other = torch.logical_not(torch.logical_or(nan, inf))
    assert torch.all(f16_input[other] == f32_output[other])


def serialize_fp8(np_data, in_dtype):
    if in_dtype == tl.float8e4b15x4:
        # triton's f8e4b15 format is optimized for software emulation
        # as a result, each pack of 4xfp8 values:
        # s0b0s1b1s2b2s3b3 (for s, b sign and bits respectively)
        # is actually internally stored as
        # s0s2b0b2s1s3b1b3
        # we apply the conversion here
        f8x4 = np_data.view(np.uint32)
        s = [(f8x4 & (0x80000000 >> i)) << i for i in range(0, 32, 8)]
        b = [(f8x4 & (0x7f000000 >> i)) << i for i in range(0, 32, 8)]
        signs = (s[0] >> 0) | (s[1] >> 16) | (s[2] >> 1) | (s[3] >> 17)
        bits = (b[0] >> 1) | (b[1] >> 17) | (b[2] >> 8) | (b[3] >> 24)
        # tensor of triton fp8 data
        return (signs | bits).view(np.int8)
    else:
        return np_data

# inverse of `serialize_fp8`


def deserialize_fp8(np_data, in_dtype):
    if in_dtype == tl.float8e4b15x4:
        f8x4 = np_data.view(np.uint32)
        s = [(f8x4 & (0x80000000 >> i)) << i for i in [0, 16, 1, 17]]
        b = [(f8x4 & (0x7f000000 >> i)) << i for i in [1, 17, 8, 24]]
        signs = (s[0] >> 0) | (s[1] >> 8) | (s[2] >> 16) | (s[3] >> 24)
        bits = (b[0] >> 0) | (b[1] >> 8) | (b[2] >> 16) | (b[3] >> 24)
        return (signs | bits).view(np.int8)
    else:
        return np_data


@pytest.mark.parametrize("in_dtype", [tl.float8e4b15, tl.float8e4b15x4, tl.float8e4nv, tl.float8e5])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_fp8_fpN_roundtrip(in_dtype, out_dtype, device):
    """
    For all possible float8 values (ref_fp8 = range(0, 256)), test that:
        - conversion tri_fp16 = convert(input=ref_fp8, out=out_dtype) matches the reference
        - conversion tri_fp8 = convert(input=tri_fp16, out=out_dtype) matches the original
    this is only possible if both conversions are correct
    """
    check_type_supported(in_dtype, device)
    check_type_supported(out_dtype, device)
    if is_hip():
        pytest.skip('test_abs_fp8 not supported on HIP.')

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    # initialize array containing all possible f8 values except NaN
    ref_fp8 = np.array(range(-128, 128), dtype=np.int8)
    is_nan = (ref_fp8 & 0b01111100) == 128 - 2**in_dtype.fp_mantissa_width
    exp_mask = 0b01111111 ^ ((1 << in_dtype.fp_mantissa_width) - 1)
    is_subnormal = np.logical_or((ref_fp8 & exp_mask) == 0, (ref_fp8 & exp_mask) == exp_mask)
    ref_fp8[is_nan] = 0
    ref_fp8[is_subnormal] = 0
    tri_fp8 = torch.from_numpy(serialize_fp8(ref_fp8, in_dtype)).cuda()
    tri_fp16 = torch.empty(256, dtype=out_dtype, device="cuda")
    copy_kernel[(1,)](triton.reinterpret(tri_fp8, in_dtype), tri_fp16, tri_fp16.shape[0], BLOCK_SIZE=1024)

    ref_fp8 = torch.from_numpy(ref_fp8).cuda()
    ref_fp16 = convert_float_to_float32(ref_fp8, in_dtype)
    assert torch.all(tri_fp16[~is_subnormal] == ref_fp16[~is_subnormal])

    ref_fp8 = torch.empty_like(tri_fp16, dtype=torch.int8)
    copy_kernel[(1,)](tri_fp16, triton.reinterpret(ref_fp8, in_dtype), tri_fp16.shape[0], BLOCK_SIZE=1024)
    assert torch.all(tri_fp8 == ref_fp8)

# ---------------
# test reduce
# ---------------


def get_reduced_dtype(dtype_str, op):
    if op in ('argmin', 'argmax'):
        return 'int32'
    if dtype_str == 'bfloat16':
        return 'float32'
    return dtype_str


@pytest.mark.parametrize("op, dtype_str, shape",
                         [(op, dtype, shape)
                          for op in ['min', 'max',
                                     'min-with-indices',
                                     'max-with-indices',
                                     'argmin-tie-break-left',
                                     'argmax-tie-break-left',
                                     'sum']
                          for dtype in dtypes_with_bfloat16
                          for shape in [32, 64, 128, 512]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_reduce1d(op, dtype_str, shape, num_ctas, device):
    check_type_supported(dtype_str, device)  # bfloat16 on cc < 80 will not be tested

    if is_hip():
        pytest.skip(f"test_reduce1d not supported on HIP")

    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        GENERATE_TEST_HERE
        tl.store(Z, z)

    if 'with-indices' in op:
        patch = f'z, _ = tl.{op.split("-")[0]}(x, axis=0, return_indices=True)'
    elif 'arg' in op:
        tie_break_left = 'tie-break-left' in op
        patch = f'z = tl.{op.split("-")[0]}(x, axis=0, tie_break_left={tie_break_left})'
    else:
        patch = f'z = tl.{op}(x, axis=0)'
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': patch})
    # input
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random((shape,), dtype_str=dtype_str, rs=rs)
    numpy_op = {'sum': np.sum, 'max': np.max, 'min': np.min,
                'max-with-indices': np.max,
                'min-with-indices': np.min,
                'argmin-tie-break-fast': np.argmin,
                'argmin-tie-break-left': np.argmin,
                'argmax-tie-break-fast': np.argmax,
                'argmax-tie-break-left': np.argmax}[op]
    if 'tie-break-left' in op:
        x[3:10] = numpy_op(x)
    x_tri = to_triton(x, device=device)
    # numpy result
    z_dtype_str = 'int32' if op in ('argmin', 'argmax') else dtype_str
    z_tri_dtype_str = z_dtype_str
    if op not in ['argmin', 'argmax'] and dtype_str == 'bfloat16':
        z_dtype_str = 'float32'
        z_ref = numpy_op(x).astype(getattr(np, z_dtype_str))
        # trunc mantissa for a fair comparison of accuracy
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
        z_tri_dtype_str = 'bfloat16'
    else:
        z_ref = numpy_op(x).astype(getattr(np, z_dtype_str))
    # triton result
    z_tri = to_triton(numpy_random((1,), dtype_str=z_dtype_str, rs=rs),
                      device=device, dst_type=z_tri_dtype_str)
    kernel[(1,)](x_tri, z_tri, BLOCK=shape, num_ctas=num_ctas)
    z_tri = to_numpy(z_tri)
    # compare
    if op == 'sum':
        np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        if op in ('argmin', 'argmax'):
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            np.testing.assert_equal(x[z_ref], x[z_tri])
        else:
            np.testing.assert_equal(z_ref, z_tri)


# TODO: [Qingyi] Fix argmin / argmax
reduce_configs1 = [
    (op, dtype, (1, 1024), axis) for dtype in dtypes_with_bfloat16
    for op in ['min', 'max', 'sum', 'argmin', 'argmax']
    for axis in [1]
]


# shape (128, 256) and (32, 1024) are not enabled on sm86 because the required shared memory
# exceeds the limit of 99KB
reduce2d_shapes = [(2, 32), (4, 32), (4, 128)]
# TODO: fix and uncomment
# , (32, 64), (64, 128)]
if torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0):
    reduce2d_shapes += [(128, 256) and (32, 1024)]


reduce_configs2 = [
    (op, 'float32', shape, axis)
    for op in ['min', 'max', 'sum', 'argmin', 'argmax']
    for shape in reduce2d_shapes
    for axis in [0, 1]
] + [
    (op, 'float32', [16, 32], None)
    for op in ['min', 'max', 'sum']
]


@pytest.mark.parametrize("op, dtype_str, shape, axis", reduce_configs1 + reduce_configs2)
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_reduce2d(op, dtype_str, shape, axis, num_ctas, device):
    check_type_supported(dtype_str, device)  # bfloat16 on cc < 80 will not be tested

    if is_hip():
        pytest.skip(f"test_reduce2d not supported on HIP")
    # triton kernel

    @triton.jit
    def kernel(X, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        z = GENERATE_TEST_HERE
        if AXIS is None:
            tl.store(Z, z)
        elif AXIS == 1:
            tl.store(Z + range_m, z)
        else:
            tl.store(Z + range_n, z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{op}(x, axis=AXIS)'})
    # input
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    x_tri = to_triton(x, device=device)
    numpy_op = {'sum': np.sum, 'max': np.max, 'min': np.min,
                'argmin': np.argmin, 'argmax': np.argmax}[op]
    z_dtype_str = get_reduced_dtype(dtype_str, op)
    z_tri_dtype_str = z_dtype_str
    # numpy result
    if op not in ['argmin', 'argmax'] and dtype_str == 'bfloat16':
        z_dtype_str = 'float32'
        z_tri_dtype_str = 'bfloat16'
        z_ref = numpy_op(x, axis=axis).astype(getattr(np, z_dtype_str))
        # trunc mantissa for a fair comparison of accuracy
        z_ref = (z_ref.view('uint32') & np.uint32(0xffff0000)).view('float32')
    else:
        z_ref = numpy_op(x, axis=axis).astype(getattr(np, z_dtype_str))
    # triton result
    ret_numel = 1 if axis is None else shape[1 - axis]
    z_tri = to_triton(numpy_random((ret_numel,), dtype_str=z_dtype_str, rs=rs),
                      device=device, dst_type=z_tri_dtype_str)
    kernel[(1,)](x_tri, z_tri, BLOCK_M=shape[0],
                 BLOCK_N=shape[1], AXIS=axis, num_ctas=num_ctas)
    z_tri = to_numpy(z_tri)
    # compare
    if op == 'sum':
        np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        if op in ('argmin', 'argmax'):
            # argmin and argmax can have multiple valid indices.
            # so instead we compare the values pointed by indices
            z_ref_index = np.expand_dims(z_ref, axis=axis)
            z_tri_index = np.expand_dims(z_tri, axis=axis)
            z_ref_value = np.take_along_axis(x, z_ref_index, axis=axis)
            z_tri_value = np.take_along_axis(x, z_tri_index, axis=axis)
            np.testing.assert_equal(z_ref_value, z_tri_value)
        else:
            np.testing.assert_equal(z_ref, z_tri)


scan2d_shapes = [(8, 32), (16, 32), (32, 16), (2, 1024), (1024, 2), (32, 32), (1, 1024)]

scan_configs = [
    (op, type, shape, axis, num_warps)
    for num_warps in [4, 16]
    for type in ['int32', 'float32']
    for axis in [1, 0]
    for shape in scan2d_shapes
    for op in ['cumsum', 'cumprod']
]


@pytest.mark.parametrize("op, dtype_str, shape, axis, num_warps", scan_configs)
def test_scan2d(op, dtype_str, shape, axis, num_warps, device):
    if is_hip():
        pytest.skip("test_scan2d is not supported in HIP")
    check_type_supported(dtype_str, device)

    # triton kernel
    @triton.jit
    def kernel(X, Z, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, AXIS: tl.constexpr):
        range_m = tl.arange(0, BLOCK_M)
        range_n = tl.arange(0, BLOCK_N)
        x = tl.load(X + range_m[:, None] * BLOCK_N + range_n[None, :])
        z = GENERATE_TEST_HERE
        tl.store(Z + range_m[:, None] * BLOCK_N + range_n[None, :], z)

    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{op}(x, axis={axis})'})
    # input
    rs = RandomState(17)
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)
    z = np.empty_like(x)
    x_tri = to_triton(x, device=device)
    numpy_op = {'cumsum': np.cumsum, 'cumprod': np.cumprod}[op]
    z_dtype_str = dtype_str
    z_ref = numpy_op(x, axis=axis).astype(getattr(np, z_dtype_str))
    # triton result
    z_tri = to_triton(z, device=device)
    kernel[(1,)](x_tri, z_tri, BLOCK_M=shape[0], BLOCK_N=shape[1], AXIS=axis, num_warps=num_warps)
    z_tri = to_numpy(z_tri)
    # compare
    if dtype_str == 'float32':
        if op == 'cumprod':
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01, atol=1e-3)
        else:
            np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)
    else:
        np.testing.assert_equal(z_ref, z_tri)


scan_layouts = [
    BlockedLayout([1, 4], [4, 8], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, 4], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [4, 8], [1, 4], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [4, 8], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [8, 4], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),

    BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [4, 8], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [4, 8], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([2, 2], [8, 4], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("M, N", [[32, 16], [32, 32], [32, 64], [64, 32]])
@pytest.mark.parametrize("src_layout", scan_layouts)
@pytest.mark.parametrize("axis", [0, 1])
def test_scan_layouts(M, N, src_layout, axis, device):
    if is_hip():
        pytest.skip("test_scan_layouts is not supported in HIP")

    ir = f"""
    #blocked = {src_layout}
    module attributes {{"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32}} {{
    tt.func public @kernel_0d1d(%arg0: !tt.ptr<i32, 1> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32, 1> {{tt.divisibility = 16 : i32}}) {{
      %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #blocked>
      %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #blocked}}>>
      %1 = tt.expand_dims %0 {{axis = 1 : i32}} : (tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #blocked}}>>) -> tensor<{M}x1xi32, #blocked>
      %2 = arith.muli %1, %cst : tensor<{M}x1xi32, #blocked>
      %3 = tt.splat %arg0 : (!tt.ptr<i32, 1>) -> tensor<{M}x1x!tt.ptr<i32, 1>, #blocked>
      %4 = tt.addptr %3, %2 : tensor<{M}x1x!tt.ptr<i32, 1>, #blocked>, tensor<{M}x1xi32, #blocked>
      %5 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #blocked}}>>
      %6 = tt.expand_dims %5 {{axis = 0 : i32}} : (tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #blocked}}>>) -> tensor<1x{N}xi32, #blocked>
      %7 = tt.broadcast %4 : (tensor<{M}x1x!tt.ptr<i32, 1>, #blocked>) -> tensor<{M}x{N}x!tt.ptr<i32, 1>, #blocked>
      %8 = tt.broadcast %6 : (tensor<1x{N}xi32, #blocked>) -> tensor<{M}x{N}xi32, #blocked>
      %9 = tt.addptr %7, %8 : tensor<{M}x{N}x!tt.ptr<i32, 1>, #blocked>, tensor<{M}x{N}xi32, #blocked>
      %10 = tt.load %9 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}xi32, #blocked>
      %11 = "tt.scan"(%10) <{{axis = {axis} : i32}}> ({{
      ^bb0(%arg2: i32, %arg3: i32):
        %16 = arith.addi %arg2, %arg3 : i32
        tt.scan.return %16 : i32
      }}) : (tensor<{M}x{N}xi32, #blocked>) -> tensor<{M}x{N}xi32, #blocked>
      %12 = tt.splat %arg1 : (!tt.ptr<i32, 1>) -> tensor<{M}x1x!tt.ptr<i32, 1>, #blocked>
      %13 = tt.addptr %12, %2 : tensor<{M}x1x!tt.ptr<i32, 1>, #blocked>, tensor<{M}x1xi32, #blocked>
      %14 = tt.broadcast %13 : (tensor<{M}x1x!tt.ptr<i32, 1>, #blocked>) -> tensor<{M}x{N}x!tt.ptr<i32, 1>, #blocked>
      %15 = tt.addptr %14, %8 : tensor<{M}x{N}x!tt.ptr<i32, 1>, #blocked>, tensor<{M}x{N}xi32, #blocked>
      tt.store %15, %11 {{cache = 1 : i32, evict = 1 : i32}} : tensor<{M}x{N}xi32, #blocked>
      tt.return
    }}
    }}
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)
    rs = RandomState(17)
    x = rs.randint(-100, 100, (M, N)).astype('int32')

    z = np.zeros((M, N)).astype('int32')
    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    kernel[(1, 1, 1)](x_tri, z_tri)

    z_ref = np.cumsum(x, axis=axis)

    np.testing.assert_equal(z_ref, z_tri.cpu().numpy())


layouts = [
    BlockedLayout([1, 4], [8, 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, 4], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 4], [2, 16], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    MmaLayout(version=(2, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1], instr_shape=[16, 8]),
    MmaLayout(version=(2, 0), warps_per_cta=[2, 2], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1], instr_shape=[16, 8]),
    MmaLayout(version=(3, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0], instr_shape=[16, 16, 16]),
]


@pytest.mark.parametrize("M, N", [[128, 16], [128, 128], [32, 128], [32, 32]])
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reduce2d", [False, True])
@pytest.mark.parametrize("dtype_str", ["int32", "float32", "float16"])
@pytest.mark.parametrize("reduce_op", ["sum", "max"])
def test_reduce_layouts(M, N, src_layout, axis, reduce2d, dtype_str, reduce_op, device):
    if is_hip():
        pytest.skip("test_reduce_layouts is not supported in HIP")
    if reduce_op == "sum" and dtype_str == "float16" and M * N > 1024:
        pytest.skip("Skipping sum reduction on float16 due to accuracy issues")

    ty = {"int32": "i32", "float32": "f32", "float16": "f16"}[dtype_str]
    arith_op = {
        "max": {"int32": "arith.maxsi", "float32": "arith.maxf", "float16": "arith.maxf"},
        "sum": {"int32": "arith.addi", "float32": "arith.addf", "float16": "arith.addf"}
    }[reduce_op][dtype_str]
    numpy_op = {
        "max": np.max,
        "sum": np.sum
    }[reduce_op]
    rdims_1d = f"{N}" if axis == 0 else f"{M}"
    rdims_2d = f"1x{N}" if axis == 0 else f"{M}x1"
    store_range = "%7" if axis == 0 else "%1"
    blocked = BlockedLayout([1, 1], [32, 1], [4, 1], [0, 1], [1, 1], [1, 1], [0, 1])
    epilogue = f"""
        %14 = "tt.reduce"(%13) ({{
        ^bb0(%arg3: {ty}, %arg4: {ty}):
          %17 = {arith_op} %arg3, %arg4 : {ty}
          tt.reduce.return %17 : {ty}
        }}) {{axis = 0 : i32}} : (tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>>) -> {ty}
        tt.store %arg2, %14 {{cache = 1 : i32, evict = 1 : i32}} : {ty}
        tt.return
        }}
        }}
    """ if reduce2d else f"""
        %14 = tt.splat %arg2 : (!tt.ptr<{ty}, 1>) -> tensor<{rdims_2d}x!tt.ptr<{ty}, 1>, #blocked>
        %15 = tt.addptr %14, {store_range} : tensor<{rdims_2d}x!tt.ptr<{ty}>, #blocked>, tensor<{rdims_2d}xi32, #blocked>
        %16 = {GPU_DIALECT}.convert_layout %13 : (tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>>) -> tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #blocked}}>>
        %17 = tt.expand_dims %16 {{axis = {axis} : i32}} : (tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #blocked}}>>) -> tensor<{rdims_2d}x{ty}, #blocked>
        tt.store %15, %17 {{cache = 1 : i32, evict = 1 : i32}} : tensor<{rdims_2d}x{ty}, #blocked>
        tt.return
        }}
        }}
    """

    ir = f"""
    #blocked = {blocked}
    #src = {src_layout}
    module attributes {{"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32}} {{
    tt.func public @kernel_0d1d2c3d4c(%arg0: !tt.ptr<{ty}, 1> {{tt.divisibility = 16 : i32}}, %arg1: i32 {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}, 1> {{tt.divisibility = 16 : i32}}) {{
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #blocked}}>>
        %1 = tt.expand_dims %0 {{axis = 1 : i32}} : (tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #blocked}}>>) -> tensor<{M}x1xi32, #blocked>
        %2 = tt.splat %arg1 : (i32) -> tensor<{M}x1xi32, #blocked>
        %3 = arith.muli %1, %2 : tensor<{M}x1xi32, #blocked>
        %4 = tt.splat %arg0 : (!tt.ptr<{ty}, 1>) -> tensor<{M}x1x!tt.ptr<{ty}, 1>, #blocked>
        %5 = tt.addptr %4, %3 : tensor<{M}x1x!tt.ptr<{ty}, 1>, #blocked>, tensor<{M}x1xi32, #blocked>
        %6 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #blocked}}>>
        %7 = tt.expand_dims %6 {{axis = 0 : i32}} : (tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #blocked}}>>) -> tensor<1x{N}xi32, #blocked>
        %8 = tt.broadcast %5 : (tensor<{M}x1x!tt.ptr<{ty}, 1>, #blocked>) -> tensor<{M}x{N}x!tt.ptr<{ty}, 1>, #blocked>
        %9 = tt.broadcast %7 : (tensor<1x{N}xi32, #blocked>) -> tensor<{M}x{N}xi32, #blocked>
        %10 = tt.addptr %8, %9 : tensor<{M}x{N}x!tt.ptr<{ty}, 1>, #blocked>, tensor<{M}x{N}xi32, #blocked>
        %11 = tt.load %10 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}x{ty}, #blocked>
        %12 = {GPU_DIALECT}.convert_layout %11 : (tensor<{M}x{N}x{ty}, #blocked>) -> tensor<{M}x{N}x{ty}, #src>
        %13 = "tt.reduce"(%12) ({{
        ^bb0(%arg3: {ty}, %arg4: {ty}):
          %17 = {arith_op} %arg3, %arg4 : {ty}
          tt.reduce.return %17 : {ty}
        }}) {{axis = {axis} : i32}} : (tensor<{M}x{N}x{ty}, #src>) -> tensor<{rdims_1d}x{ty}, #{GPU_DIALECT}.slice<{{dim = {axis}, parent = #src}}>>
    """ + epilogue

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    rs = RandomState(17)
    x = numpy_random((M, N), dtype_str=dtype_str, rs=rs, low=0, high=10)
    z_shape = (1, 1) if reduce2d else (1, N) if axis == 0 else (M, 1)
    z = np.zeros(z_shape).astype(dtype_str)

    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    pgm = kernel[(1, 1, 1)](x_tri, x_tri.stride(0), z_tri)
    z_ref = numpy_op(x) if reduce2d else numpy_op(x, axis=axis, keepdims=True)

    if dtype_str == 'float16':
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-2)
    else:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)


layouts = [
    BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, 32], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    MmaLayout(version=(2, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1], instr_shape=[16, 8])
]


@pytest.mark.parametrize("M", [32, 64, 128, 256])
@pytest.mark.parametrize("src_layout", layouts)
def test_store_op(M, src_layout, device):
    if is_hip():
        pytest.skip("test_convert1d is not supported yet in HIP")

    ir = f"""
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "{GPU_DIALECT}.num-ctas" = 1 : i32, "{GPU_DIALECT}.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<f32, 1> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f32, 1> {{tt.divisibility = 16 : i32}}) {{
            %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %1 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<{M}x!tt.ptr<f32, 1>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %2 = tt.addptr %1, %0 : tensor<{M}x!tt.ptr<f32, 1>, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %3 = tt.load %2 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}xf32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %4 = tt.expand_dims %3 {{axis = 1 : i32}} : (tensor<{M}xf32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>) -> tensor<{M}x1xf32, #src>
            %5 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
            %6 = tt.expand_dims %5 {{axis = 1 : i32}} : (tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>) -> tensor<{M}x1xi32, #src>
            %7 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<{M}x1x!tt.ptr<f32, 1>, #src>
            %8 = tt.addptr %7, %6 : tensor<{M}x1x!tt.ptr<f32, 1>, #src>, tensor<{M}x1xi32, #src>
            tt.store %8, %4 : tensor<{M}x1xf32, #src>
            tt.return
        }}
    }}
    """

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        store_kernel = triton.compile(f.name)

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, 1)).astype('float32')
    y = np.zeros((M, 1), dtype='float32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)

    pgm = store_kernel[(1, 1, 1)](x_tri, y_tri)
    y_ref = x
    np.testing.assert_allclose(y_ref, y_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


layouts = [
    BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, 32], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    MmaLayout(version=(2, 0), warps_per_cta=[4, 1], ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[0, 1], instr_shape=[16, 8])
]


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("dst_layout", layouts)
@pytest.mark.parametrize("src_dim", [0, 1])
@pytest.mark.parametrize("dst_dim", [0, 1])
def test_convert1d(M, src_layout, dst_layout, src_dim, dst_dim, device):
    if is_hip():
        pytest.skip("test_convert1d is not supported in HIP")

    ir = f"""
    #dst = {dst_layout}
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
        tt.func public @kernel(%arg0: !tt.ptr<i32, 1> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32, 1> {{tt.divisibility = 16 : i32}}) {{
            %0 = tt.splat %arg0 : (!tt.ptr<i32, 1>) -> tensor<{M}x!tt.ptr<i32, 1>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %2 = tt.addptr %0, %1 : tensor<{M}x!tt.ptr<i32, 1>, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %3 = tt.load %2 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>
            %4 = tt.splat %arg1 : (!tt.ptr<i32, 1>) -> tensor<{M}x!tt.ptr<i32, 1>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %5 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %6 = tt.addptr %4, %5 : tensor<{M}x!tt.ptr<i32, 1>, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>, tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            %7 = {GPU_DIALECT}.convert_layout %3 : (tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {src_dim}, parent = #src}}>>) -> tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.store %6, %7 : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = {dst_dim}, parent = #dst}}>>
            tt.return
        }}
    }}
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, )).astype('int32')
    y = np.zeros((M, ), dtype='int32')
    x_tri = torch.tensor(x, device=device)
    y_tri = torch.tensor(y, device=device)
    pgm = kernel[(1, 1, 1)](x_tri, y_tri)
    y_ref = x
    np.testing.assert_allclose(y_ref, y_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


@triton.jit
def _welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = weight_2 / new_weight
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


layouts = [
    BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, 32], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [1, 32], [1, 4], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [8, 4], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1])
]


@pytest.mark.parametrize("M, N", [[128, 128], [256, 128], [256, 256], [128, 256]])
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("first_axis", [0, 1])
def test_chain_reduce(M, N, src_layout, op, device, first_axis):
    if is_hip():
        pytest.skip("test_chain_reduce is not supported in HIP")

    op_str = ""
    if op == "sum":
        op_str = f"""
        %13 = arith.addi %arg2, %arg3 : i32
        tt.reduce.return %13 : i32"""
    elif op == "max":
        op_str = f"""
        %13 = "{GPU_DIALECT}.cmpi"(%arg2, %arg3) <{{predicate = 4 : i64}}> : (i32, i32) -> i1
        %14 = arith.select %13, %arg2, %arg3 : i32
        tt.reduce.return %14 : i32"""
    ir = f"""
    #src = {src_layout}
    module attributes {{"{GPU_DIALECT}.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
    tt.func public @sum_kernel_0d1d(%arg0: !tt.ptr<i32, 1> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32, 1> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #src>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>
        %1 = tt.expand_dims %0 {{axis = 1 : i32}} : (tensor<{M}xi32, #{GPU_DIALECT}.slice<{{dim = 1, parent = #src}}>>) -> tensor<{M}x1xi32, #src>
        %2 = arith.muli %1, %cst : tensor<{M}x1xi32, #src>
        %3 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #src}}>>
        %4 = tt.expand_dims %3 {{axis = 0 : i32}} : (tensor<{N}xi32, #{GPU_DIALECT}.slice<{{dim = 0, parent = #src}}>>) -> tensor<1x{N}xi32, #src>
        %5 = tt.broadcast %2 : (tensor<{M}x1xi32, #src>) -> tensor<{M}x{N}xi32, #src>
        %6 = tt.broadcast %4 : (tensor<1x{N}xi32, #src>) -> tensor<{M}x{N}xi32, #src>
        %7 = arith.addi %5, %6 : tensor<{M}x{N}xi32, #src>
        %8 = tt.splat %arg0 : (!tt.ptr<i32, 1>) -> tensor<{M}x{N}x!tt.ptr<i32, 1>, #src>
        %9 = tt.addptr %8, %7 : tensor<{M}x{N}x!tt.ptr<i32, 1>, #src>, tensor<{M}x{N}xi32, #src>
        %10 = tt.load %9 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}xi32, #src>
        %11 = "tt.reduce"(%10) ({{
        ^bb0(%arg2: i32, %arg3: i32):
        {op_str}
        }}) {{axis = {first_axis} : i32}} : (tensor<{M}x{N}xi32, #src>) -> tensor<{M if first_axis == 1 else N}xi32, #{GPU_DIALECT}.slice<{{dim = {first_axis}, parent = #src}}>>
        %12 = "tt.reduce"(%11) ({{
        ^bb0(%arg2: i32, %arg3: i32):
        {op_str}
        }}) {{axis = 0 : i32}} : (tensor<{M if first_axis == 1 else N}xi32, #{GPU_DIALECT}.slice<{{dim = {first_axis}, parent = #src}}>>) -> i32
        tt.store %arg1, %12 {{cache = 1 : i32, evict = 1 : i32}} : i32
        tt.return
    }}
    }}
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)

    rs = RandomState(17)
    x = rs.randint(0, 4, (M, N)).astype('int32')

    z = np.zeros((1,)).astype('int32')

    x_tri = torch.tensor(x, device=device)
    z_tri = torch.tensor(z, device=device)

    pgm = kernel[(1, 1, 1)](x_tri, z_tri)
    if op == "sum":
        z_ref = np.sum(x)
    elif op == "max":
        z_ref = np.max(x)

    np.testing.assert_allclose(z_ref, z_tri.cpu().numpy(), rtol=0.01, atol=1e-3)


def test_generic_reduction(device):

    @triton.jit
    def var_mean_kernel(X, out_mean, out_var, BLOCK: tl.constexpr):
        xindex = tl.arange(0, BLOCK)
        x = tl.load(X + xindex)
        mean = x
        m2 = tl.zeros_like(x)
        weight = tl.full(x.shape, 1, x.dtype)
        (mean, m2, weight) = tl.reduce((mean, m2, weight), 0, _welford_combine)
        tl.store(out_mean, mean)
        tl.store(out_var, m2 / weight)

    SIZE = 512
    x = torch.rand(SIZE, device=device)
    out_mean = torch.empty((), device=device)
    out_var = torch.empty((), device=device)

    var_mean_kernel[(1,)](x, out_mean, out_var, BLOCK=SIZE)

    expect_var, expect_mean = torch.var_mean(x, dim=0, correction=0)
    torch.testing.assert_close(out_mean, expect_mean)
    torch.testing.assert_close(out_var, expect_var)


# ---------------
# test permute
# ---------------


@pytest.mark.parametrize("dtype_str, shape, perm",
                         [(dtype, shape, perm)
                          # TODO: bfloat16
                          for dtype in ['float8e4b15', 'float16', 'float32']
                             for shape in [(64, 64), (128, 128)]
                             for perm in [(1, 0)]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_permute(dtype_str, shape, perm, num_ctas, device):
    check_type_supported(dtype_str, device)  # bfloat16 on cc < 80 will not be tested
    if is_hip():
        pytest.skip(f"test_permute is not supported in HIP")

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xn,
               Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        tl.store(Zs, tl.load(Xs))
    # input
    x = numpy_random(shape, dtype_str=dtype_str)
    # triton result
    z_tri = to_triton(np.empty_like(x), device=device, dst_type=dtype_str)
    z_tri_contiguous = to_triton(np.empty_like(x), device=device, dst_type=dtype_str)
    x_tri = to_triton(x, device=device, dst_type=dtype_str)
    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                         z_tri, z_tri.stride(1), z_tri.stride(0),
                         BLOCK_M=shape[0], BLOCK_N=shape[1],
                         num_ctas=num_ctas)
    pgm_contiguous = kernel[(1, 1)](x_tri, x_tri.stride(1), x_tri.stride(0),
                                    z_tri_contiguous, z_tri_contiguous.stride(0), z_tri_contiguous.stride(1),
                                    BLOCK_M=shape[0], BLOCK_N=shape[1],
                                    num_ctas=num_ctas)
    # numpy result
    if dtype_str == 'float8e4b15':
        ty = tl.float8e4b15
        z_ref = serialize_fp8(deserialize_fp8(x, ty).T.copy(), ty)
        z_tri = z_tri.base
        z_tri_contiguous = z_tri_contiguous.base
    else:
        z_ref = x.transpose(*perm)
    # compare
    np.testing.assert_allclose(to_numpy(z_tri), z_ref)
    np.testing.assert_allclose(to_numpy(z_tri_contiguous), z_ref)

    if is_hip():
        return

    # parse ptx to make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx
    ptx = pgm_contiguous.asm['ptx']
    assert 'ld.global.v4' in ptx
    assert 'st.global.v4' in ptx


# ---------------
# test dot
# ---------------


@pytest.mark.parametrize("M, N, K, num_warps, col_a, col_b, epilogue, allow_tf32, in_dtype, out_dtype",
                         [(*shape, 4, False, False, epilogue, allow_tf32, in_dtype, out_dtype)
                          for shape in [(64, 64, 64), (32, 32, 32), (16, 16, 16)]
                          for epilogue in ['none', 'trans', 'add-matrix', 'add-rows', 'add-cols', 'softmax', 'chain-dot']
                          for allow_tf32 in [True, False]
                          for in_dtype, out_dtype in [('float16', 'float16'),
                                                      ('float16', 'float32'),
                                                      ('float32', 'float32')]
                          if not (allow_tf32 and (in_dtype in ['float16']))] +

                         [(*shape_nw, col_a, col_b, 'none', allow_tf32, in_dtype, out_dtype)
                          for shape_nw in [[128, 256, 32, 8],
                                           [128, 16, 32, 4],
                                           [32, 128, 64, 4],
                                           [128, 128, 64, 4],
                                           [64, 128, 128, 4],
                                           [32, 128, 64, 2],
                                           [64, 64, 32, 4],
                                           [32, 32, 128, 16],
                                           [128, 128, 64, 2],
                                           [64, 128, 128, 2]]
                          for allow_tf32 in [True]
                          for col_a in [True, False]
                          for col_b in [True, False]
                          for in_dtype, out_dtype in [('int8', 'int8'),
                                                      ('float16', 'float16'),
                                                      ('float16', 'float32'),
                                                      ('float32', 'float32')]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_dot(M, N, K, num_warps, col_a, col_b, epilogue, allow_tf32, in_dtype, out_dtype, num_ctas, device):
    check_cuda_only(device)

    capability = torch.cuda.get_device_capability()

    if is_hip():
        # set capability to large number to jump over check below
        # check are not relevant to amd gpu, left them for smaller diff between test_core.py and test_core_amd.py tests
        capability = (100, 100)
        if out_dtype is None:
            if in_dtype in float_dtypes:
                out_dtype = "float32"
            else:
                out_dtype = "int32"

    if capability[0] < 7:
        pytest.skip("Only test tl.dot() on devices with sm >= 70")
    if capability[0] < 8:
        if in_dtype == 'int8':
            pytest.skip("Only test int8 on devices with sm >= 80")
        elif allow_tf32:
            pytest.skip("Only test tf32 on devices with sm >= 80")
    if capability[0] == 7:
        if (M, N, K, num_warps) == (128, 256, 32, 8):
            pytest.skip("shared memory out of resource")
        if out_dtype == 'float16':
            # TODO: support out_dtype=float16 for tl.dot on V100
            pytest.skip("Only test out_dtype=float16 on devices with sm >=80")

    if is_hip():
        if (M, N, K) in [(64, 128, 128)]:
            pytest.skip(f"test_dot{(M, N, K)} not supported on HIP: memory out of resource.")
        if (M, N, K, num_warps) in [(128, 256, 32, 8), (128, 128, 64, 4)]:
            pytest.skip(f"test_dot{(M, N, K)} not supported on HIP. Reduce Warp to work")
        if M == 16 or N == 16 or K == 16:
            pytest.skip(f"test_dot{(M, N, K)} segfaults on HIP")
        if epilogue == "softmax":
            pytest.skip(f"test_dot{epilogue} segfaults on HIP")

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    if num_ctas > 1 and in_dtype == 'int8':
        # FIXME: mma v2 with num_ctas > 1 does not work
        pytest.skip()

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk,
               Y, stride_yk, stride_yn,
               W, stride_wn, stride_wl,
               Z, stride_zm, stride_zn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,
               ALLOW_TF32: tl.constexpr,
               DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,
               COL_A: tl.constexpr, COL_B: tl.constexpr,
               out_dtype: tl.constexpr = tl.float32):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_l = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Ws = W + off_n[:, None] * stride_wn + off_l[None, :] * stride_wl
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        x = tl.load(Xs)
        y = tl.load(Ys)
        z = tl.dot(x, y, allow_tf32=ALLOW_TF32, out_dtype=out_dtype)
        if ADD_MATRIX:
            z += tl.load(Zs)
        if ADD_ROWS:
            ZRs = Z + off_m * stride_zm
            z += tl.load(ZRs)[:, None]
        if ADD_COLS:
            ZCs = Z + off_n * stride_zn
            z += tl.load(ZCs)[None, :]
        if DO_SOFTMAX:
            max = tl.max(z, 1)
            z = z - max[:, None]
            num = tl.exp(z.to(tl.float32)).to(max.dtype)
            den = tl.sum(num, 1)
            z = num / den[:, None]
        if CHAIN_DOT:
            w = tl.load(Ws)
            z = tl.dot(z.to(w.dtype), w, allow_tf32=ALLOW_TF32, out_dtype=out_dtype)
        tl.store(Zs, z)
    # input
    rs = RandomState(17)
    if col_a:
        x = numpy_random((K, M), dtype_str=in_dtype, rs=rs).T
    else:
        x = numpy_random((M, K), dtype_str=in_dtype, rs=rs)
    if col_b:
        y = numpy_random((N, K), dtype_str=in_dtype, rs=rs).T
    else:
        y = numpy_random((K, N), dtype_str=in_dtype, rs=rs)
    w = numpy_random((N, N), dtype_str=in_dtype, rs=rs)
    if 'int' not in in_dtype:
        x *= .1
        y *= .1
    if in_dtype == 'float32' and allow_tf32:
        x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
        y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
        w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    w_tri = to_triton(w, device=device)
    # triton result
    if out_dtype == 'int8':
        z = 1 + numpy_random((M, N), dtype_str='int32', rs=rs)
    else:
        z = 1 + numpy_random((M, N), dtype_str=in_dtype, rs=rs) * .1

    z_tri = to_triton(z, device=device)
    if epilogue == 'trans':
        z_tri = torch.as_strided(z_tri, (M, N), z_tri.stride()[::-1])

    if out_dtype == 'int8':
        out_dtype = tl.int8
    elif out_dtype == 'float16' and epilogue != 'softmax':
        # TODO: for out_dtype == 'float16' and epilogue == 'softmax', it will
        # fail with the following error: 'llvm.fmul' op requires the same type
        # for all operands and results
        out_dtype = tl.float16
    else:
        out_dtype = tl.float32

    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                         y_tri, y_tri.stride(0), y_tri.stride(1),
                         w_tri, w_tri.stride(0), w_tri.stride(1),
                         z_tri, z_tri.stride(0), z_tri.stride(1),
                         COL_A=col_a, COL_B=col_b,
                         BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                         ADD_MATRIX=epilogue == 'add-matrix',
                         ADD_ROWS=epilogue == 'add-rows',
                         ADD_COLS=epilogue == 'add-cols',
                         DO_SOFTMAX=epilogue == 'softmax',
                         CHAIN_DOT=epilogue == 'chain-dot',
                         ALLOW_TF32=allow_tf32,
                         num_warps=num_warps, num_ctas=num_ctas,
                         out_dtype=out_dtype)

    if epilogue == 'softmax' and (in_dtype != 'float32' or allow_tf32):
        if is_hip():
            pass
        else:
            ptx = pgm.asm["ptx"]
            start = ptx.find("shfl.sync")
            end = ptx.find("cvt.rn.f16.f32")
            red_code = ptx[start:end]
            assert len(red_code) > 0
            import os
            enable_mmav3 = os.environ.get('ENABLE_MMA_V3', 'not found').lower()
            enable_tma = os.environ.get('ENABLE_TMA', 'not found').lower()
            # skip this check on hopper because there are some functions whose name contain "shared" in ptx.
            # TODO: we should eliminate these unused functions in ptx code.
            if not (enable_mmav3 in ["on", "true", "1"] and enable_tma in ["on", "true", "1"]):
                assert "shared" not in red_code
            assert "bar.sync" not in red_code
    # torch result
    if in_dtype == 'int8':
        z_ref = np.matmul(x.astype(np.float32),
                          y.astype(np.float32())).astype(np.int32)
    else:
        z_ref = np.matmul(x, y)

    if epilogue == 'add-matrix':
        z_ref += z
    if epilogue == 'add-rows':
        z_ref += z[:, 0][:, None]
    if epilogue == 'add-cols':
        z_ref += z[0, :][None, :]
    if epilogue == 'softmax':
        num = np.exp(z_ref - np.max(z_ref, axis=-1, keepdims=True))
        denom = np.sum(num, axis=-1, keepdims=True)
        z_ref = num / denom
    if epilogue == 'chain-dot':
        z_ref = np.matmul(z_ref, w)
    # compare
    if in_dtype == 'float32':
        # XXX: Somehow there's a larger difference when we use float32
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)
    elif out_dtype == tl.float16:
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-2)
    else:
        # added atol, to loose precision for float16xfloat16->float32 case
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01, atol=1e-3)
    if is_hip():
        return
    # make sure ld/st are vectorized
    ptx = pgm.asm['ptx']
    if (K > 16 or N > 16 or M > 16) and (M * N // (num_warps * 32) >= 4):
        # XXX: skip small sizes because they are not vectorized
        assert 'ld.global.v4' in ptx
        assert 'st.global.v4' in ptx
    if in_dtype == 'float32' and allow_tf32:
        assert re.search(r'[mma|wgmma.mma_async].sync.aligned.m\d+n\d+k8(?:.row.col)?.f32.tf32.tf32', ptx)
    elif in_dtype == 'float16' and out_dtype == tl.float32:
        if capability[0] == 7 and capability[1] == 5:  # Turing
            assert re.search(r'mma.sync.aligned.m\d+n\d+k8(?:.row.col)?.f32.f16.f16', ptx)
        else:
            assert re.search(r'[mma|wgmma.mma_async].sync.aligned.m\d+n\d+k16(?:.row.col)?.f32.f16.f16', ptx)
    elif in_dtype == 'float16' and out_dtype == tl.float16:
        if capability[0] == 7 and capability[1] == 5:  # Turing
            assert re.search(r'mma.sync.aligned.m\d+n\d+k8(?:.row.col)?.f16.f16.f16', ptx)
        else:
            assert re.search(r'[mma|wgmma.mma_async].sync.aligned.m\d+n\d+k16(?:.row.col)?.f16.f16.f16', ptx)
    elif in_dtype == 'int8':
        assert 'wgmma.mma_async.sync.aligned' in ptx or\
            'mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32' in ptx


@pytest.mark.parametrize('in_dtype', ['float32'])
def test_dot_mulbroadcastred(in_dtype, device):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        pytest.skip("Requires sm >= 80 to run")

    @triton.jit
    def kernel(Z, X, Y,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pidn = tl.program_id(1)
        pidm = tl.program_id(0)
        offm = tl.arange(0, BM)[:, None]
        offn = tl.arange(0, BN)[None, :]
        offak = tl.arange(0, BK)[None, :]
        offbk = tl.arange(0, BK)[:, None]
        acc = tl.full((BM, BN), 0.0, tl.float32)
        for ridx5 in range(0, K // BK):
            x = tl.load(X + ((pidm * K * BM) + (offm * K) + (ridx5 * BK) + offak))
            y = tl.load(Y + ((pidn * BN) + (offbk * N) + (ridx5 * N * BK) + offn))
            x = tl.expand_dims(x, axis=2)
            y = tl.expand_dims(y, axis=0)
            t = tl.sum(x * y, axis=1)
            acc = t + acc
        tl.store(Z + ((pidm * BM * N) + (pidn * BN) + (offm * N) + offn), acc)
    M, N, K = 256, 192, 160
    BM, BN, BK = 128, 32, 32
    rs = RandomState(17)
    x = numpy_random((M, K), dtype_str=in_dtype, rs=rs)
    y = numpy_random((K, N), dtype_str=in_dtype, rs=rs)
    x = x * 0.1
    y = y * 0.1
    z = numpy_random((M, N), dtype_str=in_dtype, rs=rs)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    z_tri = to_triton(z, device=device)
    grid = M // BM, N // BN
    h = kernel[grid](z_tri, x_tri, y_tri, M, N, K, BM, BN, BK)
    z_ref = np.matmul(x, y)
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), atol=0.01)

    if is_hip():
        return
    assert "tt.dot" in h.asm['ttir']
    # with option ENABLE_MMA_V3 on, we will not pipeline the load op for Y
    # as the loaded value is in rowmajor. But MMAv3 requires it's second
    # operand is in colmajor because transpose is not supported for MMAv3
    # with float32 input.
    import os
    enable_mmav3 = os.environ.get('ENABLE_MMA_V3', 'not found').lower()
    if enable_mmav3 in ["on", "true", "1"]:
        assert "triton_gpu.async_wait {num = 1 : i32}" in h.asm['ttgir']
    else:
        assert "triton_gpu.async_wait {num = 2 : i32}" in h.asm['ttgir']


@pytest.mark.parametrize("dtype_str", int_dtypes + uint_dtypes + float_dtypes + ['bfloat16'])
@pytest.mark.parametrize("shape", [(), (1,), (128,)])
def test_full(dtype_str, shape, device):
    if dtype_str in uint_dtypes and not hasattr(torch, dtype_str):
        # PyTorch only has unsigned 8, but not 16, 32, or 64
        dtype = getattr(torch, dtype_str[1:])  # uintx -> intx
    else:
        dtype = getattr(torch, dtype_str)
    check_type_supported(dtype, device)  # bfloat16 on cc < 80 will not be tested

    @triton.jit
    def kernel_static(out):
        a = GENERATE_TEST_HERE
        tl.static_assert(a.shape == SHAPE)
        out_ptr = out + tl.arange(0, 128)[:]
        tl.store(out_ptr, a)

    @triton.jit
    def kernel_dynamic(out, val, dtype: tl.constexpr):
        a = tl.full(SHAPE, val, dtype)
        tl.static_assert(a.shape == SHAPE)
        out_ptr = out + tl.arange(0, 128)[:]
        tl.store(out_ptr, a)

    kernel_static_patched = patch_kernel(kernel_static, {
        'GENERATE_TEST_HERE': f"tl.full({shape}, 2, tl.{dtype_str})",
        'SHAPE': str(list(shape)),
    })
    out_static = torch.zeros((128), dtype=dtype, device=device)
    kernel_static_patched[(1,)](out_static)
    assert torch.all(out_static == 2)

    kernel_dynamic_patched = patch_kernel(kernel_dynamic, {'SHAPE': str(list(shape))})
    out_dynamic = torch.zeros((128), dtype=dtype, device=device)
    kernel_dynamic_patched[(1,)](out_dynamic, 2, getattr(triton.language, dtype_str))
    assert torch.all(out_dynamic == 2)


@pytest.mark.parametrize("literal, dtype_str",
                         [(1e+50, "f64"), (1e+10, "f32"), (1.0, "f32"),
                          ('float("inf")', "f32"), ('float("-inf")', "f32"),
                          ('float("nan")', "f32"), ('float("-nan")', "f32"),
                          (0., "f32"),
                          (5, "i32"), (2**40, "i64"),])
def test_constexpr(literal, dtype_str, device):
    @triton.jit
    def kernel(out_ptr):
        val = GENERATE_TEST_HERE
        tl.store(out_ptr.to(tl.pointer_type(val.dtype)), val)

    kernel_patched = patch_kernel(kernel, {'GENERATE_TEST_HERE': f"{literal}"})
    out = torch.zeros((1,), dtype=torch.float32, device=device)
    h = kernel_patched[(1,)](out)
    assert re.search(r"arith.constant .* : " + dtype_str, h.asm["ttir"]) is not None


@pytest.mark.parametrize("dtype_str", ['float32', 'float16'])
def test_dot_without_load(dtype_str, device):
    capability = torch.cuda.get_device_capability()
    allow_tf32 = capability[0] > 7

    if is_hip() and dtype_str == "float16":
        pytest.skip("test_dot_without_load[float16] not supported in HIP")

    @triton.jit
    def _kernel(out, ALLOW_TF32: tl.constexpr):
        a = GENERATE_TEST_HERE
        b = GENERATE_TEST_HERE
        c = tl.dot(a, b, allow_tf32=ALLOW_TF32)
        out_ptr = out + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
        tl.store(out_ptr, c)
    kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.full((32, 32), 1.0, tl.{dtype_str})"})
    a = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device=device)
    b = torch.ones((32, 32), dtype=getattr(torch, dtype_str), device=device)
    out_ref = torch.matmul(a, b)
    out = torch.zeros((32, 32), dtype=getattr(torch, dtype_str), device=device)
    kernel[(1,)](out, ALLOW_TF32=allow_tf32)
    assert torch.all(out == out_ref)

# ---------------
# test arange
# ---------------


@pytest.mark.parametrize("start", [0, 1, 7, 16])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_arange(start, num_ctas, device):
    BLOCK = 128
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(z, BLOCK: tl.constexpr,
                START: tl.constexpr, END: tl.constexpr):
        off = tl.arange(0, BLOCK)
        val = tl.arange(START, END)
        tl.store(z + off, val)
    _kernel[(1,)](z_tri, START=start, END=start + BLOCK, BLOCK=BLOCK, num_ctas=num_ctas)
    z_ref = torch.arange(start, BLOCK + start, dtype=torch.int32, device=device)
    np.testing.assert_allclose(to_numpy(z_tri), to_numpy(z_ref))

# ---------------
# test load
# ---------------


@pytest.mark.parametrize("dtype_str, size, size_diff", [(dtype_str, size, size_diff) for dtype_str in torch_dtypes for size in [128, 512] for size_diff in [0, 1, 2, 3, 4]])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_masked_load(dtype_str, size, size_diff, num_ctas, device):
    dtype = getattr(torch, dtype_str)
    check_type_supported(dtype, device)  # bfloat16 on cc < 80 will not be tested

    input_size = size - size_diff
    output_size = size
    if dtype_str == 'bool':
        input = torch.randint(0, 2, (input_size,), dtype=dtype, device=device)
    elif dtype_str in int_dtypes or dtype_str in uint_dtypes:
        input = torch.randint(0, 127, (input_size,), dtype=dtype, device=device)
    else:
        input = torch.rand(input_size, dtype=dtype, device=device)
    output = torch.zeros((output_size,), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in_ptr, out_ptr, in_size: tl.constexpr, out_size: tl.constexpr):
        in_offsets = tl.arange(0, out_size)
        # Load inputs.
        x = GENERATE_TEST_HERE
        # Store output
        output_offsets = tl.arange(0, out_size)
        tl.store(out_ptr + output_offsets, x)

    mask_str = "mask=in_offsets < in_size, other=1" if size_diff > 0 else "None"
    kernel = patch_kernel(_kernel, {'GENERATE_TEST_HERE': f"tl.load(in_ptr + in_offsets, {mask_str})"})
    kernel[(1,)](input, output, input_size, output_size, num_ctas=num_ctas)

    reference_out = torch.cat((input, torch.ones((size_diff,), dtype=dtype, device=device)))
    # print((output - reference_out).nonzero())
    torch.testing.assert_close(output, reference_out)

# Testing masked loads with an intermate copy to shared memory run.


# FIXME: Shape too small for ldmatrix when num_ctas=4
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_masked_load_shared_memory(dtype, device):
    if is_hip():
        pytest.skip("test_masked_load_shared_memory is not supported in HIP")

    check_type_supported(dtype, device)  # bfloat16 on cc < 80 will not be tested

    M = 32
    N = 32
    K = 16

    in1 = torch.rand((M, K), dtype=dtype, device=device)
    in2 = torch.rand((K, N), dtype=dtype, device=device)
    out = torch.zeros((M, N), dtype=dtype, device=device)

    @triton.jit
    def _kernel(in1_ptr, in2_ptr, output_ptr,
                in_stride, in2_stride, out_stride,
                in_numel, in2_numel, out_numel,
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):

        M_offsets = tl.arange(0, M)
        N_offsets = tl.arange(0, N)
        K_offsets = tl.arange(0, K)

        in_offsets = M_offsets[:, None] * in_stride + K_offsets[None, :]
        in2_offsets = K_offsets[:, None] * in2_stride + N_offsets[None, :]

        # Load inputs.
        x = tl.load(in1_ptr + in_offsets, mask=in_offsets < M * K)
        w = tl.load(in2_ptr + in2_offsets, mask=in2_offsets < K * N)

        # Without a dot product the memory doesn't get promoted to shared.
        o = tl.dot(x, w, out_dtype=tl.float32)

        # Store output
        output_offsets = M_offsets[:, None] * out_stride + N_offsets[None, :]
        tl.store(output_ptr + output_offsets, o, mask=output_offsets < M * N)

    pgm = _kernel[(1,)](in1, in2, out,
                        in1.stride()[0],
                        in2.stride()[0],
                        out.stride()[0],
                        in1.numel(),
                        in2.numel(),
                        out.numel(),
                        M=M, N=N, K=K)

    reference_out = torch.matmul(in1, in2)
    torch.testing.assert_close(out, reference_out, atol=1e-2, rtol=0)


@pytest.mark.parametrize("cache", ["", ".ca", ".cg"])
def test_load_cache_modifier(cache, device):
    src = torch.empty(128, device=device)
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst, src, CACHE: tl.constexpr):
        offsets = tl.arange(0, 128)
        x = tl.load(src + offsets, cache_modifier=CACHE)
        tl.store(dst + offsets, x)

    pgm = _kernel[(1,)](dst, src, CACHE=cache)
    if is_hip():
        return

    ptx = pgm.asm['ptx']
    if cache == '':
        assert 'ld.global.ca' not in ptx
        assert 'ld.global.cg' not in ptx
    if cache == '.cg':
        assert 'ld.global.cg' in ptx
        assert 'ld.global.ca' not in ptx
    if cache == '.ca':
        assert 'ld.global.ca' in ptx
        assert 'ld.global.cg' not in ptx


@pytest.mark.parametrize("N", [16, 10, 11, 1024])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_vectorization(N, num_ctas, device):
    block_size = 1024 * num_ctas
    src = torch.empty(block_size, device=device)
    dst = torch.empty(block_size, device=device)

    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)
    pgm = _kernel[(1,)](
        dst, src, N=N, BLOCK_SIZE=block_size)

    if is_hip():
        return

    ptx = pgm.asm["ptx"]
    if N % 16 == 0:
        assert "ld.global.v4.b32" in ptx
    else:
        assert "ld.global.b32" in ptx
    # np.testing.assert_allclose(dst, src[:N])


@pytest.mark.parametrize("has_hints", [False, True])
def test_vectorization_hints(has_hints, device):
    src = torch.empty(1024, device=device)
    dst = torch.empty(1024, device=device)
    off = torch.zeros(1, device=device, dtype=torch.int32)

    @triton.jit
    def _kernel(dst, src, off, N, BLOCK_SIZE: tl.constexpr, HINT: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offsets = offsets + tl.load(off)
        if HINT:
            tl.max_contiguous(tl.multiple_of(offsets, 1024), 1024)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)
    pgm = _kernel[(1,)](dst, src, off, N=1024, BLOCK_SIZE=src.shape[0], HINT=has_hints)
    if is_hip():
        return

    ptx = pgm.asm["ptx"]
    if has_hints:
        assert "ld.global.v4.b32" in ptx
    else:
        assert "ld.global.v4.b32" not in ptx

# ---------------
# test store
# ---------------


@pytest.mark.parametrize("cache", ["", ".wb", ".cg", ".cs", ".wt"])
def test_store_cache_modifier(cache):
    src = torch.empty(128, device='cuda')
    dst = torch.empty(128, device='cuda')

    @triton.jit
    def _kernel(dst, src, CACHE: tl.constexpr):
        offsets = tl.arange(0, 128)
        x = tl.load(src + offsets)
        tl.store(dst + offsets, x, cache_modifier=CACHE)

    if is_hip():
        return
    pgm = _kernel[(1,)](dst, src, CACHE=cache)
    ptx = pgm.asm['ptx']
    if cache == '':
        assert 'st.global.wb' not in ptx
        assert 'st.global.cg' not in ptx
        assert 'st.global.cs' not in ptx
        assert 'st.global.wt' not in ptx
    if cache == '.wb':
        assert 'st.global.wb' in ptx
        assert 'st.global.cg' not in ptx
        assert 'st.global.cs' not in ptx
        assert 'st.global.wt' not in ptx
    if cache == '.cg':
        assert 'st.global.wb' not in ptx
        assert 'st.global.cg' in ptx
        assert 'st.global.cs' not in ptx
        assert 'st.global.wt' not in ptx
    if cache == '.cs':
        assert 'st.global.wb' not in ptx
        assert 'st.global.cg' not in ptx
        assert 'st.global.cs' in ptx
        assert 'st.global.wt' not in ptx
    if cache == '.wt':
        assert 'st.global.wb' not in ptx
        assert 'st.global.cg' not in ptx
        assert 'st.global.cs' not in ptx
        assert 'st.global.wt' in ptx

# ---------------
# test if
# ---------------

# ---------------
# test for
# ---------------

# ---------------
# test while
# ---------------

# ---------------
# test default
# ---------------
# TODO: can't be local to test_default


@triton.jit
def _impl(value=10):
    return value


def test_default(device):
    value = 5
    ret0 = torch.zeros(1, dtype=torch.int32, device=device)
    ret1 = torch.zeros(1, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(ret0, ret1, value=3):
        tl.store(ret0, _impl())
        tl.store(ret1, _impl(value))

    _kernel[(1,)](ret0, ret1, value)
    assert ret0.item() == 10
    assert ret1.item() == value

    _kernel[(1,)](ret0, ret1)
    assert ret0.item() == 10
    assert ret1.item() == 3

# ---------------
# test noop
# ----------------


def test_noop(device):
    @triton.jit
    def kernel(x):
        pass
    x = to_triton(numpy_random((1,), dtype_str='int32'), device=device)
    kernel[(1, )](x)


@pytest.mark.parametrize("device", ['cuda', 'cpu', 'cpu_pinned'])
def test_pointer_arguments(device):
    @triton.jit
    def kernel(x):
        pass
    pin_memory = 'pinned' in device
    x = torch.empty(1024, device=device.split('_')[0], pin_memory=pin_memory)
    if device == "cpu":
        with pytest.raises(ValueError):
            kernel[(1,)](x)
    else:
        kernel[(1, )](x)


@pytest.mark.parametrize("value, value_type", [
    (-1, 'i32'), (0, 'i32'), (-2**31, 'i32'), (2**31 - 1, 'i32'),
    (2**31, 'i64'), (2**32 - 1, 'i64'), (2**32, 'i64'), (2**63 - 1, 'i64'),
    (-2**63, 'i64'), (2**63, 'u64'), (2**64 - 1, 'u64')
])
def test_value_specialization(value: int, value_type: str, device) -> None:
    spec_type = None

    def cache_hook(*args, **kwargs):
        nonlocal spec_type
        spec_type = kwargs["compile"]["signature"][0]
    JITFunction.cache_hook = cache_hook

    @triton.jit
    def kernel(VALUE, X):
        pass

    x = torch.tensor([3.14159], device=device)
    pgm = kernel[(1, )](value, x)

    JITFunction.cache_hook = None
    assert spec_type == value_type

# --------------------
# value specialization
# --------------------


@pytest.mark.parametrize(
    "value, overflow",
    [(2**64 - 1, False), (2**64, True), (-2**63, False), (-2**63 - 1, True)]
)
def test_value_specialization_overflow(value: int, overflow: bool, device) -> None:

    @triton.jit
    def kernel(VALUE, X):
        pass

    x = torch.tensor([3.14159], device=device)

    if overflow:
        with pytest.raises(OverflowError):
            kernel[(1, )](value, x)
    else:
        kernel[(1, )](value, x)


# ----------------
# test constexpr
# ----------------

@pytest.mark.parametrize("op", ['+', '-', '*', '/', '%', '<', '>', '<<', '>>', '&', '^', '|'])
@pytest.mark.parametrize("is_lhs_constexpr", [False, True])
@pytest.mark.parametrize("is_rhs_constexpr", [True, False])
def test_bin_op_constexpr(op, is_lhs_constexpr, is_rhs_constexpr, device):
    if is_hip():
        if (is_rhs_constexpr, is_lhs_constexpr, op) in [(False, False, "<<"), (False, False, ">>"), (False, True, "<<")]:
            pytest.skip(f"test_bin_op_constexpr[{is_lhs_constexpr}-{is_rhs_constexpr}-{op}] is not supported in HIP")

    @triton.jit
    def kernel(Z, X, Y):
        x = tl.load(X)
        y = tl.load(Y)
        z = GENERATE_TEST_HERE
        tl.store(Z, z)

    if op in ['<<', '>>', '&', '^', '|']:  # int op
        x_str = "3" if is_lhs_constexpr else "x"
        y_str = "4" if is_rhs_constexpr else "y"
        x = numpy_random((1,), dtype_str="int32")
        y = numpy_random((1,), dtype_str="int32")
    else:
        x_str = "3.14" if is_lhs_constexpr else "x"
        y_str = "4.13" if is_rhs_constexpr else "y"
        x = numpy_random((1,), dtype_str="float32")
        y = numpy_random((1,), dtype_str="float32")
    kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f"{x_str} {op} {y_str}"})
    z = np.array(eval(f"{x_str} {op} {y_str}"))
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    z_tri = to_triton(np.empty((1,), dtype=z.dtype), device=device)
    kernel[(1,)](z_tri, x_tri, y_tri)
    np.testing.assert_allclose(z, to_numpy(z_tri))


def test_constexpr_shape(device):

    @triton.jit
    def kernel(X):
        off = tl.arange(0, 128 + 128)
        tl.store(X + off, off)

    x_tri = to_triton(np.empty((256, ), dtype=np.int32), device=device)
    kernel[(1,)](x_tri)
    np.testing.assert_equal(to_numpy(x_tri), np.arange(0, 256))


def test_constexpr_scalar_shape(device):

    @triton.jit
    def kernel(X, s):
        off = tl.arange(0, 256)
        val = off % (256 // s)
        tl.store(X + off, val)

    x_tri = to_triton(np.empty((256, ), dtype=np.int32), device=device)
    kernel[(1,)](x_tri, 32)
    np.testing.assert_equal(to_numpy(x_tri), np.arange(0, 256) % 8)

# -------------
# test call
# -------------


@triton.jit
def val_multiplier(val, i):
    return val * i


@triton.jit(noinline=True)
def val_multiplier_noinline(val, i):
    return val * i


@triton.jit
def vecmul_kernel(ptr, n_elements, rep, type: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * 128 + tl.arange(0, 128)
    mask = offsets < n_elements
    vec = tl.load(ptr + offsets, mask=mask)
    for i in range(1, rep):
        if type == "inline":
            vec = val_multiplier(vec, i)
        else:
            vec = val_multiplier_noinline(vec, i)
    tl.store(ptr + offsets, vec, mask=mask)


@pytest.mark.parametrize("type", ["inline", "noinline"])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_call(type, num_ctas, device):

    @triton.jit
    def kernel(ptr, n_elements, num1, num2, type: tl.constexpr):
        vecmul_kernel(ptr, n_elements, num1, type)
        vecmul_kernel(ptr, n_elements, num2, type)

    size = 1024
    rand_val = numpy_random((size,), dtype_str="float32")
    rand_val_tri = to_triton(rand_val, device=device)
    err_msg = ""
    try:
        kernel[(size // 128,)](rand_val_tri, size, 3, 5, type, num_ctas=num_ctas)
    except Exception as e:
        err_msg = str(e)

    if type == "noinline":
        assert err_msg is not ""
    else:
        ans = rand_val * 1 * 2 * 1 * 2 * 3 * 4
        np.testing.assert_equal(to_numpy(rand_val_tri), ans)

# -------------
# test if
# -------------

# TODO(Keren): if_exp_dynamic


@pytest.mark.parametrize("if_type", ["if", "if_and_dynamic", "if_exp_static", "if_and_static"])
def test_if(if_type, device):

    @triton.jit
    def kernel(Cond, XTrue, XFalse, Ret, IfType: tl.constexpr, BoolVar: tl.constexpr, StaticVaue: tl.constexpr):
        pid = tl.program_id(0)
        cond = tl.load(Cond)
        if IfType == "if":
            if pid % 2 == 0:
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_exp_dynamic":
            tl.store(Ret, tl.load(XTrue)) if pid % 2 == 0 else tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_exp_static":
            tl.store(Ret, tl.load(XTrue)) if BoolVar else tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_and_dynamic":
            if BoolVar and pid % 2 == 0:
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_and_static":
            if StaticVaue != 0 and StaticVaue != 0:
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))

    cond = torch.ones(1, dtype=torch.int32, device=device)
    x_true = torch.tensor([3.14], dtype=torch.float32, device=device)
    x_false = torch.tensor([1.51], dtype=torch.float32, device=device)
    ret = torch.zeros(1, dtype=torch.float32, device=device)

    kernel[(1,)](cond, x_true, x_false, ret, if_type, True, 1)
    assert torch.equal(ret, x_true)


def test_num_warps_pow2(device):
    dst = torch.empty(128, device=device)

    @triton.jit
    def _kernel(dst):
        pass

    with pytest.raises(AssertionError, match='must be a power of 2'):
        _kernel[(1,)](dst=dst, num_warps=3)
    _kernel[(1,)](dst=dst, num_warps=1)
    _kernel[(1,)](dst=dst, num_warps=2)
    _kernel[(1,)](dst=dst, num_warps=4)

# -------------
# test extern
# -------------


@pytest.mark.parametrize("dtype_str, expr, lib_path",
                         [('int32', 'math.ffs', ''),
                          ('float32', 'math.log2', ''),
                          ('float32', 'math.scalbn', ''),
                          ('float32', 'math.pow', tl.math.libdevice_path()),
                          ('float64', 'math.pow_dtype', tl.math.libdevice_path()),
                          ('float64', 'math.norm4d', '')])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_math_tensor(dtype_str, expr, lib_path, num_ctas, device):

    if is_hip() and expr == "math.scalbn":
        pytest.skip("test_math_tensor[math.scalbn] is not supported in HIP")

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)

    shape = (128, )
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random(shape, dtype_str=dtype_str, rs=rs)

    if expr == 'math.log2':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.broadcast_to(tl.{expr}(5.0), x.shape)'})
        y_ref = np.log2(5.0)
    elif expr == 'math.ffs':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{expr}(x)'})
        y_ref = np.zeros(shape, dtype=x.dtype)
        for i in range(shape[0]):
            y_ref[i] = (int(x[i]) & int(-x[i])).bit_length()
    elif expr == 'math.scalbn':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{expr}(x, 2)'})
        y_ref = x * pow(2, 2)
    elif expr == 'math.pow_dtype':
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.math.pow(x, 0.5)'})
        y_ref = np.power(x, 0.5)
    elif expr == 'math.pow':
        # numpy does not allow negative factors in power, so we use abs()
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{expr}(x, x)'})
        y_ref = np.power(x, x)
    elif expr == 'math.pow_dtype':
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.math.pow(x, 0.5)'})
        y_ref = np.power(x, 0.5)
    elif expr == 'math.norm4d':
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': f'tl.{expr}(x, x, x, x)'})
        y_ref = np.sqrt(4 * np.power(x, 2))

    x_tri = to_triton(x, device=device)
    # triton result
    y_tri = to_triton(numpy_random((shape[0],), dtype_str=dtype_str, rs=rs), device=device)
    kernel[(1,)](x_tri, y_tri, BLOCK=shape[0], extern_libs={'libdevice': lib_path}, num_ctas=num_ctas)
    # compare
    if expr == 'math.ffs':
        np.testing.assert_equal(y_ref, to_numpy(y_tri))
    else:
        np.testing.assert_allclose(y_ref, to_numpy(y_tri), rtol=0.01)


@pytest.mark.parametrize("dtype_str, expr, lib_path",
                         [('float32', 'math.pow', ''),
                          ('float64', 'math.pow_dtype', ''),
                          ('float64', 'math.pow', tl.math.libdevice_path())])
@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_math_scalar(dtype_str, expr, lib_path, num_ctas, device):

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = X
        y = GENERATE_TEST_HERE
        tl.store(Y + tl.arange(0, BLOCK), y)

    shape = (128, )
    rs = RandomState(17)
    # limit the range of integers so that the sum does not overflow
    x = numpy_random((1,), dtype_str=dtype_str, rs=rs)
    y_ref = np.zeros(shape, dtype=x.dtype)

    # numpy does not allow negative factors in power, so we use abs()
    if expr == 'math.pow':
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.math.pow(x, x)'})
        y_ref[:] = np.power(x, x)
    elif expr == 'math.pow_dtype':
        x = np.abs(x)
        kernel = patch_kernel(kernel, {'GENERATE_TEST_HERE': 'tl.math.pow(x, 0.5)'})
        y_ref[:] = np.power(x, 0.5)

    # triton result
    x_tri = to_triton(x, device=device)[0].item()
    y_tri = to_triton(numpy_random((shape[0],), dtype_str=dtype_str, rs=rs), device=device)
    kernel[(1,)](x_tri, y_tri, BLOCK=shape[0], extern_libs={'libdevice': lib_path}, num_ctas=num_ctas)
    # compare
    np.testing.assert_allclose(y_ref, to_numpy(y_tri), rtol=0.01)


# -----------------------
# test inline asm
# -----------------------

@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_inline_asm(num_ctas, device):
    check_cuda_only(device)

    if is_hip():
        pytest.skip("test_inline_asm is not supported in HIP")

    @triton.jit
    def kernel(X, Y, Z, n: tl.constexpr, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        y = tl.load(Y + tl.arange(0, BLOCK))
        s = tl.full([BLOCK], n, tl.int32)
        z = tl.inline_asm_elementwise("shf.l.wrap.b32 $0, $1, $2, $3;", "=r,r, r, r", [x, y, s], dtype=tl.int32, is_pure=True, pack=1)
        tl.store(Z + tl.arange(0, BLOCK), z)

    shape = (128, )
    rs = RandomState(17)
    x = numpy_random(shape, dtype_str='uint32', rs=rs)
    y = numpy_random(shape, dtype_str='uint32', rs=rs)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    n = 17
    z_tri = to_triton(numpy_random(shape, dtype_str='uint32', rs=rs), device=device)
    kernel[(1,)](x_tri, y_tri, z_tri, n, BLOCK=shape[0], num_ctas=num_ctas)
    y_ref = (y << n) | (x >> (32 - n))
    # compare
    np.testing.assert_equal(y_ref, to_numpy(z_tri))


@pytest.mark.parametrize("num_ctas", num_ctas_list)
def test_inline_asm_packed(num_ctas, device):
    check_cuda_only(device)

    if is_hip():
        pytest.skip("test_inline_asm is not supported in HIP")

    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK))
        # shift 4x8bits values together.
        y = tl.inline_asm_elementwise("and.b32 $0, $1, 0x1F1F1F1F; \
                                       shl.b32 $0, $0, 3;",
                                      "=r,r", [x,], dtype=tl.int8, is_pure=True, pack=4)
        tl.store(Y + tl.arange(0, BLOCK), y)

    shape = (512, )
    rs = RandomState(17)
    x = numpy_random(shape, dtype_str='uint8', rs=rs)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(numpy_random(shape, dtype_str='uint8', rs=rs), device=device)
    kernel[(1,)](x_tri, y_tri, BLOCK=shape[0], num_ctas=num_ctas)
    y_ref = x << 3
    # compare
    np.testing.assert_equal(y_ref, to_numpy(y_tri))

# -----------------------
# test control flow
# -----------------------


@pytest.mark.parametrize("lo, hi, iv", [(2**35, 2**35 + 20, 1), (2**35, 2**35 + 20, 2), (2**35, 2**35 + 20, 3),
                                        (15, -16, -1), (15, -16, -2), (15, -16, -3),
                                        (-18, -22, -1), (22, 18, -1)])
def test_for_iv(lo, hi, iv, device):

    @triton.jit
    def kernel(Out, lo, hi, iv: tl.constexpr):
        acc = 0
        acc = acc.to(tl.int64)
        for i in range(lo, hi, iv):
            acc += i
        tl.store(Out, acc)

    lo = 2**35
    hi = 2**35 + 20
    out = to_triton(np.zeros((1,), dtype=np.int64), device=device)
    kernel[(1,)](out, lo, hi, iv)
    assert out[0] == sum(range(lo, hi, iv))


def test_if_else(device):

    @triton.jit
    def kernel(Cond, TrueVal, FalseVal, Out):
        if tl.load(Cond):
            val = tl.load(TrueVal)
        else:
            val = tl.load(FalseVal)
        tl.store(Out, val)

    out = to_triton(np.zeros((1,), dtype=np.int32), device=device)
    true_val = to_triton(np.full((1,), 1, dtype=np.int32), device=device)
    false_val = to_triton(np.full((1,), 2, dtype=np.int32), device=device)
    cond = to_triton(np.zeros((1,), dtype=np.int32), device=device)
    # True
    cond[0] = True
    kernel[(1,)](cond, true_val, false_val, out)
    assert to_numpy(out)[0] == true_val[0]
    # False
    cond[0] = False
    kernel[(1,)](cond, true_val, false_val, out)
    assert to_numpy(out)[0] == false_val[0]


@pytest.mark.parametrize("mode", ["dynamic", "static"])
def test_if_return(mode, device):

    @triton.jit
    def kernel(ExitEarly, Out, cond: tl.constexpr, mode: tl.constexpr):
        if mode == "dynamic":
            if tl.load(ExitEarly):
                tl.store(Out, 0)
                return
        else:
            if cond:
                tl.store(Out, 0)
                return
        tl.store(Out, 1)

    out = to_triton(np.zeros((1,), dtype=np.int32), device=device)
    exit_early = to_triton(np.zeros((1,), dtype=np.int32), device=device)
    # exit early path taken
    exit_early[0] = 1
    kernel[(1,)](exit_early, out, True, mode)
    assert to_numpy(out)[0] == 0
    # exit early path not taken
    exit_early[0] = 0
    kernel[(1,)](exit_early, out, False, mode)
    assert to_numpy(out)[0] == 1


@triton.jit
def add_fn(x):
    return x + 1


@triton.jit(noinline=True)
def add_fn_noinline(x):
    return x + 1


@triton.jit
def add_fn_return(x, pid):
    if pid == 0:
        return x + 1
    else:
        return x + 2


@triton.jit
def add_fn_expr(Out, x):
    tl.store(Out, x)


@triton.jit
def add_fn_static_cond(x, cond: tl.constexpr):
    if cond == "":
        return x
    else:
        return x + 1


# TODO(Keren): if_exp
@pytest.mark.parametrize("call_type", ["attribute", "attribute_jit",
                                       "jit", "jit_if", "jit_expr",
                                       "jit_static_cond", "jit_noinline", "jit_extern"])
def test_if_call(call_type, device):
    @triton.jit
    def kernel(Out, call_type: tl.constexpr):
        pid = tl.program_id(0)
        o = tl.load(Out)
        if call_type == "attribute":
            # call attribute
            if pid == 0:
                a = o
                a = a.to(tl.int32).to(tl.int32) + 1
                o = a
        elif call_type == "attribute_jit":
            # call attribute and jit function
            if pid == 0:
                a = o
                a = tl.load(Out + add_fn(a) - 1).to(tl.int32) + 1
                o = a
        elif call_type == "jit":
            if pid == 0:
                # regular function call
                a = o
                a = add_fn(a)
                o = a
        elif call_type == "jit_if":
            # function without end_if block
            if pid == 0:
                a = o
                a = add_fn_return(a, pid)
                o = a
        elif call_type == "jit_if_exp":
            # ifexp expression
            if pid == 0:
                a = o
                a = add_fn(a) if pid == 0 else add_fn_return(a, pid)
                o = a
        elif call_type == "jit_expr":
            # call without return
            if pid == 0:
                a = o + 1
                add_fn_expr(Out, a)
                o = a
        elif call_type == "jit_static_cond":
            if pid == 0:
                a = o + 1
                add_fn_static_cond(o, call_type)
                o = a
        elif call_type == "jit_noinline":
            if pid == 0:
                a = o + 1
                add_fn_noinline(a)
                o = a
        elif call_type == "jit_extern":
            if pid == 0:
                a = o + 1
                tl.cdiv(a, a)
                o = a

        tl.store(Out, o)

    out = to_triton(np.zeros((1,), dtype=np.int32), device=device)
    kernel[(1,)](out, call_type)
    assert to_numpy(out)[0] == 1


@pytest.mark.parametrize("_cond1", [True, False])
@pytest.mark.parametrize("_cond2", [True, False])
@pytest.mark.parametrize("_cond3", [True, False])
def test_nested_if_else_return(_cond1, _cond2, _cond3, device):

    @triton.jit
    def kernel(Cond1, Cond2, Cond3, Val1, Val2, Val3, Out):
        val = 0
        if tl.load(Cond1):
            if tl.load(Cond2):
                val = tl.load(Val1)
            else:
                return
        else:
            if tl.load(Cond3):
                val = tl.load(Val2)
            else:
                val = tl.load(Val3)
        tl.store(Out, val)

    out = to_triton(np.full((1,), -1, dtype=np.int32), device=device)
    cond1 = to_triton(np.full((1,), _cond1, dtype=np.int32), device=device)
    cond2 = to_triton(np.full((1,), _cond2, dtype=np.int32), device=device)
    cond3 = to_triton(np.full((1,), _cond3, dtype=np.int32), device=device)
    val1 = to_triton(np.full((1,), 1, dtype=np.int32), device=device)
    val2 = to_triton(np.full((1,), 2, dtype=np.int32), device=device)
    val3 = to_triton(np.full((1,), 3, dtype=np.int32), device=device)
    kernel[(1,)](cond1, cond2, cond3, val1, val2, val3, out)
    targets = {
        (True, True, True): val1[0],
        (True, True, False): val1[0],
        (True, False, True): out[0],
        (True, False, False): out[0],
        (False, True, True): val2[0],
        (False, True, False): val3[0],
        (False, False, True): val2[0],
        (False, False, False): val3[0],
    }
    assert out[0] == targets[(_cond1, _cond2, _cond3)]


def test_while(device):

    @triton.jit
    def kernel(InitI, Bound, CutOff, OutI, OutInitI, OutJ):
        init_i = tl.load(InitI)
        curr_i = init_i
        j = 0
        # Check that init_i is not updated by the loop
        while j < tl.load(Bound):
            curr_i = curr_i + (j == tl.load(CutOff))
            j += 1
            tl.store(OutInitI, init_i)
        tl.store(OutI, curr_i)
        tl.store(OutJ, j)

    out_i = to_triton(np.zeros((1,), dtype=np.int32), device=device)
    out_j = to_triton(np.zeros((1,), dtype=np.int32), device=device)
    init_i = to_triton(np.full((1,), 1, dtype=np.int32), device=device)
    out_init_i = to_triton(np.full((1,), 0, dtype=np.int32), device=device)
    bound = to_triton(np.full((1,), 10, dtype=np.int32), device=device)
    cut_off = to_triton(np.full((1,), 5, dtype=np.int32), device=device)
    kernel[(1,)](init_i, bound, cut_off, out_i, out_init_i, out_j)
    assert out_init_i[0] == init_i[0]
    assert out_i[0] == init_i[0] + 1
    assert out_j[0] == bound[0]


def test_while(device):
    @triton.jit
    def nested_while(data, countPtr):
        for i in range(10):
            count = tl.load(countPtr)
            while count > 0:
                tl.store(data, tl.load(data) + 1.0)
                count = count - 2

    counter = torch.tensor([8], dtype=torch.int32, device=device)
    data = torch.zeros((1,), device=device, dtype=torch.float32)
    nested_while[(1,)](data, counter)
    assert data[0] == 40


# def test_for_if(device):

#     @triton.jit
#     def kernel(bound, cutoff, M, N):
#         m = 0
#         n = 0
#         for i in range(bound):
#             if i > cutoff:
#                 m = m + 1
#             else:
#                 n = n + 1
#         tl.store(M, m)
#         tl.store(N, n)

#     m = to_triton(np.zeros((1,), dtype=np.int32), device=device)
#     n = to_triton(np.zeros((1,), dtype=np.int32), device=device)
#     kernel[(1,)](10, 7, m, n)
#     print(m[0])
#     print(n[0])

# -----------------------
# test extra
# -----------------------


def test_globaltimer(device):
    if is_hip():
        pytest.skip("test_globaltimer is not supported in HIP")
    check_cuda_only(device)

    @triton.jit
    def kernel(Out1, Out2):
        start = tl.extra.cuda.globaltimer()
        off = tl.arange(0, 128)
        for i in range(10000):
            tl.store(Out1 + off, tl.load(Out1 + off) + 1)
        end = tl.extra.cuda.globaltimer()
        tl.store(Out2, end - start)

    out1 = to_triton(np.zeros((128,), dtype=np.int64), device=device)
    out2 = to_triton(np.zeros((1,), dtype=np.int64), device=device)
    h = kernel[(1,)](out1, out2)
    assert out2[0] > 0
    assert h.asm["ptx"].count("%globaltimer") == 2


def test_smid(device):
    if is_hip():
        pytest.skip("test_smid is not supported in HIP")
    check_cuda_only(device)

    @triton.jit
    def kernel(Out):
        tl.store(Out + tl.program_id(0), tl.extra.cuda.smid())

    out = to_triton(np.zeros((1024,), dtype=np.int32), device=device)
    h = kernel[(out.shape[0],)](out)
    assert out.sort()[0].unique().shape[0] > 0
    assert h.asm["ptx"].count("%smid") == 1

# -----------------------
# test layout conversions
# -----------------------
# TODO: backend should be tested separately


layouts = [
    # MmaLayout(1, [1, 4], [1, 1], [0, 1]),
    # MmaLayout((2, 0), [1, 4], [1, 1], [1, 1], [0, 1], [16, 8]),
    # MmaLayout(1, [4, 1], [1, 1], [0, 1]),
    # MmaLayout((2, 0), [4, 1], [1, 1], [1, 1], [0, 1], [16, 8]),
    BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 4], [4, 8], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 1], [1, 32], [2, 2], [1, 0], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([8, 1], [16, 2], [1, 4], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 1], [8, 4], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([1, 1], [32, 1], [2, 2], [0, 1], [1, 1], [1, 1], [0, 1]),
    BlockedLayout([4, 4], [1, 32], [4, 1], [1, 0], [1, 1], [1, 1], [0, 1])
]

intermediate_layouts = [
    None,
    SharedLayout(1, 1, 1, [1, 0], [1, 1], [1, 1], [0, 1]),
    SharedLayout(4, 2, 4, [1, 0], [1, 1], [1, 1], [0, 1]),
    SharedLayout(2, 2, 4, [1, 0], [1, 1], [1, 1], [0, 1]),
]


@pytest.mark.parametrize("M, N", [[64, 1], [64, 64], [128, 128], [1, 64]])
@pytest.mark.parametrize("dtype", ['float16'])
@pytest.mark.parametrize("src_layout", layouts)
@pytest.mark.parametrize("interm_layout", intermediate_layouts)
@pytest.mark.parametrize("dst_layout", layouts)
def test_convert2d(M, N, src_layout, interm_layout, dst_layout, dtype, device):
    if is_hip():
        pytest.skip("test_convert2d is not supported in HIP")
    if (M == 1 or N == 1) and interm_layout:
        pytest.skip("Out of bound access when maxPhase > 1")
    if str(src_layout) == str(dst_layout):
        pytest.skip()
    if 'mma' in str(src_layout) and 'mma' in str(dst_layout):
        pytest.skip()

    layouts = f"""
    #src = {src_layout}
    #dst = {dst_layout}
    """ if interm_layout is None else f"""
    #src = {src_layout}
    #interm = {interm_layout}
    #dst = {dst_layout}
    """

    conversion = f"""
    %12 = triton_gpu.convert_layout %9 : (tensor<{M}x{N}xi32, #src>) -> tensor<{M}x{N}xi32, #dst>
    %13 = triton_gpu.convert_layout %11 : (tensor<{M}x{N}xf16, #src>) -> tensor<{M}x{N}xf16, #dst>
    """ if interm_layout is None else f"""
    %15 = triton_gpu.convert_layout %9 : (tensor<{M}x{N}xi32, #src>) -> tensor<{M}x{N}xi32, #interm>
    %16 = triton_gpu.convert_layout %15 : (tensor<{M}x{N}xi32, #interm>) -> tensor<{M}x{N}xi32, #src>
    %17 = triton_gpu.convert_layout %11 : (tensor<{M}x{N}xf16, #src>) -> tensor<{M}x{N}xf16, #interm>
    %18 = triton_gpu.convert_layout %17 : (tensor<{M}x{N}xf16, #interm>) -> tensor<{M}x{N}xf16, #src>

    %12 = triton_gpu.convert_layout %16 : (tensor<{M}x{N}xi32, #src>) -> tensor<{M}x{N}xi32, #dst>
    %13 = triton_gpu.convert_layout %18 : (tensor<{M}x{N}xf16, #src>) -> tensor<{M}x{N}xf16, #dst>
    """

    ir = layouts + f"""
    module attributes {{"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32}} {{
  tt.func public @kernel_0d1d(%arg0: !tt.ptr<f16, 1> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<f16, 1> {{tt.divisibility = 16 : i32}}) {{
    %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #src>
    %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #src}}>>
    %1 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #src}}>>
    %2 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<{M}x{N}x!tt.ptr<f16, 1>, #src>
    %4 = tt.expand_dims %0 {{axis = 1 : i32}} : (tensor<{M}xi32, #triton_gpu.slice<{{dim = 1, parent = #src}}>>) -> tensor<{M}x1xi32, #src>
    %5 = arith.muli %4, %cst : tensor<{M}x1xi32, #src>
    %6 = tt.expand_dims %1 {{axis = 0 : i32}} : (tensor<{N}xi32, #triton_gpu.slice<{{dim = 0, parent = #src}}>>) -> tensor<1x{N}xi32, #src>
    %7 = tt.broadcast %6 : (tensor<1x{N}xi32, #src>) -> tensor<{M}x{N}xi32, #src>
    %8 = tt.broadcast %5 : (tensor<{M}x1xi32, #src>) -> tensor<{M}x{N}xi32, #src>
    %9 = arith.addi %8, %7 : tensor<{M}x{N}xi32, #src>
    %10 = tt.addptr %2, %9 : tensor<{M}x{N}x!tt.ptr<f16, 1>, #src>, tensor<{M}x{N}xi32, #src>
    %11 = tt.load %10 {{cache = 1 : i32, evict = 1 : i32, isVolatile = false}} : tensor<{M}x{N}xf16, #src>
    %3 = tt.splat %arg1 : (!tt.ptr<f16, 1>) -> tensor<{M}x{N}x!tt.ptr<f16, 1>, #dst>
    """ + conversion + f"""
    %14 = tt.addptr %3, %12 : tensor<{M}x{N}x!tt.ptr<f16, 1>, #dst>, tensor<{M}x{N}xi32, #dst>
    tt.store %14, %13 : tensor<{M}x{N}xf16, #dst>
    tt.return
  }}
}}
"""

    x = to_triton(numpy_random((M, N), dtype_str=dtype), device=device)
    z = torch.empty_like(x)

    # write the IR to a temporary file using mkstemp
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir') as f:
        f.write(ir)
        f.flush()
        kernel = triton.compile(f.name)
    kernel[(1, 1, 1)](x.data_ptr(), z.data_ptr())

    assert torch.equal(z, x)


def test_load_scalar_with_mask(device):
    @triton.jit
    def kernel(Input, Index, Out, N: int):
        index = tl.load(Index)
        scalar = tl.load(Input + index, mask=index < N, other=0)
        tl.store(Out, scalar, mask=index < N)
    Index = torch.tensor([0], dtype=torch.int32, device=device)
    Input = torch.tensor([0], dtype=torch.int32, device=device)
    Out = torch.empty_like(Index, device=device)
    kernel[(1,)](Input, Index, Out, Index.numel())
    assert Out.data[0] == 0


# This test is used to test our own PTX codegen for float16 and int16 conversions
# maybe delete it later after ptxas has been fixed
@pytest.mark.parametrize("dtype_str", ['float16', 'int16'])
def test_ptx_cast(dtype_str, device):
    @triton.jit
    def kernel(in_ptr0, out_ptr2, xnumel, rnumel, dtype: tl.constexpr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        rbase = tl.arange(0, RBLOCK)[None, :]
        x0 = xindex
        _tmp4 = (tl.zeros([XBLOCK, RBLOCK], dtype) - 10000).to(dtype)
        for roffset in range(0, rnumel, RBLOCK):
            rindex = roffset + rbase
            rmask = rindex < rnumel
            r1 = rindex
            tmp0 = tl.load(in_ptr0 + (r1 + (197 * x0)), rmask & xmask).to(dtype)
            tmp1 = 2
            tmp2 = tmp0 * tmp1
            tmp3 = tmp2.to(dtype)
            tmp5 = _tmp4 < tmp3
            _tmp4 = tl.where(rmask & xmask & tmp5, tmp3, _tmp4)
            tl.store(out_ptr2 + (r1 + (197 * x0) + tl.zeros([XBLOCK, RBLOCK], tl.int32)), _tmp4, rmask & xmask)

    torch.manual_seed(123)
    if dtype_str == 'int16':
        torch_dtype = torch.int16
        triton_dtype = tl.int32
    else:
        torch_dtype = torch.float16
        triton_dtype = tl.float32

    s0 = 4
    buf11 = -torch.ones((6 * s0, 197, 197), device=device, dtype=torch_dtype)
    buf14 = -torch.ones((s0, 6, 197, 197), device=device, dtype=torch_dtype)
    kernel[(4728,)](buf11, buf14, 1182 * s0, 197, triton_dtype, 1, 256, num_warps=2)
    assert buf14.to(torch.float32).mean() == -2.0

# -----------------------
# test fp8 -> fp32 dot
# -----------------------


def f8_to_f16(x, dtype):
    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    dtype = getattr(tl, dtype)
    kernel[grid](ret, triton.reinterpret(x, dtype), ret.numel(), BLOCK_SIZE=1024)
    return ret


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    low_precision_acc: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator, max_num_imprecise_acc=low_precision_acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)


@pytest.mark.parametrize("in_type_str", ['float8e5', 'float8e4nv'])
@pytest.mark.parametrize("low_precision_acc", [0, 32, 64, 128])
def test_fp8_dot_acc(in_type_str, low_precision_acc, device):
    check_type_supported(in_type_str, device)
    M, N, K = 128, 256, 256
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 128
    A = numpy_random((M, K), dtype_str=in_type_str)
    B = numpy_random((K, N), dtype_str=in_type_str)
    Bt = B.T
    C = torch.empty((M, N), dtype=torch.float32, device='cuda')
    num_warps = 8
    a = to_triton(A, device='cuda', dst_type=in_type_str)
    b = to_triton(B, device='cuda', dst_type=in_type_str)
    grid = (triton.cdiv(M, BLOCK_M), 1)
    matmul_kernel[grid](a, b, C, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(
                            1), C.stride(0), C.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K, low_precision_acc, num_warps=num_warps)
    torch_a = torch.from_numpy(A)
    th_a = f8_to_f16(torch_a.cuda(), in_type_str)
    torch_b = torch.from_numpy(B)
    th_b = f8_to_f16(torch_b.cuda(), in_type_str)
    ref_out = torch.matmul(th_a, th_b).to(torch.float32)
    if in_type_str == 'float8e4nv':
        torch.testing.assert_close(ref_out, C, rtol=0.01, atol=0.01)
    elif low_precision_acc > 32:
        torch.testing.assert_close(ref_out, C, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(ref_out, C)
