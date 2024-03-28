import inspect

import math
import numpy as np

import triton
import triton.language as tl
from dataclasses import dataclass
from .errors import InterpreterError
from .._C.libtriton import interpreter as _interpreter


class TensorHandle:

    def __init__(self, data, dtype):
        self.data = data
        self.dtype = dtype

    def __bool__(self):
        return bool(self.data.all())

    def get_element_ty(self):
        dtype = self.dtype
        while hasattr(dtype, "element_ty"):
            dtype = dtype.element_ty
        return dtype

    def clone(self):
        return TensorHandle(self.data.copy(), self.dtype)


class BlockPointerHandle:

    def __init__(self, base, shape, strides, offsets, tensor_shape, order):
        self.base = base
        self.shape = shape
        self.strides = strides
        self.offsets = offsets
        self.tensor_shape = tensor_shape
        self.order = order

    def materialize_pointers(self, boundary_check):
        dtype_tt = self.base.dtype.element_ty
        n_bytes = dtype_tt.primitive_bitwidth // 8
        tensor_shape = self.tensor_shape
        ptrs = np.broadcast_to(self.base.data, self.tensor_shape)
        masks = np.ones(self.tensor_shape, dtype=bool)
        for dim in range(len(tensor_shape)):
            bcast_dims = [1] * len(tensor_shape)
            bcast_dims[dim] = tensor_shape[dim]
            off = (self.offsets[dim].data + np.arange(tensor_shape[dim])).reshape(bcast_dims)
            ptrs = ptrs + (n_bytes * off * self.strides[dim].data).astype(np.uint64)
            if dim in boundary_check:
                masks = np.logical_and(masks, off < self.shape[dim].data)
        ptrs = TensorHandle(ptrs, self.base.dtype)
        return ptrs, masks


@dataclass(frozen=True)
class InterpreterOptions:
    extern_libs: dict = None
    debug: bool = False
    arch: str = None
    allow_fp8e4nv: bool = False
    max_num_imprecise_acc_default: int = 0


def _get_signed_np_dtype(dtype):
    if dtype == np.uint8:
        return np.int8
    if dtype == np.uint16:
        return np.int16
    if dtype == np.uint32:
        return np.int32
    if dtype == np.uint64:
        return np.int64
    return dtype


def _get_np_dtype(tt_dtype):
    if isinstance(tt_dtype, tl.pointer_type):
        return np.dtype(np.uint64)
    np_types = {
        tl.int1: np.dtype(bool),
        tl.float16: np.dtype(np.float16),
        tl.float32: np.dtype(np.float32),
        tl.float64: np.dtype(np.float64),
        tl.int8: np.dtype(np.int8),
        tl.uint8: np.dtype(np.uint8),
        tl.int16: np.dtype(np.int16),
        tl.uint16: np.dtype(np.uint16),
        tl.int32: np.dtype(np.int32),
        tl.uint32: np.dtype(np.uint32),
        tl.int64: np.dtype(np.int64),
        tl.uint64: np.dtype(np.uint64),
    }
    if isinstance(tt_dtype, tl.block_type):
        if isinstance(tt_dtype.element_ty, tl.pointer_type):
            return np.dtype(np.uint64)
        return np_types[tt_dtype.element_ty]
    return np_types[tt_dtype]


def _erf(x):
    # Numpy does not support erf
    return math.erf(x)


def _umulhi_64(a, b):
    # Numpy does not support 128-bit multiplication
    # So we have to implement it manually
    return (int(a) * int(b)) >> 64


np_erf_fp32 = np.vectorize(_erf, otypes=[np.float32])
np_erf_fp64 = np.vectorize(_erf, otypes=[np.float64])
np_umulhi_u64 = np.vectorize(_umulhi_64, otypes=[np.uint64])


class InterpreterBuilder:

    def __init__(self) -> None:
        self.arch = None
        self.options = InterpreterOptions()
        # pass

    def set_grid_idx(self, x, y, z):
        if not x < self.grid_dim[0]:
            raise ValueError("x >= grid_dim[0]")
        if not y < self.grid_dim[1]:
            raise ValueError("y >= grid_dim[1]")
        if not z < self.grid_dim[2]:
            raise ValueError("z >= grid_dim[2]")
        self.grid_idx = (x, y, z)

    def set_grid_dim(self, nx, ny, nz):
        self.grid_dim = (nx, ny, nz)

    # constants

    def get_half_ty(self):
        return tl.float16

    def get_float_ty(self):
        return tl.float32

    def get_double_ty(self):
        return tl.float64

    def get_int8_ty(self):
        return tl.int8

    def get_uint8_ty(self):
        return tl.uint8

    def get_int16_ty(self):
        return tl.int16

    def get_uint16_ty(self):
        return tl.uint16

    def get_int32_ty(self):
        return tl.int32

    def get_uint32_ty(self):
        return tl.uint32

    def get_int64_ty(self):
        return tl.int64

    def get_uint64_ty(self):
        return tl.uint64

    def get_ptr_ty(self, elt_ty, addr_space):
        return tl.pointer_type(elt_ty, addr_space)

    def get_block_ty(self, dtype, shape):
        return tl.block_type(dtype, shape)

    def get_int1(self, value):
        return TensorHandle(np.array([value], dtype=np.bool_), tl.int1)

    def get_uint8(self, value):
        return TensorHandle(np.array([value], dtype=np.uint8), tl.uint8)

    def get_int8(self, value):
        return TensorHandle(np.array([value], dtype=np.int8), tl.int8)

    def get_uint16(self, value):
        return TensorHandle(np.array([value], dtype=np.uint16), tl.uint16)

    def get_int16(self, value):
        return TensorHandle(np.array([value], dtype=np.int16), tl.int16)

    def get_uint32(self, value):
        return TensorHandle(np.array([value], dtype=np.uint32), tl.uint32)

    def get_int32(self, value):
        return TensorHandle(np.array([value], dtype=np.int32), tl.int32)

    def get_uint64(self, value):
        return TensorHandle(np.array([value], dtype=np.uint64), tl.uint64)

    def get_int64(self, value):
        return TensorHandle(np.array([value], dtype=np.int64), tl.int64)

    def get_fp16(self, value):
        return TensorHandle(np.array([value], dtype=np.float16), tl.float16)

    def get_fp32(self, value):
        return TensorHandle(np.array([value], dtype=np.float32), tl.float32)

    def get_fp64(self, value):
        return TensorHandle(np.array([value], dtype=np.float64), tl.float64)

    def get_null_value(self, type):
        return TensorHandle(np.array([0], dtype=_get_np_dtype(type)), type)

    # programming model
    def create_get_program_id(self, axis):
        if self.grid_idx is None:
            raise ValueError("grid_idx is None")
        return TensorHandle(np.array([self.grid_idx[axis]], dtype=np.int32), tl.int32)

    def create_get_num_programs(self, axis):
        return TensorHandle(np.array([self.grid_dim[axis]], dtype=np.int32), tl.int32)

    # memory ops
    def create_load(self, ptr, _0, _1, is_volatile):
        mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
        other = None
        return self.create_masked_load(ptr, mask, other, _0, _1, is_volatile)

    def create_store(self, ptr, val, _0, _1):
        mask = TensorHandle(np.ones_like(ptr.data, dtype=bool), tl.int1)
        return self.create_masked_store(ptr, val, mask, None, None)

    def create_masked_load(self, ptrs, mask, other, cache_modifier, eviction_policy, is_volatile):
        dtype_tt = ptrs.get_element_ty()
        dtype_np = _get_np_dtype(dtype_tt)
        if other is None:
            other = TensorHandle(np.ones_like(ptrs.data, dtype=dtype_np), dtype_tt)
        ret = _interpreter.load(ptrs.data, mask.data, other.data, dtype_np)
        return TensorHandle(ret, dtype_tt)

    def create_masked_store(self, ptrs, value, mask, cache_modifier, eviction_policy):
        return _interpreter.store(ptrs.data, value.data, mask.data)

    # casting ops
    def cast_impl(self, src, dst_type):
        return TensorHandle(src.data.astype(_get_np_dtype(dst_type)), dst_type)

    create_si_to_fp = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_ui_to_fp = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_to_si = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_to_ui = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_ext = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_fp_trunc = lambda self, src, dst_type: self.cast_impl(src, dst_type)
    create_int_cast = lambda self, src, dst_type, is_signed: self.cast_impl(src, dst_type)

    def create_fp_to_fp(self, src, dst_type):
        raise NotImplementedError("fp_to_fp not supported in interpreter mode")

    def create_bitcast(self, src, dst_type):
        return TensorHandle(src.data.view(_get_np_dtype(dst_type)), dst_type)

    # binary operators
    def binary_op(self, lhs, rhs, op):
        return TensorHandle(op(lhs.data, rhs.data), lhs.dtype)

    create_fadd = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.add)
    create_fmul = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.multiply)
    create_fdiv = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.divide)
    create_frem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.remainder)
    create_fsub = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.subtract)
    create_mul = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.multiply)
    create_precise_divf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.divide)
    create_sdiv = lambda self, lhs, rhs: self.create_idiv(lhs, rhs)
    create_udiv = lambda self, lhs, rhs: self.create_idiv(lhs, rhs)
    # LLVM has 'numpy.fmod', not 'numpy.remainder', semantics on integer remainders.
    create_srem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.fmod)
    create_urem = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.fmod)
    create_add = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.add)
    create_sub = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.subtract)
    create_shl = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.left_shift)
    create_lshr = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.right_shift)
    create_minsi = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minui = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minimumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_minnumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.minimum)
    create_maxsi = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maxui = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maximumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_maxnumf = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.maximum)
    create_icmpSLE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_icmpSLT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_icmpSGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_icmpSGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_icmpULE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_icmpULT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_icmpUGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_icmpUGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_icmpEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_icmpNE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_fcmpOLT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_fcmpOGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_fcmpOLE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_fcmpOGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_fcmpOEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_fcmpONE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_fcmpULT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less)
    create_fcmpUGT = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater)
    create_fcmpULE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.less_equal)
    create_fcmpUGE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.greater_equal)
    create_fcmpUEQ = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.equal)
    create_fcmpUNE = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.not_equal)
    create_and = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_and)
    create_xor = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_xor)
    create_or = lambda self, lhs, rhs: self.binary_op(lhs, rhs, np.bitwise_or)

    def create_idiv(self, lhs, rhs):
        # Triton has IEEE, not numpy/torch, semantics for %, and those carry
        # through to //, so we have to use a nonstandard expression to get a
        # reference result for //.
        return TensorHandle((lhs.data - np.fmod(lhs.data, rhs.data)) // rhs.data, lhs.dtype)

    def create_ashr(self, lhs, rhs):
        # Triton's rshift operator depends on the signedness of the left operand
        lhs_dtype = _get_signed_np_dtype(lhs.data.dtype)
        rhs_dtype = _get_signed_np_dtype(rhs.data.dtype)
        lhs.data = lhs.data.astype(lhs_dtype)
        rhs.data = rhs.data.astype(rhs_dtype)
        return self.binary_op(lhs, rhs, np.right_shift)

    def create_umulhi(self, lhs, rhs):
        dtype = lhs.data.dtype
        if dtype == np.int64 or dtype == np.uint64:
            return TensorHandle(np_umulhi_u64(lhs.data, rhs.data), lhs.dtype)
        else:
            compute_dtype = getattr(np, f"uint{dtype.itemsize * 8 * 2}")
            lhs_data = lhs.data.astype(compute_dtype)
            rhs_data = rhs.data.astype(compute_dtype)
            ret_data = np.multiply(lhs_data, rhs_data) >> (dtype.itemsize * 8)
            return TensorHandle(ret_data.astype(dtype), lhs.dtype)

    # ternary functions
    def ternary_op(self, lhs, rhs, other, op):
        return TensorHandle(op(lhs.data, rhs.data, other.data), other.dtype)

    create_clampf = lambda self, arg, lo, hi, propagate_nans: self.ternary_op(arg, lo, hi, np.clip)
    create_select = lambda self, cond, lhs, rhs: self.ternary_op(cond, lhs, rhs, np.where)

    def create_fma(self, x, y, z):
        return TensorHandle(x.data * y.data + z.data, z.dtype)

    # unary functions
    def unary_op(self, arg, op):
        return TensorHandle(op(arg.data), arg.dtype)

    create_cos = lambda self, arg: self.unary_op(arg, np.cos)
    create_exp = lambda self, arg: self.unary_op(arg, np.exp)
    create_exp2 = lambda self, arg: self.unary_op(arg, np.exp2)
    create_fabs = lambda self, arg: self.unary_op(arg, np.abs)
    create_iabs = lambda self, arg: self.unary_op(arg, np.abs)
    create_floor = lambda self, arg: self.unary_op(arg, np.floor)
    create_log = lambda self, arg: self.unary_op(arg, np.log)
    create_log2 = lambda self, arg: self.unary_op(arg, np.log2)
    create_precise_sqrt = lambda self, arg: self.unary_op(arg, np.sqrt)
    create_sqrt = lambda self, arg: self.unary_op(arg, np.sqrt)
    create_sin = lambda self, arg: self.unary_op(arg, np.sin)

    def create_erf(self, arg):
        ret = np_erf_fp32(arg.data) if arg.data.dtype == np.float32 else np_erf_fp64(arg.data)
        return TensorHandle(ret, arg.dtype)

    # tensor operators
    create_reshape = lambda self, arg, shape, allow_reorder: TensorHandle(arg.data.reshape(shape), arg.dtype)

    def create_trans(self, arg, perm):
        return TensorHandle(np.transpose(arg.data, perm), arg.dtype)

    def create_dot(self, a, b, d, allow_tf32, max_num_imprecise_acc):
        return TensorHandle(np.matmul(a.data, b.data) + d.data, d.dtype)

    def create_make_range(self, start, stop):
        return TensorHandle(np.arange(start, stop, dtype=np.int32), tl.int32)

    def create_histogram(self, data, bins):
        return TensorHandle(np.histogram(data.data, bins=bins, range=(0, bins))[0], tl.int32)

    # pointer arithmetic

    def create_addptr(self, ptr, offset):
        dtype_tt = ptr.get_element_ty()
        element_bitwidth = dtype_tt.primitive_bitwidth
        # int1's bitwidth is 1, but we need to use 8 for pointer arithmetic
        element_bytewidth = max(1, element_bitwidth // 8)
        return TensorHandle(ptr.data + element_bytewidth * offset.data.astype(np.uint64), ptr.dtype)

    def create_tensor_pointer_load(self, ptr, boundary_check, padding_option, cache_modifier, eviction_policy,
                                   is_volatile):
        ptrs, masks = ptr.materialize_pointers(boundary_check)
        if padding_option is not None:
            raise NotImplementedError("padding_option not supported in interpreter mode")
        other = None
        return self.create_masked_load(ptrs, masks, other, cache_modifier, eviction_policy, is_volatile)

    def create_tensor_pointer_store(self, ptr, value, boundary_check, cache_modifier, eviction_policy):
        ptrs, masks = ptr.materialize_pointers(boundary_check)
        return self.create_masked_store(ptrs, value, masks, cache_modifier, eviction_policy)

    def create_expand_dims(self, arg, axis):
        return TensorHandle(np.expand_dims(arg.data, axis), arg.dtype)

    def create_broadcast(self, arg, shape):
        return TensorHandle(np.broadcast_to(arg.data, shape), arg.dtype)

    def create_int_to_ptr(self, val, dst_ty):
        return TensorHandle(val.data.astype(np.uint64), dst_ty)

    def create_ptr_to_int(self, val, dst_ty):
        return TensorHandle(val.data.astype(np.uint64), dst_ty)

    def create_cat(self, lhs, rhs):
        return TensorHandle(np.concatenate([lhs.data, rhs.data]), lhs.dtype)

    def create_join(self, lhs, rhs):
        # Triton only supports joining two original tensors into a new one along the last axis
        return TensorHandle(np.stack([lhs.data, rhs.data], axis=-1), lhs.dtype)

    def create_split(self, val):
        # Triton only supports splitting the original tensor into two along the last axis
        return (TensorHandle(val.data[..., 0], val.dtype), TensorHandle(val.data[..., 1], val.dtype))

    def create_splat(self, arg, shape):
        if isinstance(arg.dtype, tl.block_type):
            return TensorHandle(np.full(shape, arg.data[0], dtype=_get_np_dtype(arg.dtype)), arg.dtype)
        else:  # scalar
            block_type = tl.block_type(arg.dtype, shape)
            return TensorHandle(np.full(shape, arg.data, dtype=_get_np_dtype(arg.dtype)), block_type)

    # def create_atomic_cas(self, ptr, cmp, val, sem):
    #     pass

    # def create_atomic_rmw(self, rmwOp, ptr, val, mask, sem):
    #     pass

    # def create_extern_elementwise(self, libName, libPath, symbol, argList, retType, isPure):
    #     pass

    # def create_int_to_ptr(self, val, type):
    #     pass

    # def create_inline_asm(self, inlineAsm, constraints, values, type, isPure, pack):
    #     pass

    # def create_print(self, prefix, values):
    #     pass

    # def create_assert(self, condition, message, fileName, funcName, lineNo):
    #     pass

    # def create_undef(self, type):
    #     pass

    # def create_barrier(self):
    #     pass

    def create_make_block_ptr(self, base, shape, strides, offsets, tensor_shape, order):
        # Create new offsets to avoid modifying the original
        new_offsets = [offset.clone() for offset in offsets]
        return BlockPointerHandle(base, shape, strides, new_offsets, tensor_shape, order)

    def create_advance(self, ptr, offsets):
        if len(ptr.offsets) != len(offsets):
            raise ValueError("len(ptr.offsets) != len(offsets)")
        # Create new offsets to avoid modifying the original
        new_offsets = [offset.clone() for offset in ptr.offsets]
        ret = BlockPointerHandle(ptr.base, ptr.shape, ptr.strides, new_offsets, ptr.tensor_shape, ptr.order)
        for i in range(len(offsets)):
            ret.offsets[i].data += offsets[i].data
        return ret

    def get_all_ones_value(self, type):
        np_type = _get_np_dtype(type)
        if "int" in np_type.name:
            return TensorHandle(np.full(1, -1, dtype=np_type), type)
        else:
            raise TypeError(f"unsupported type {type}")


def _patch_attr(obj, name, member, builder):
    new_member = lambda *args, member=member, **kwargs: (member(*args, **
                                                                {k: v
                                                                 for k, v in kwargs.items()
                                                                 if k != "_builder"}, _builder=builder))
    setattr(obj, name, new_member)


def _patch_builtin(pkg, builder):
    for name, member in inspect.getmembers(pkg):
        if tl.core.is_builtin(member):
            _patch_attr(pkg, name, member, builder)


def _patch_lang_tensor(tensor):

    def _get_bool(self):
        data = self.handle.data
        # in triton, only scalars can be converted to booleans
        # here we need this hack because all scalars are tensors
        return bool(data) if data.size == 1 else True

    def _get_transpose(self):
        return tl.core.tensor(TensorHandle(np.transpose(self.handle.data), self.handle.dtype), self.dtype)

    tensor.__index__ = lambda self: int(self.handle.data)
    tensor.__bool__ = lambda self: _get_bool(self)
    tensor.__repr__ = lambda self: repr(self.handle.data)
    tensor.__str__ = lambda self: str(self.handle.data)
    tensor.T = property(_get_transpose)


class ReduceScanOpIneterface:

    def __init__(self, axis, combine_fn):
        self.axis = axis
        self.combine_fn = combine_fn

    def check_axis(self, shape, axis):
        if axis is not None and axis >= len(shape):
            raise ValueError(f"axis {axis} out of bounds for shape {shape}")

    def check_tensor(self, input):
        for arg in input:
            if not isinstance(arg, tl.core.tensor):
                raise ValueError(f"input must be a tensor, got {type(arg)}")
            self.check_axis(arg.shape, self.axis)

    def to_tensor(self, ret, dtype):
        if hasattr(ret, "shape") and ret.shape:
            ret_type = tl.block_type(dtype, ret.shape)
        else:
            ret = np.array([ret], dtype=_get_np_dtype(dtype))
            ret_type = dtype
        return tl.core.tensor(TensorHandle(ret, dtype), ret_type)

    def apply(self, input):
        if not isinstance(input, tuple):
            input = (input, )
        self.check_tensor(input)
        return self.apply_impl(input)

    def apply_impl(self, input):
        raise NotImplementedError("apply_impl not implemented")


class ReduceOps(ReduceScanOpIneterface):

    def __init__(self, axis, combine_fn, keep_dims):
        super().__init__(axis, combine_fn)
        self.keep_dims = keep_dims

    def unravel(self, input, axis):
        ret = []
        for data in input:
            if axis is not None:
                ret.append(data)
            else:
                axis = 0
                ret.append(self.to_tensor(data.handle.data.flatten(), data.dtype))
        return tuple(ret), axis

    def generic_reduce(self, input):
        original_axis = self.axis
        input, axis = self.unravel(input, self.axis)
        input_data = []
        output_data = []
        input_shape = input[0].handle.data.shape
        output_shape = input_shape[0:axis] + input_shape[axis + 1:]
        for arg in input:
            input_data.append(arg.handle.data)
            output_data.append(np.zeros(output_shape, dtype=arg.handle.data.dtype))
        # Reduce on axis
        for i in range(input_data[0].size):
            # Recover input_index from i using input_shape
            input_index = np.unravel_index(i, input_shape)
            output_index = input_index[0:axis] + input_index[axis + 1:]
            input_tuple = tuple(self.to_tensor(d[input_index], input[ii].dtype) for ii, d in enumerate(input_data))
            if input_index[axis] == 0:
                # First element
                for j in range(len(output_data)):
                    output_data[j][output_index] = input_tuple[j].handle.data.item()
            else:
                acc_tuple = tuple(self.to_tensor(o[output_index], input[oi].dtype) for oi, o in enumerate(output_data))
                combine_fn_ret = self.combine_fn.fn(*acc_tuple, *input_tuple)
                acc_tuple = (combine_fn_ret, ) if not isinstance(combine_fn_ret, tuple) else combine_fn_ret
                for j in range(len(output_data)):
                    output_data[j][output_index] = acc_tuple[j].handle.data.item() if isinstance(
                        acc_tuple[j], tl.core.tensor) else acc_tuple[j]
        # Pack output
        ret = []
        for i, data in enumerate(output_data):
            if self.keep_dims:
                if original_axis is not None:
                    data = np.expand_dims(data, axis)
                else:
                    for _ in range(len(input_shape)):
                        data = np.expand_dims(data, 0)

            elif original_axis is None:
                # Take a scalar
                data = data.item()
            ret.append(self.to_tensor(data, input[i].dtype))
        return ret[0] if len(ret) == 1 else tuple(ret)

    def min_max(self, input, val_reduce_op, idx_reduce_op=None):
        # If input is a tuple, it must be (val, index), and we only take val
        input = input[0] if isinstance(input, tuple) else input
        val = None
        idx = None
        if val_reduce_op:
            val = self.to_tensor(val_reduce_op(input.handle.data, axis=self.axis, keepdims=self.keep_dims), input.dtype)
        if idx_reduce_op:
            idx = self.to_tensor(idx_reduce_op(input.handle.data, axis=self.axis, keepdims=self.keep_dims), input.dtype)
        if val and idx:
            return val, idx
        elif val:
            return val
        elif idx:
            return idx
        else:
            raise ValueError("val_reduce_op and idx_reduce_op are both None")

    def sum(self, input):
        return self.to_tensor(np.sum(input.handle.data, axis=self.axis, keepdims=self.keep_dims), input.dtype)

    def apply_impl(self, input):
        if self.combine_fn == tl.standard._argmin_combine_tie_break_left:
            return self.min_max(input[0], val_reduce_op=np.min, idx_reduce_op=np.argmin)
        elif self.combine_fn == tl.standard._argmax_combine_tie_break_left:
            return self.min_max(input[0], val_reduce_op=np.max, idx_reduce_op=np.argmax)
        elif self.combine_fn == tl.standard._elementwise_max:
            return self.min_max(input[0], val_reduce_op=np.max, idx_reduce_op=None)
        elif self.combine_fn == tl.standard._elementwise_min:
            return self.min_max(input[0], val_reduce_op=np.min, idx_reduce_op=None)
        elif self.combine_fn == tl.standard._sum_combine:
            return self.sum(input[0])
        else:
            # Fall back to the slow mode
            return self.generic_reduce(input)


class ScanOps(ReduceScanOpIneterface):

    def __init__(self, axis, combine_fn, reverse):
        super().__init__(axis, combine_fn)
        self.reverse = reverse

    def cumsum(self, input):
        return [self.to_tensor(np.cumsum(input.handle.data, axis=self.axis), dtype=input.dtype)]

    def cumprod(self, input):
        return [self.to_tensor(np.cumprod(input.handle.data, axis=self.axis), dtype=input.dtype)]

    def generic_scan(self, input):
        input_data = []
        output_data = []
        shape = input[0].handle.data.shape
        for arg in input:
            input_data.append(arg.handle.data)
            output_data.append(np.zeros(shape, dtype=arg.handle.data.dtype))
        # Scan on axis
        for i in range(input_data[0].size):
            # Recover index from i using shape
            index = np.unravel_index(i, shape)
            data = tuple(self.to_tensor(d[index], input[ii].dtype) for ii, d in enumerate(input_data))
            if index[self.axis] == 0:
                # First element
                for j in range(len(output_data)):
                    output_data[j][index] = data[j].handle.data.item()
            else:
                prev_index = tuple(index[i] - 1 if i == self.axis else index[i] for i in range(len(index)))
                acc_tuple = tuple(self.to_tensor(o[prev_index], input[oi].dtype) for oi, o in enumerate(output_data))
                combine_fn_ret = self.combine_fn.fn(*acc_tuple, *data)
                acc_tuple = (combine_fn_ret, ) if not isinstance(combine_fn_ret, tuple) else combine_fn_ret
                for j in range(len(output_data)):
                    output_data[j][index] = acc_tuple[j].handle.data.item() if isinstance(
                        acc_tuple[j], tl.core.tensor) else acc_tuple[j]
        # Pack output
        ret = []
        for i, data in enumerate(output_data):
            ret.append(self.to_tensor(data, input[i].dtype))
        return ret

    def apply_impl(self, input):
        new_input = []
        if self.reverse:
            for arg in input:
                new_input.append(self.to_tensor(np.flip(arg.handle.data, axis=self.axis), arg.dtype))
        else:
            new_input = input
        if self.combine_fn == tl.standard._sum_combine:
            ret = self.cumsum(new_input[0])
        elif self.combine_fn == tl.standard._prod_combine:
            ret = self.cumprod(new_input[0])
        else:
            # Fall back to the slow mode
            ret = self.generic_scan(new_input)
        if self.reverse:
            for arg in ret:
                arg.handle.data = np.flip(arg.handle.data, axis=self.axis)
        return len(ret) == 1 and ret[0] or tuple(ret)


def _patch_reduce_scan():
    # Because interpreter doesn't support region_builder_fn, we cannot patch the builder
    # to use the new reduce and scan functions.
    # Instead, we need to patch reduce and reduce functions in tl and tl.core
    def _new_reduce(input, axis, combine_fn, keep_dims=False, **kwargs):
        return ReduceOps(axis, combine_fn, keep_dims).apply(input)

    def _new_scan(input, axis, combine_fn, reverse=False, **kwargs):
        return ScanOps(axis, combine_fn, reverse).apply(input)

    tl.reduce = _new_reduce
    tl.associative_scan = _new_scan
    tl.core.reduce = _new_reduce
    tl.core.associative_scan = _new_scan


def _patch_lang_core(lang):

    def _new_to_ir(self, builder):
        # We need to specify signedness for integer types in the numpy mode
        if self.name == 'void':
            return builder.get_void_ty()
        elif self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name == 'int8':
            return builder.get_int8_ty()
        elif self.name == 'uint8':
            return builder.get_uint8_ty()
        elif self.name == 'int16':
            return builder.get_int16_ty()
        elif self.name == 'uint16':
            return builder.get_uint16_ty()
        elif self.name == 'int32':
            return builder.get_int32_ty()
        elif self.name == 'uint32':
            return builder.get_uint32_ty()
        elif self.name == 'int64':
            return builder.get_int64_ty()
        elif self.name == 'uint64':
            return builder.get_uint64_ty()
        elif self.name == 'fp8e5':
            return builder.get_fp8e5_ty()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_ty()
        elif self.name == 'fp8e4b15':
            return builder.get_fp8e4b15_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        elif self.name == 'fp64':
            return builder.get_double_ty()
        raise ValueError(f'fail to convert {self} to ir type')

    # can't just map lang.static_range to `range`, because `tl.static_range`
    # can get `step` passed by keyword
    def _new_range(arg1, arg2=None, step=None, **kwargs):
        if step is None:
            step = 1
        if arg2 is None:
            start, end = 0, arg1
        else:
            start, end = arg1, arg2
        return range(start, end, step)

    def _new_static_assert(cond, msg=""):
        assert cond, msg

    lang.range = _new_range
    lang.static_range = _new_range
    lang.static_assert = _new_static_assert
    lang.dtype.to_ir = _new_to_ir

    _patch_reduce_scan()


def _patch_lang(fn):
    lang = [value for _, value in fn.__globals__.items() if value in [tl, tl.core]]
    assert len(lang) == 1, "triton.language must be visible from within jit'd function"
    _patch_builtin(lang[0], interpreter_builder)
    _patch_builtin(lang[0].tensor, interpreter_builder)
    if lang[0] == tl:
        _patch_builtin(lang[0].math, interpreter_builder)
    _patch_lang_tensor(lang[0].tensor)
    _patch_lang_core(lang[0])


# TODO: wrap everything in triton tensors
def _implicit_cvt(arg):
    if isinstance(arg, int):
        ty = tl.str_to_ty(triton.runtime.jit.JITFunction._type_of(triton.runtime.jit.JITFunction._key_of(arg)))
        dtype = np.int32
        if -2**31 <= arg < 2**31:
            dtype = np.int32
        elif 2**31 <= arg < 2**32:
            dtype = np.uint32
        elif -2**63 <= arg < 2**63:
            dtype = np.int64
        elif 2**63 <= arg < 2**64:
            dtype = np.uint64
        else:
            raise ValueError(f"Unsupported integer value {arg}")
        handle = TensorHandle(np.array([arg], dtype=dtype), ty)
        return tl.tensor(handle, ty)
    if hasattr(arg, "data_ptr"):
        ty = tl.str_to_ty(triton.runtime.jit.JITFunction._type_of(triton.runtime.jit.JITFunction._key_of(arg)))
        handle = TensorHandle(np.array([arg.data_ptr()], dtype=np.uint64), ty)
        return tl.tensor(handle, ty)
    return arg


interpreter_builder = InterpreterBuilder()

# These keywords are not supported by the interpreter
RESERVED_KWS = ["num_warps", "num_stages", "num_ctas", "enable_fp_fusion", "grid"]


class GridExecutor:

    def __init__(self, fn, arg_names, grid):
        from .jit import _normalize_ty  # TODO: modularize

        self.fn = fn
        self.arg_names = arg_names
        self.grid = grid
        __annotations__ = {name: _normalize_ty(ty) for name, ty in fn.__annotations__.items()}
        self.constexprs = [name for name in arg_names if __annotations__.get(name) == "constexpr"]

    def _init_args_hst(self, args_dev):
        args_hst = []
        for arg in args_dev:
            if hasattr(arg, "data_ptr"):
                args_hst.append(arg.cpu())
            else:
                args_hst.append(arg)
        return args_hst

    def _restore_args_dev(self, args_dev, args_hst):
        for arg_dev, arg_hst in zip(args_dev, args_hst):
            if hasattr(arg_dev, "data_ptr"):
                arg_dev.copy_(arg_hst.to(arg_dev.device))

    def __call__(self, *args_dev, **kwargs):
        # copy arguments to the host
        args_hst = self._init_args_hst(args_dev)
        # removes reserved keywords from kwargs
        kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_KWS}
        if kwargs.pop("warmup", False):
            return
        # remaps core language functions to interpreted ones
        _patch_lang(self.fn)
        # we need to copy arguments to the host for the interpreter
        # implicitly convert tensor arguments to their base pointers
        args = inspect.getcallargs(self.fn, *args_hst, **kwargs)
        args = {name: arg if name in self.constexprs else _implicit_cvt(arg) for name, arg in args.items()}
        # iterate through grid
        grid = self.grid(args) if callable(self.grid) else self.grid
        assert len(grid) <= 3, "grid must have at most 3 dimensions"
        grid = grid + (1, ) * (3 - len(grid))
        interpreter_builder.set_grid_dim(*grid)
        try:
            for x in range(grid[0]):
                for y in range(grid[1]):
                    for z in range(grid[2]):
                        interpreter_builder.set_grid_idx(x, y, z)
                        self.fn(**args)
        except Exception as e:
            raise InterpreterError(repr(e)) from e
        # copy arguments back to propagate side-effects
        self._restore_args_dev(args_dev, args_hst)


class InterpretedFunction:

    def __init__(self, fn) -> None:
        self.fn = fn

        def run(*args, **kwargs):
            grid = kwargs["grid"]
            return GridExecutor(self.fn, self.arg_names, grid)(*args, **kwargs)

        self.run = run
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]

    @property
    def __name__(self):
        return self.fn.__name__

    def __getitem__(self, grid):
        return GridExecutor(self.fn, self.arg_names, grid)

    def __call__(self, *args, **kwargs):
        # This is a device function call
        _patch_lang(self.fn)
        try:
            return self.fn(*args, **kwargs)
        except Exception as e:
            raise InterpreterError(repr(e)) from e
