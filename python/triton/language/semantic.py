from __future__ import annotations  # remove after python 3.11

from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar

from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl

T = TypeVar('T')

# Create custom exception that prints message "hello"


class IncompatibleTypeErrorImpl(Exception):
    def __init__(self, type_a, type_b):
        self.type_a = type_a
        self.type_b = type_b
        self.message = "invalid operands of type " + self.type_a.__repr__() + " and " + self.type_b.__repr__()
        super(IncompatibleTypeErrorImpl, self).__init__(self.message)


# ===----------------------------------------------------------------------===##
# Programming Model
# ===----------------------------------------------------------------------===##

def program_id(axis: int, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_get_program_id(axis), tl.int32)


def num_programs(axis: int, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_get_num_programs(axis), tl.int32)

# ===----------------------------------------------------------------------===//
#                               Implicit Casting Utilities
# ===----------------------------------------------------------------------===//


def integer_promote_impl(a_ty: tl.dtype, b_ty: tl.dtype) -> tl.dtype:
    a_rank = a_ty.int_bitwidth
    b_rank = b_ty.int_bitwidth
    a_sn = a_ty.int_signedness
    b_sn = b_ty.int_signedness
    # Rules for signedness taken from "Usual arithmetic conversions" on
    # https://en.cppreference.com/w/c/language/conversion.
    if a_sn == b_sn:
        return a_ty if a_rank > b_rank else b_ty
    elif a_sn == tl.dtype.SIGNEDNESS.UNSIGNED:
        return a_ty if a_rank >= b_rank else b_ty
    elif b_sn == tl.dtype.SIGNEDNESS.UNSIGNED:
        return b_ty if b_rank >= a_rank else a_ty
    assert False


def computation_type_impl(a_ty: tl.dtype, b_ty: tl.dtype, div_or_mod: bool) -> tl.dtype:
    # 1) if one operand is double, the other is implicitly
    #    converted to double
    if a_ty.is_fp64() or b_ty.is_fp64():
        return tl.float64
    # 2) if one operand is float, the other is implicitly
    #    converted to float
    if a_ty.is_fp32() or b_ty.is_fp32():
        return tl.float32
    # 3 ) if one operand is half, the other is implicitly converted to half
    #     unless we're doing / or %, which do not exist natively in PTX for fp16.
    #     Supported PTX op: add, sub, mul, fma, neg, abs, min, max, tanh, ex2, setp
    if a_ty.is_fp16() or b_ty.is_fp16():
        if div_or_mod:
            return tl.float32
        else:
            return tl.float16
    # 4) return bf16 only if both operands are of bf16
    if a_ty.is_bf16() or b_ty.is_bf16():
        if div_or_mod:
            return tl.float32
        if a_ty.is_bf16() and b_ty.is_bf16():
            return tl.bfloat16
        return tl.float32
    if not a_ty.is_int() or not b_ty.is_int():
        assert False
    # 5 ) both operands are integer and undergo
    #    integer promotion
    if div_or_mod and a_ty.int_signedness != b_ty.int_signedness:
        raise ValueError("Cannot use /, #, or % with " + a_ty.__repr__() + " and " + b_ty.__repr__() + " because they have different signedness;"
                         "this is unlikely to result in a useful answer. Cast them to the same signedness.")
    return integer_promote_impl(a_ty, b_ty)

# ===----------------------------------------------------------------------===//
#                               Binary Operators
# ===----------------------------------------------------------------------===//


def check_ptr_type_impl(type_a: tl.dtype, type_b: tl.dtype, allow_ptr_a: bool) -> None:
    if type_a.is_ptr():
        if not allow_ptr_a:
            raise IncompatibleTypeErrorImpl(type_a, type_b)
        # T* + U* with T != U
        if type_b.is_ptr() and (type_a != type_b):
            raise IncompatibleTypeErrorImpl(type_a, type_b)
        # T* + float
        if type_b.is_floating():
            raise IncompatibleTypeErrorImpl(type_a, type_b)


def binary_op_type_checking_impl(lhs: tl.tensor,
                                 rhs: tl.tensor,
                                 builder: ir.builder,
                                 allow_lhs_ptr=False, allow_rhs_ptr=False,
                                 arithmetic_check=True, div_or_mod=False
                                 ) -> Tuple[tl.tensor, tl.tensor]:
    # implicit broadcasting
    lhs, rhs = broadcast_impl_value(lhs, rhs, builder)
    # implicit typecasting
    lhs_sca_ty = lhs.type.scalar
    rhs_sca_ty = rhs.type.scalar
    check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr)
    check_ptr_type_impl(rhs_sca_ty, lhs_sca_ty, allow_rhs_ptr)
    if arithmetic_check and not lhs_sca_ty.is_ptr() and not rhs_sca_ty.is_ptr():
        ret_sca_ty = computation_type_impl(lhs_sca_ty, rhs_sca_ty, div_or_mod)
        lhs = cast(lhs, ret_sca_ty, builder)
        rhs = cast(rhs, ret_sca_ty, builder)
    return lhs, rhs


def add(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar

    # offset + ptr
    # ptr + offset
    if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
        input, other = other, input
        input_scalar_ty = input.type.scalar
        other_scalar_ty = other.type.scalar
    if input_scalar_ty.is_ptr():
        return tl.tensor(builder.create_addptr(input.handle, other.handle), input.type)
    # float + float
    elif input_scalar_ty.is_floating():
        return tl.tensor(builder.create_fadd(input.handle, other.handle), input.type)
    # int + int
    elif input_scalar_ty.is_int():
        return tl.tensor(builder.create_add(input.handle, other.handle), input.type)
    assert False


def sub(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, True, False)
    scalar_ty = input.type.scalar
    # ptr - offset
    if scalar_ty.is_ptr():
        return tl.tensor(builder.create_addptr(input.handle, minus(other, builder).handle),
                         input.type)
    # float - float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fsub(input.handle, other.handle), input.type)
    # int - int
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_sub(input.handle, other.handle), input.type)
    assert False


def mul(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float * float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fmul(input.handle, other.handle), input.type)
    # * int
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_mul(input.handle, other.handle), input.type)
    assert False


def truediv(input: tl.tensor,
            other: tl.tensor,
            builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float / int
    if input_scalar_ty.is_floating() and other_scalar_ty.is_int():
        other = cast(other, input_scalar_ty, builder)
    # int / float
    elif input_scalar_ty.is_int() and other_scalar_ty.is_floating():
        input = cast(input, other_scalar_ty, builder)
    # int / int (cast to tl.float32)
    elif input_scalar_ty.is_int() and other_scalar_ty.is_int():
        input = cast(input, tl.float32, builder)
        other = cast(other, tl.float32, builder)
    # float / float (cast to the highest exponent type)
    elif input_scalar_ty.is_floating() and other_scalar_ty.is_floating():
        if input_scalar_ty.fp_mantissa_width > other_scalar_ty.fp_mantissa_width:
            other = cast(other, input_scalar_ty, builder)
        else:
            input = cast(input, other_scalar_ty, builder)
    # unreachable
    else:
        assert False
    return tl.tensor(builder.create_fdiv(input.handle, other.handle), input.type)


def floordiv(input: tl.tensor,
             other: tl.tensor,
             builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if input_scalar_ty.is_int() and other_scalar_ty.is_int():
        ret_ty = integer_promote_impl(input_scalar_ty, other_scalar_ty)
        input = cast(input, ret_ty, builder)
        other = cast(other, ret_ty, builder)
        if ret_ty.is_int_signed():
            return tl.tensor(builder.create_sdiv(input.handle, other.handle), input.type)
        else:
            return tl.tensor(builder.create_udiv(input.handle, other.handle), input.type)
    assert False


def fdiv(input: tl.tensor,
         other: tl.tensor,
         ieee_rounding: bool,
         builder: ir.builder) -> tl.tensor:
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    if not input_scalar_ty.is_floating() or not other_scalar_ty.is_floating():
        raise ValueError("both operands of fdiv must have floating scalar type")
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, False, True)
    ret = builder.create_fdiv(input.handle, other.handle)
    return tl.tensor(ret, input.type)


def mod(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, True, True)
    scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar
    # float % float
    if scalar_ty.is_floating():
        # input - input.div(other, rounding_mode="floor") * other
        ret = sub(input, mul(floor(fdiv(input, other, False, builder), builder),
                             other, builder),
                  builder)
        return ret
    # % int
    elif scalar_ty.is_int():
        if scalar_ty.int_signedness != other_scalar_ty.int_signedness:
            raise ValueError("Cannot mod " + scalar_ty.__repr__() + " by " + other_scalar_ty.__repr__() + " "
                             "because they have different signedness;"
                             "this is unlikely to result in a useful answer. Cast them to the same signedness.")
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_srem(input.handle, other.handle), input.type)
        else:
            return tl.tensor(builder.create_urem(input.handle, other.handle), input.type)
    assert False

##############
# bitwise ops
##############


def bitwise_op_type_checking_impl(input: tl.tensor,
                                  other: tl.tensor,
                                  builder: ir.builder) -> Tuple[tl.tensor, tl.tensor]:
    input, other = binary_op_type_checking_impl(input, other, builder, False, False, False)
    input_sca_ty = input.type.scalar
    other_sca_ty = other.type.scalar
    if not input_sca_ty.is_int() or not other_sca_ty.is_int():
        raise IncompatibleTypeErrorImpl(input_sca_ty, other_sca_ty)
    ret_sca_ty = integer_promote_impl(input_sca_ty, other_sca_ty)
    if ret_sca_ty != input_sca_ty:
        input = cast(input, ret_sca_ty, builder)
    if ret_sca_ty != other_sca_ty:
        other = cast(other, ret_sca_ty, builder)
    return input, other


def and_(input: tl.tensor,
         other: tl.tensor,
         builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_and(input.handle, other.handle), input.type)


def or_(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_or(input.handle, other.handle), input.type)


def xor_(input: tl.tensor,
         other: tl.tensor,
         builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_xor(input.handle, other.handle), input.type)


def logical_and(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    if not input.type.is_int1():
        input = bitcast(input, tl.dtype("int1"), builder)
    if not other.type.is_int1():
        other = bitcast(other, tl.dtype("int1"), builder)
    return and_(input, other, builder)


def logical_or(input: tl.tensor, other: tl.tensor, builder: ir.builder) -> tl.tensor:
    if not input.type.is_int1():
        input = bitcast(input, tl.dtype("int1"), builder)
    if not other.type.is_int1():
        other = bitcast(other, tl.dtype("int1"), builder)
    return or_(input, other, builder)


def not_(input: tl.tensor, builder: ir.builder):
    if not input.type.is_int1():
        input = bitcast(input, tl.dtype("int1"), builder)
    return invert(input, builder)


def lshr(input: tl.tensor,
         other: tl.tensor,
         builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_lshr(input.handle, other.handle), input.type)


def ashr(input: tl.tensor,
         other: tl.tensor,
         builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_ashr(input.handle, other.handle), input.type)


def shl(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = bitwise_op_type_checking_impl(input, other, builder)
    return tl.tensor(builder.create_shl(input.handle, other.handle), input.type)

# ===----------------------------------------------------------------------===//
#                               Unary Operators
# ===----------------------------------------------------------------------===//


def plus(input: tl.tensor) -> tl.tensor:
    return input


def minus(input: tl.tensor,
          builder: ir.builder) -> tl.tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr():
        raise ValueError("wrong type argument to unary minus (" + input_sca_ty.__repr__() + ")")
    _0 = tl.tensor(builder.get_null_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return sub(_0, input, builder)


def invert(input: tl.tensor,
           builder: tl.tensor) -> tl.tensor:
    input_sca_ty = input.type.scalar
    if input_sca_ty.is_ptr() or input_sca_ty.is_floating():
        raise ValueError("wrong type argument to unary invert (" + input_sca_ty.__repr__() + ")")
    _1 = tl.tensor(builder.get_all_ones_value(input_sca_ty.to_ir(builder)), input_sca_ty)
    return xor_(input, _1, builder)


# ===----------------------------------------------------------------------===//
#                               Comparison Operators
# ===----------------------------------------------------------------------===//
def _bool_like(v: tl.tensor) -> tl.block_type:
    if not v.type.is_block():
        return tl.int1
    shape = v.type.shape
    return tl.block_type(tl.int1, shape)


def greater_than(input: tl.tensor,
                 other: tl.tensor,
                 builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float > float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOGT(input.handle, other.handle), _bool_like(input))
    # > int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_icmpSGT(input.handle, other.handle), _bool_like(input))
        else:
            return tl.tensor(builder.create_icmpUGT(input.handle, other.handle), _bool_like(input))
    assert False


def greater_equal(input: tl.tensor,
                  other: tl.tensor,
                  builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float >= float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOGE(input.handle, other.handle), _bool_like(input))
    # >= int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_icmpSGE(input.handle, other.handle), _bool_like(input))
        else:
            return tl.tensor(builder.create_icmpUGE(input.handle, other.handle), _bool_like(input))
    assert False


def less_than(input: tl.tensor,
              other: tl.tensor,
              builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float < float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOLT(input.handle, other.handle), _bool_like(input))
    # < int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_icmpSLT(input.handle, other.handle), _bool_like(input))
        else:
            return tl.tensor(builder.create_icmpULT(input.handle, other.handle), _bool_like(input))
    assert False


def less_equal(input: tl.tensor,
               other: tl.tensor,
               builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float < float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOLE(input.handle, other.handle), _bool_like(input))
    # < int
    elif scalar_ty.is_int():
        if scalar_ty.is_int_signed():
            return tl.tensor(builder.create_icmpSLE(input.handle, other.handle), _bool_like(input))
        else:
            return tl.tensor(builder.create_icmpULE(input.handle, other.handle), _bool_like(input))
    assert False


def equal(input: tl.tensor,
          other: tl.tensor,
          builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float == float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpOEQ(input.handle, other.handle), _bool_like(input))
    # == int
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_icmpEQ(input.handle, other.handle), _bool_like(input))
    assert False


def not_equal(input: tl.tensor,
              other: tl.tensor,
              builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder)
    scalar_ty = input.type.scalar
    # float == float
    if scalar_ty.is_floating():
        return tl.tensor(builder.create_fcmpUNE(input.handle, other.handle), _bool_like(input))
    # == int
    elif scalar_ty.is_int():
        return tl.tensor(builder.create_icmpNE(input.handle, other.handle), _bool_like(input))
    assert False

# ===----------------------------------------------------------------------===//
#                               Block Creation
# ===----------------------------------------------------------------------===//


def arange(start: int, end: int, builder: ir.builder) -> tl.tensor:
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("arange's arguments must be of type tl.constexpr")
    is_start_int64 = bool(start >> 32)
    is_end_int64 = bool(end >> 32)
    if is_start_int64 or is_end_int64:
        raise ValueError("arange must fit in int32")
    if end <= start:
        raise ValueError("arange's end argument must be greater than the start argument")

    shape = [end - start]
    ret_ty = tl.block_type(tl.int32, shape)
    return tl.tensor(builder.create_make_range(start, end), ret_ty)


def full(shape: List[int], value, dtype: tl.dtype, builder: ir.builder) -> tl.tensor:
    if isinstance(value, tl.tensor):
        assert value.numel.value == 1, "only accepts size-1 tensor"
        value = cast(value, dtype, builder)
    else:
        # scalar
        if dtype is None:
            raise ValueError("dtype must be specified when value is not a tensor")
        if value == 0:
            value = builder.get_null_value(dtype.to_ir(builder))
        else:
            get_value_fn = getattr(builder, f"get_{dtype.name}")
            value = get_value_fn(value)
        value = tl.tensor(value, dtype)

    return splat(value, shape, builder)


# ===----------------------------------------------------------------------===//
#                               Shape Manipulation
# ===----------------------------------------------------------------------===//

def splat(value: tl.tensor, shape: List[int], builder: ir.builder) -> tl.tensor:
    assert not value.type.is_block(), "Cannot splat a block tensor"
    if len(shape) == 0:
        return value
    ret_ty = tl.block_type(value.dtype, shape)
    return tl.tensor(builder.create_splat(value.handle, shape), ret_ty)


def view(input: tl.tensor,
         dst_shape: List[int],
         builder: ir.builder) -> tl.tensor:
    # TODO: disable when TritonToTritonGPU handles views properly

    # assert len(input.shape) == len(dst_shape)
    numel = 1
    for s in dst_shape:
        numel *= s
    if input.type.numel != numel:
        raise ValueError("cannot view block of different shape")
    ret_ty = tl.block_type(input.type.scalar, dst_shape)
    return tl.tensor(builder.create_view(input.handle, dst_shape), ret_ty)


def reshape(input: tl.tensor,
            dst_shape: List[int],
            builder: ir.builder) -> tl.tensor:
    raise ValueError("`reshape` is not supported yet. Please use `view` instead if applicable. "
                     "Note that view may reorder elements in an implementation- and context- dependent way.")


def expand_dims(input: tl.tensor, axis: int, builder: ir.builder) -> tl.tensor:
    dst_shape = [tl._constexpr_to_value(x) for x in input.shape]
    dst_shape.insert(axis, 1)

    if not input.type.is_block():
        return splat(input, shape=dst_shape, builder=builder)

    ret_ty = tl.block_type(input.type.scalar, dst_shape)
    return tl.tensor(builder.create_expand_dims(input.handle, axis), ret_ty)


def cat(lhs: tl.tensor, rhs: tl.tensor, can_reorder: bool, builder: ir.builder) -> tl.tensor:
    assert can_reorder, "current implementation of `cat` always may reorder elements"
    assert len(lhs.shape) == 1
    ret_type = tl.block_type(lhs.type.scalar, [lhs.shape[0] + rhs.shape[0]])
    return tl.tensor(builder.create_cat(lhs.handle, rhs.handle), ret_type)


def trans(input: tl.tensor, builder: ir.builder) -> tl.tensor:
    if len(input.shape) != 2:
        raise ValueError("Only 2D tensors can be transposed")
    ret_type = tl.block_type(input.type.scalar, [input.shape[1], input.shape[0]])
    return tl.tensor(builder.create_trans(input.handle), ret_type)


def broadcast_impl_shape(input: tl.tensor,
                         shape: List[int],
                         builder: ir.builder) -> tl.tensor:
    if not input.type.is_block():
        ret_ty = tl.block_type(input.type, shape)
        return tl.tensor(builder.create_splat(input.handle, shape), ret_ty)
    src_shape = input.type.get_block_shapes()
    if len(src_shape) != len(shape):
        raise ValueError(f"Cannot broadcast, rank mismatch: {src_shape}, {shape}")
    if shape == src_shape:
        return input
    for i, item in enumerate(src_shape):
        if shape[i] != item and item != 1:
            raise ValueError(f"Cannot broadcast, the expanded size of the tensor ({shape[i]})"
                             f" must match the existing size ({item}) at non-singleton dimension"
                             f" {i}: {src_shape}, {shape}")
    ret_ty = tl.block_type(input.type.scalar, shape)
    return tl.tensor(builder.create_broadcast(input.handle, shape), ret_ty)


def broadcast_impl_value(lhs: tl.tensor,
                         rhs: tl.tensor,
                         builder: ir.builder) -> tl.tensor:
    lhs_ty = lhs.type
    rhs_ty = rhs.type

    # make_shape_compatible(block, scalar)
    if lhs_ty.is_block() and not rhs_ty.is_block():
        rhs_ty = tl.block_type(rhs_ty.scalar, lhs_ty.shape)
        rhs = tl.tensor(builder.create_splat(rhs.handle, lhs_ty.get_block_shapes()), rhs_ty)
    # make_shape_compatible(scalar, block)
    elif not lhs_ty.is_block() and rhs_ty.is_block():
        lhs_ty = tl.block_type(lhs_ty.scalar, rhs_ty.shape)
        lhs = tl.tensor(builder.create_splat(lhs.handle, rhs_ty.get_block_shapes()), lhs_ty)
    # make_shape_compatible(block, block)
    elif lhs_ty.is_block() and rhs_ty.is_block():
        lhs_shape = lhs_ty.get_block_shapes()
        rhs_shape = rhs_ty.get_block_shapes()

        if len(lhs_shape) < len(rhs_shape):
            # Add new axes to lhs
            for dim in range(len(lhs_shape), len(rhs_shape)):
                lhs = tl.tensor(builder.create_expand_dims(lhs.handle, 0), tl.block_type(lhs_ty.scalar, [1] + lhs_shape))
                lhs_ty = lhs.type
                lhs_shape = lhs_ty.get_block_shapes()
        elif len(rhs_shape) < len(lhs_shape):
            # Add new axes to rhs
            for dim in range(len(rhs_shape), len(lhs_shape)):
                rhs = tl.tensor(builder.create_expand_dims(rhs.handle, 0), tl.block_type(rhs_ty.scalar, [1] + rhs_shape))
                rhs_ty = rhs.type
                rhs_shape = rhs_ty.get_block_shapes()
        assert len(rhs_shape) == len(lhs_shape)

        ret_shape = []
        for i, left in enumerate(lhs_shape):
            right = rhs_shape[i]
            if left == 1:
                ret_shape.append(right)
            elif right == 1:
                ret_shape.append(left)
            elif left == right:
                ret_shape.append(left)
            else:
                raise ValueError("Cannot make_shape_compatible: incompatible dimensions "
                                 "at index " + str(i) + ": " + str(left) + " and " + str(right))
        if lhs_shape != ret_shape:
            ret_ty = tl.block_type(lhs_ty.scalar, ret_shape)
            lhs = tl.tensor(builder.create_broadcast(lhs.handle, ret_shape), ret_ty)
        if rhs_shape != ret_shape:
            ret_ty = tl.block_type(rhs_ty.scalar, ret_shape)
            rhs = tl.tensor(builder.create_broadcast(rhs.handle, ret_shape), ret_ty)
    # (scalar, scalar) => returns original blocks
    return lhs, rhs

#######
# cast
#######


def bitcast(input: tl.tensor,
            dst_ty: tl.dtype,
            builder: ir.builder) -> tl.tensor:
    src_ty = input.type
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty.scalar, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input
    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar
    if src_sca_ty.is_ptr() or dst_sca_ty.is_ptr():
        return cast(input, dst_ty, builder)
    # Bitcast
    src_bits = src_sca_ty.primitive_bitwidth
    dst_bits = dst_sca_ty.primitive_bitwidth
    if src_bits != dst_bits:
        raise ValueError("Cannot bitcast data-type of size " + str(src_bits) + " to "
                         "data-type of size " + str(dst_bits))
    return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)),
                     dst_ty)


# TODO: architecture descriptor class
def _is_cuda(arch):
    return isinstance(arch, int)


def cast(input: tl.tensor,
         dst_ty: tl.dtype,
         builder: ir.builder) -> tl.tensor:
    src_ty = input.type
    if isinstance(dst_ty, tl.constexpr):
        dst_ty = dst_ty.value
    if src_ty.is_block():
        dst_ty = tl.block_type(dst_ty.scalar, input.type.get_block_shapes())
    if src_ty == dst_ty:
        return input

    src_sca_ty = src_ty.scalar
    dst_sca_ty = dst_ty.scalar

    if _is_cuda(builder.arch) and builder.arch < 89 and \
       (src_sca_ty.is_fp8e4nv() or dst_sca_ty.is_fp8e4nv()):
        assert False, "fp8e4nv data type is not supported on CUDA arch < 89"

    # Casting with customized floating types involved: fp8 <=> bf16, fp16, fp32, fp64
    if (src_sca_ty.is_fp8() and dst_sca_ty.is_floating()) or \
       (src_sca_ty.is_floating() and dst_sca_ty.is_fp8()):
        return tl.tensor(builder.create_fp_to_fp(input.handle, dst_ty.to_ir(builder)),
                         dst_ty)

    # bf16 <=> (not fp32)
    if (src_sca_ty.is_fp16() and not dst_sca_ty.is_fp32()) or \
       (src_sca_ty.is_bf16() and not dst_sca_ty.is_fp32()):
        return cast(cast(input, tl.float32, builder), dst_sca_ty, builder)

    # Standard floating types' casting: truncation
    #   fp64 => fp32, fp16, bf16
    #   fp32 => fp16, bf16
    truncate_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth > dst_sca_ty.primitive_bitwidth
    if truncate_fp:
        return tl.tensor(builder.create_fp_trunc(input.handle,
                                                 dst_ty.to_ir(builder)),
                         dst_ty)

    # Standard floating types' casting: extension
    #   fp32 => fp64
    #   fp16 => fp32, fp64
    #   bf16 => fp32, fp64
    ext_fp = src_sca_ty.is_floating() and \
        dst_sca_ty.is_floating() and \
        src_sca_ty.primitive_bitwidth < dst_sca_ty.primitive_bitwidth
    if ext_fp:
        return tl.tensor(builder.create_fp_ext(input.handle,
                                               dst_ty.to_ir(builder)),
                         dst_ty)

    # Casting between integer types
    if src_sca_ty.is_int() and dst_sca_ty.is_int() and \
       (src_sca_ty.int_bitwidth != dst_sca_ty.int_bitwidth or src_sca_ty.int_signedness != dst_sca_ty.int_signedness):
        sign_extend = src_sca_ty.is_int_signed() and not src_sca_ty.is_bool()
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        else:
            return tl.tensor(builder.create_int_cast(input.handle,
                                                     dst_ty.to_ir(builder), sign_extend),
                             dst_ty)

    # Casting standard floating types to integer types
    if src_sca_ty.is_standard_floating() and dst_sca_ty.is_int():
        if dst_sca_ty.is_bool():
            ty = input.dtype.to_ir(builder)
            _0 = tl.tensor(builder.get_null_value(ty), input.dtype)
            return not_equal(input, _0, builder)
        elif dst_sca_ty.is_int_signed():
            return tl.tensor(builder.create_fp_to_si(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)
        else:
            return tl.tensor(builder.create_fp_to_ui(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)

    # Casting integer types to standard floating types
    if src_sca_ty.is_int() and dst_sca_ty.is_standard_floating():
        if src_sca_ty.is_bool() or not src_sca_ty.is_int_signed():
            return tl.tensor(builder.create_ui_to_fp(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)
        else:
            return tl.tensor(builder.create_si_to_fp(input.handle,
                                                     dst_ty.to_ir(builder)),
                             dst_ty)

    # Casting pointer types to integer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_int():
        bitwidth = dst_sca_ty.int_bitwidth
        if bitwidth == 64:
            return tl.tensor(builder.create_ptr_to_int(input.handle, dst_ty.to_ir(builder)),
                             dst_ty)
        if bitwidth == 1:
            return not_equal(cast(input, tl.int64, builder),
                             tl.tensor(builder.get_int64(0), tl.int64),
                             builder)

    # Casting integer types to pointer types
    if src_sca_ty.is_int() and dst_sca_ty.is_ptr():
        return tl.tensor(builder.create_int_to_ptr(input.handle, dst_ty.to_ir(builder)), dst_ty)

    # Casting pointer types to pointer types
    if src_sca_ty.is_ptr() and dst_sca_ty.is_ptr():
        return tl.tensor(builder.create_bitcast(input.handle, dst_ty.to_ir(builder)), dst_ty)

    assert False, f'cannot cast {input} to {dst_ty}'

# ===----------------------------------------------------------------------===//
#                               Memory Operators
# ===----------------------------------------------------------------------===//


def _str_to_load_cache_modifier(cache_modifier):
    cache = ir.CACHE_MODIFIER.NONE  # default
    if cache_modifier:
        if cache_modifier == ".ca":
            cache = ir.CACHE_MODIFIER.CA
        elif cache_modifier == ".cg":
            cache = ir.CACHE_MODIFIER.CG
        else:
            raise ValueError(f"Cache modifier {cache_modifier} not supported")
    return cache


def _str_to_store_cache_modifier(cache_modifier):
    cache = ir.CACHE_MODIFIER.NONE  # default
    if cache_modifier:
        if cache_modifier == ".wb":
            cache = ir.CACHE_MODIFIER.WB
        elif cache_modifier == ".cg":
            cache = ir.CACHE_MODIFIER.CG
        elif cache_modifier == ".cs":
            cache = ir.CACHE_MODIFIER.CS
        elif cache_modifier == ".wt":
            cache = ir.CACHE_MODIFIER.WT
        else:
            raise ValueError(f"Cache modifier {cache_modifier} not supported")
    return cache


def _str_to_eviction_policy(eviction_policy):
    eviction = ir.EVICTION_POLICY.NORMAL  # default
    if eviction_policy:
        if eviction_policy == "evict_last":
            eviction = ir.EVICTION_POLICY.EVICT_LAST
        elif eviction_policy == "evict_first":
            eviction = ir.EVICTION_POLICY.EVICT_FIRST
        else:
            raise ValueError(f"Eviction policy {eviction_policy} not supported")
    return eviction


def _str_to_padding_option(padding_option):
    padding = None  # default
    if padding_option:
        if padding_option == "zero":
            padding = ir.PADDING_OPTION.PAD_ZERO
        elif padding_option == "nan":
            padding = ir.PADDING_OPTION.PAD_NAN
        else:
            raise ValueError(f"Padding option {padding_option} not supported")
    return padding


def _str_to_sem(sem_option):
    sem = ir.MEM_SEMANTIC.ACQUIRE_RELEASE
    if sem_option:
        if sem_option == "acquire":
            sem = ir.MEM_SEMANTIC.ACQUIRE
        elif sem_option == "release":
            sem = ir.MEM_SEMANTIC.RELEASE
        elif sem_option == "acq_rel":
            sem = ir.MEM_SEMANTIC.ACQUIRE_RELEASE
        elif sem_option == "relaxed":
            sem = ir.MEM_SEMANTIC.RELAXED
        else:
            raise ValueError(f"Memory semantic {sem_option} not supported")
    return sem


def _canonicalize_boundary_check(boundary_check, block_shape):
    if boundary_check:
        if not hasattr(boundary_check, "__iter__"):
            boundary_check = [boundary_check]
        boundary_check = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in boundary_check]
        for dim in boundary_check:
            assert isinstance(dim, int) and 0 <= dim < len(block_shape)
        assert len(boundary_check) > 0
        assert len(boundary_check) == len(set(boundary_check)), "Duplicate dimension in `boundary_check`"
        return sorted(boundary_check)
    return tuple()


def _load_block_pointer(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder):
    # Load by a block pointer: `pointer_type<block_type<>>`
    # Block pointer can not have `mask` and `other` arguments
    if mask or other:
        raise ValueError("`mask` and `other` arguments cannot be specified for loading block pointers")

    elt_ty = ptr.type.element_ty.element_ty
    assert elt_ty != tl.int1, "`tl.int1` should be rewrited in `tl.make_block_ptr`"
    if elt_ty.is_int() and padding == ir.PADDING_OPTION.PAD_NAN:
        raise ValueError("Padding option `nan` is not supported for integer block pointers")

    # `dst_ty` is de-referenced type of the pointer type
    dst_ty = ptr.type.element_ty

    # Check `boundary_check` argument
    boundary_check = _canonicalize_boundary_check(boundary_check, dst_ty.get_block_shapes())

    # Build IR
    return tl.tensor(builder.create_tensor_pointer_load(ptr.handle, boundary_check, padding, cache, eviction,
                                                        is_volatile), dst_ty)


def _load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder):
    # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
    if not ptr.type.scalar.is_ptr():
        raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")

    # Check `mask`, `other`, `boundary_check`, and `padding` arguments
    if not mask and other:
        raise ValueError("`other` cannot be provided without `mask`")
    if padding or boundary_check:
        raise ValueError("`padding_option` or `boundary_check` argument is not supported for loading a tensor of"
                         "pointers or loading a scalar. Because the compiler does not know the boundary; please "
                         "use block pointers (defined by `make_block_ptr`) instead")

    # For a pointer of scalar, check the type of `mask` and `other`
    if not ptr.type.is_block():
        if mask and mask.type.is_block():
            raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
        if other and other.type.is_block():
            raise ValueError("Other argument cannot be block type if pointer argument is not a block")

    # Make `mask` and `other` into the same shape as `ptr`
    if ptr.type.is_block():
        if mask:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if other:
            other = broadcast_impl_shape(other, ptr.type.get_block_shapes(), builder)

    # Get `pointer_type<elt_ty>` and `elt_ty`
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty

    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    if elt_ty == tl.int1:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)

    # Cast `other` into `ele_ty` type
    if other:
        other = cast(other, elt_ty, builder)

    # Create loaded result type `dst_ty`
    if ptr.type.is_block():
        shape = ptr.type.get_block_shapes()
        dst_ty = tl.block_type(elt_ty, shape)
    else:
        # Load by de-referencing the pointer of scalar
        dst_ty = elt_ty

    # Build IR
    if not mask:
        return tl.tensor(builder.create_load(ptr.handle, cache, eviction, is_volatile), dst_ty)
    else:
        return tl.tensor(builder.create_masked_load(ptr.handle, mask.handle, other.handle if other else None, cache,
                                                    eviction, is_volatile), dst_ty)


def load(ptr: tl.tensor,
         mask: Optional[tl.tensor],
         other: Optional[tl.tensor],
         boundary_check,
         padding_option: str,
         cache_modifier: str,
         eviction_policy: str,
         is_volatile: bool,
         builder: ir.builder) -> tl.tensor:
    # Cache, eviction and padding options
    cache = _str_to_load_cache_modifier(cache_modifier)
    eviction = _str_to_eviction_policy(eviction_policy)
    padding = _str_to_padding_option(padding_option)

    if ptr.type.is_ptr() and ptr.type.element_ty.is_block():
        # Load by a block pointer: `pointer_type<block_type<>>`
        return _load_block_pointer(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder)
    else:
        # Load by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        return _load_legacy(ptr, mask, other, boundary_check, padding, cache, eviction, is_volatile, builder)


def _store_block_pointer(ptr, val, mask, boundary_check, cache, eviction, builder):
    # Store by a block pointer: `pointer_type<block_type<>>`
    # Block pointers can not have the `mask` argument
    if mask:
        raise ValueError("`mask` and `other` arguments cannot be specified for loading block pointers")

    # Check same shape and element type
    block_shape = ptr.type.element_ty.get_block_shapes()
    if not val.type.is_block():
        val = broadcast_impl_shape(val, block_shape, builder)
    assert val.type.is_block(), "Value argument must be block type or a scalar"
    assert block_shape == val.type.get_block_shapes(), f"Block shape({block_shape}) and value shape({val.type.get_block_shapes()}) mismatch"
    assert ptr.type.element_ty.element_ty == val.type.element_ty, f"Block element type({ptr.type.element_ty.element_ty}) and value element type({val.type.element_ty}) mismatch"

    elt_ty = ptr.type.element_ty.element_ty
    assert elt_ty != tl.int1, "`tl.int1` should be rewrited in `tl.make_block_ptr`"

    # Check `boundary_check` argument
    boundary_check = _canonicalize_boundary_check(boundary_check, block_shape)

    # Build IR
    return tl.tensor(builder.create_tensor_pointer_store(ptr.handle, val.handle, boundary_check, cache, eviction),
                     tl.void)


def _store_legacy(ptr, val, mask, boundary_check, cache, eviction, builder):
    # Store by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
    if not ptr.type.scalar.is_ptr():
        raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.store`")

    # Check `boundary_check` argument
    if boundary_check:
        raise ValueError("`boundary_check` argument is not supported for storing a tensor of pointers or storing a "
                         "scalar. Because the compiler does not know the boundary; please use block pointers "
                         "(defined by `make_block_ptr`) instead")

    # For a pointer of scalar, check the type of `val` and `mask`
    if not ptr.type.is_block():
        if val.type.is_block():
            raise ValueError("Value argument cannot be block type if pointer argument is not a block")
        if mask and mask.type.is_block():
            raise ValueError("Mask argument cannot be block type if pointer argument is not a block")

    # Make `mask` and `val` into the same shape as `ptr`
    if ptr.type.is_block():
        val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
        if mask:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)

    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty

    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    if elt_ty == tl.int1:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = cast(ptr, ptr_ty, builder)

    # Cast to target data type
    val = cast(val, elt_ty, builder)

    # Build IR
    if not mask:
        return tl.tensor(builder.create_store(ptr.handle, val.handle, cache, eviction), tl.void)
    if not mask.type.scalar.is_bool():
        raise ValueError("Mask must have boolean scalar type")
    return tl.tensor(builder.create_masked_store(ptr.handle, val.handle, mask.handle, cache, eviction), tl.void)


def store(ptr: tl.tensor,
          val: tl.tensor,
          mask: Optional[tl.tensor],
          boundary_check,
          cache_modifier: str,
          eviction_policy: str,
          builder: ir.builder) -> tl.tensor:
    # Cache and eviction options
    cache = _str_to_store_cache_modifier(cache_modifier)
    eviction = _str_to_eviction_policy(eviction_policy)

    if ptr.type.is_ptr() and ptr.type.element_ty.is_block():
        # Store by a block pointer: `pointer_type<block_type<>>`
        return _store_block_pointer(ptr, val, mask, boundary_check, cache, eviction, builder)
    else:
        # Store by a tensor of pointers or a pointer of scalar: `block_type<pointer_type<>>` or `pointer_type<>`
        return _store_legacy(ptr, val, mask, boundary_check, cache, eviction, builder)


#########
# atomic
#########


def atomic_load(ptr: tl.tensor, sem: str, builder: ir.builder) -> tl.tensor:
    if sem is None:
        sem = "acquire"
    if sem not in {"relaxed", "acquire"}:
        raise ValueError(f"atomic_load: Unsupported sem argument '{sem}', must be 'relaxed' or 'acquire'")
    sem = _str_to_sem(sem)

    elt_ty = ptr.type.scalar.element_ty
    if ptr.type.is_block() or elt_ty.is_block():
        raise ValueError(f"atomic_load: Only scalar pointers are supported, but got {ptr.type}")

    return tl.tensor(builder.create_atomic_load(ptr.handle, sem), elt_ty)


def atomic_store(ptr: tl.tensor, value: tl.tensor, sem: str, builder: ir.builder) -> tl.tensor:
    if sem is None:
        sem = "release"
    if sem not in {"relaxed", "release"}:
        raise ValueError(f"atomic_store: Unsupported sem argument '{sem}', must be 'relaxed' or 'release'")
    sem = _str_to_sem(sem)

    elt_ty = ptr.type.scalar.element_ty
    if ptr.type.is_block() or elt_ty.is_block():
        raise ValueError(f"atomic_store: Only scalar pointers are supported, but got {ptr.type}")

    # Cast to target data type
    value = cast(value, elt_ty, builder)

    return tl.tensor(builder.create_atomic_store(ptr.handle, value.handle, sem), elt_ty)


def atomic_cas(ptr: tl.tensor,
               cmp: tl.tensor,
               val: tl.tensor,
               sem: str,
               builder: ir.builder) -> tl.tensor:
    sem = _str_to_sem(sem)
    element_ty = ptr.type.scalar.element_ty
    if element_ty.primitive_bitwidth not in [16, 32, 64]:
        raise ValueError("atomic_cas only supports elements with width {16, 32, 64}")
    return tl.tensor(builder.create_atomic_cas(ptr.handle, cmp.handle, val.handle, sem), val.type)


def atom_red_typechecking_impl(ptr: tl.tensor,
                               val: tl.tensor,
                               mask: tl.tensor,
                               op: str,
                               builder: ir.builder) -> Tuple[tl.tensor, tl.tensor, tl.tensor]:
    if not ptr.type.scalar.is_ptr():
        raise ValueError("Pointer argument of store instruction is " + ptr.type.__repr__())
    element_ty = ptr.type.scalar.element_ty
    if element_ty is tl.float16 and op != 'add':
        raise ValueError("atomic_" + op + " does not support fp16")
    if element_ty in [tl.int1, tl.int8, tl.int16, tl.bfloat16]:
        raise ValueError("atomic_" + op + " does not support " + str(element_ty))
    if ptr.type.is_block():
        if mask:
            mask = broadcast_impl_shape(mask, ptr.type.get_block_shapes(), builder)
        if val:
            val = broadcast_impl_shape(val, ptr.type.get_block_shapes(), builder)
    val = cast(val, ptr.type.scalar.element_ty, builder)
    if not mask:
        mask_ir = builder.get_int1(True)
        mask_ty = tl.int1
        if ptr.type.is_block():
            mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
            mask_ty = tl.block_type(tl.int1, ptr.type.get_block_shapes())
        mask = tl.tensor(mask_ir, mask_ty)
    return ptr, val, mask


def atomic_max(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               sem: str,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'max', builder)
    sem = _str_to_sem(sem)
    sca_ty = val.type.scalar
    # direct call to atomic_max for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MAX,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle,
                                                       sem),
                             val.type)
        else:
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle,
                                                       sem),
                             val.type)
    # for float
    # return atomic_smax(i_ptr, i_val) if val >= 0
    # return atomic_umin(i_ptr, i_val) if val < 0
    if sca_ty not in {tl.float32, tl.float64}:
        raise TypeError(f"atomic_max not supported for dtype {sca_ty}")

    itype = tl.int32 if sca_ty == tl.float32 else tl.float64
    zero = full([], 0.0, sca_ty, builder)

    i_val = bitcast(val, itype, builder)
    i_ptr = bitcast(ptr, tl.pointer_type(itype, 1), builder)
    pos = greater_equal(val, zero, builder)
    neg = less_than(val, zero, builder)
    pos_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, i_ptr.handle, i_val.handle, and_(mask, pos, builder).handle, sem), i_val.type)
    neg_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN, i_ptr.handle, i_val.handle, and_(mask, neg, builder).handle, sem), i_val.type)
    ret = where(pos, pos_ret, neg_ret, builder)
    return bitcast(ret, sca_ty, builder)


def atomic_min(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               sem: str,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'min', builder)
    sem = _str_to_sem(sem)
    sca_ty = val.type.scalar
    # direct call to atomic_min for integers
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MIN,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle,
                                                       sem),
                             val.type)
        else:
            return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMIN,
                                                       ptr.handle,
                                                       val.handle,
                                                       mask.handle,
                                                       sem),
                             val.type)
    # for float
    # return atomic_smin(i_ptr, i_val) if val >= 0
    # return atomic_umax(i_ptr, i_val) if val < 0
    if sca_ty not in {tl.float32, tl.float64}:
        raise TypeError(f"atomic_min not supported for dtype {sca_ty}")

    itype = tl.int32 if sca_ty == tl.float32 else tl.float64
    zero = full([], 0.0, sca_ty, builder)

    i_val = bitcast(val, itype, builder)
    i_ptr = bitcast(ptr, tl.pointer_type(itype, 1), builder)
    pos = greater_equal(val, zero, builder)
    neg = less_than(val, zero, builder)
    pos_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.MIN,
                                                  i_ptr.handle,
                                                  i_val.handle,
                                                  and_(mask, pos, builder).handle,
                                                  sem),
                        i_val.type)
    neg_ret = tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.UMAX,
                                                  i_ptr.handle,
                                                  i_val.handle,
                                                  and_(mask, neg, builder).handle,
                                                  sem),
                        i_val.type)
    ret = where(pos, pos_ret, neg_ret, builder)
    return bitcast(ret, sca_ty, builder)


def atomic_add(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               sem: str,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'add', builder)
    sem = _str_to_sem(sem)
    sca_ty = val.type.scalar
    op = ir.ATOMIC_OP.FADD if sca_ty.is_floating() else ir.ATOMIC_OP.ADD
    return tl.tensor(builder.create_atomic_rmw(op, ptr.handle, val.handle, mask.handle, sem), val.type)


def atomic_and(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               sem: str,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'and', builder)
    sem = _str_to_sem(sem)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.AND, ptr.handle, val.handle, mask.handle, sem), val.type)


def atomic_or(ptr: tl.tensor,
              val: tl.tensor,
              mask: tl.tensor,
              sem: str,
              builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'or', builder)
    sem = _str_to_sem(sem)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.OR, ptr.handle, val.handle, mask.handle, sem), val.type)


def atomic_xor(ptr: tl.tensor,
               val: tl.tensor,
               mask: tl.tensor,
               sem: str,
               builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'xor', builder)
    sem = _str_to_sem(sem)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XOR, ptr.handle, val.handle, mask.handle, sem), val.type)


def atomic_xchg(ptr: tl.tensor,
                val: tl.tensor,
                mask: tl.tensor,
                sem: str,
                builder: ir.builder) -> tl.tensor:
    ptr, val, mask = atom_red_typechecking_impl(ptr, val, mask, 'xchg', builder)
    sem = _str_to_sem(sem)
    return tl.tensor(builder.create_atomic_rmw(ir.ATOMIC_OP.XCHG, ptr.handle, val.handle, mask.handle, sem), val.type)

# ===----------------------------------------------------------------------===//
#                               Linear Algebra
# ===----------------------------------------------------------------------===//


def gpu_has_mfma() -> bool:
    if not is_hip():
        return False
    return True  # mfma supported in ['gfx908', 'gfx90a']


def mfma_supported(M, N, K, allow_tf32, ret_scalar_ty) -> bool:
    if not gpu_has_mfma():
        return False
    # TODO: Add check for configurations and types.
    return True


def dot(lhs: tl.tensor,
        rhs: tl.tensor,
        acc: tl.tensor,
        allow_tf32: bool,
        max_num_imprecise_acc: int,
        out_dtype: tl.dtype,
        builder: ir.builder) -> tl.tensor:
    def assert_dtypes_valid(lhs_dtype, rhs_dtype, arch):
        # Checks for non-cuda archs
        if not _is_cuda(builder.arch):
            assert lhs_dtype == rhs_dtype, f"First input ({lhs_dtype}) and second input ({rhs_dtype}) must have the same dtype!"
            return
        # Checks for cuda arch
        if arch < 90:
            assert not lhs_dtype.is_fp8e4nv() and not rhs_dtype.is_fp8e4nv(), "Dot op does not support fp8e4nv on CUDA arch < 90"
            if lhs_dtype.is_fp8() and rhs_dtype.is_fp8():
                return
            assert lhs_dtype == rhs_dtype, f"First input ({lhs_dtype}) and second input ({rhs_dtype}) must have the same dtype!"
        else:
            assert not lhs_dtype.is_fp8e4b15() and not rhs_dtype.is_fp8e4b15(), "Dot op does not support fp8e4b15 on CUDA arch >= 90"
            assert not lhs_dtype.is_fp8e4b15x4() and not rhs_dtype.is_fp8e4b15x4(), "Dot op does not support fp8e4b15x4 on CUDA arch >= 90"
            if lhs_dtype.is_int() or rhs_dtype.is_int():
                assert lhs_dtype == rhs_dtype, f"Both operands must be same type. First operand ({lhs_dtype}) and second operand ({rhs_dtype})"
                assert lhs_dtype.is_int8() or lhs_dtype.is_uint8(), f"Both operands must be either int8 or uint8. Operand type ({lhs_dtype})"
            elif lhs_dtype.is_fp8() or rhs_dtype.is_fp8():
                assert lhs_dtype.is_fp8e4nv() or lhs_dtype.is_fp8e5(), f"Only supports fp8e4nv or fp8e5. First operand ({lhs_dtype})"
                assert rhs_dtype.is_fp8e4nv() or rhs_dtype.is_fp8e5(), f"Only supports fp8e4nv or fp8e5. Second operand ({rhs_dtype})"
            else:
                assert lhs_dtype.is_fp16() or lhs_dtype.is_bf16() or lhs_dtype.is_fp32() or lhs_dtype.is_int1(), f"Unsupported dtype {lhs_dtype}"
                assert rhs_dtype.is_fp16() or rhs_dtype.is_bf16() or rhs_dtype.is_fp32() or rhs_dtype.is_int1(), f"Unsupported dtype {rhs_dtype}"
                assert lhs_dtype == rhs_dtype, f"First input ({lhs_dtype}) and second input ({rhs_dtype}) must have the same dtype!"

    assert lhs.type.is_block() and rhs.type.is_block()

    assert_dtypes_valid(lhs.dtype, rhs.dtype, builder.arch)

    assert len(lhs.shape) == 2, f"First input shape ({lhs.shape}) is not two dimensional!"
    assert len(rhs.shape) == 2, f"Second input shape ({rhs.shape}) is not two dimensional!"
    assert lhs.shape[1].value == rhs.shape[0].value, f"First input shape ({lhs.shape}) and second input shape {rhs.shape} are not compatible for matmul (second index of first shape ({lhs.shape[1].value}) must be equal to first index of second shape ({rhs.shape[0].value})"
    assert lhs.shape[0].value >= 16 and lhs.shape[1].value >= 16 \
        and rhs.shape[1].value >= 16, \
        f"All values in both first input shape ({lhs.shape}) and second input shape ({rhs.shape}) must be >= 16!"
    if lhs.type.scalar.is_int():
        assert lhs.type.scalar == tl.int8, "only int8 supported!"
        # TODO: This is CUDA specific, check if ROCm has the same limitation
        assert lhs.shape[1].value >= 32, "small blocks not supported!"
        _0 = builder.get_int32(0)
        ret_scalar_ty = tl.int32
    elif out_dtype.is_bf16():
        raise ValueError("out_dtype=bfloat16 is unsupported. Please use out_dtype=float32/float16 and cast with `.to(tl.bfloat16)`")
    elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
        _0 = builder.get_fp32(0)
        ret_scalar_ty = tl.float32
    else:
        _0 = builder.get_fp16(0) if out_dtype.is_fp16() else builder.get_fp32(0)
        ret_scalar_ty = out_dtype

    M = lhs.type.shape[0]
    N = rhs.type.shape[1]

    # Cast operands of types f16 and i8 for configurations where FMA only supported.
    if is_hip() and not mfma_supported(M, N, lhs.type.shape[1], allow_tf32, ret_scalar_ty):
        ret_cast_scalar_ty = tl.float32 if lhs.type.scalar.is_int() else ret_scalar_ty
        lhs = cast(lhs, ret_cast_scalar_ty, builder)
        rhs = cast(rhs, ret_cast_scalar_ty, builder)
        if ret_cast_scalar_ty == tl.float16:
            _0 = builder.create_splat(builder.get_fp16(0), [M, N])
        else:
            _0 = builder.create_splat(builder.get_fp32(0), [M, N])
        ret_ty = tl.block_type(ret_cast_scalar_ty, [M, N])
        ret = tl.tensor(builder.create_dot(lhs.handle, rhs.handle, _0, allow_tf32),
                        ret_ty)
        return cast(ret, ret_scalar_ty, builder)
    if is_hip() and mfma_supported(M, N, lhs.type.shape[1], allow_tf32, ret_scalar_ty) and ret_scalar_ty.primitive_bitwidth < 32:
        if lhs.type.scalar.is_int():
            ret_dot_scalar_ty = tl.int32
            _0 = builder.create_splat(builder.get_int32(0), [M, N])
        else:
            ret_dot_scalar_ty = tl.float32
            _0 = builder.create_splat(builder.get_fp32(0), [M, N])
        ret_ty = tl.block_type(ret_dot_scalar_ty, [M, N])
        ret = tl.tensor(builder.create_dot(lhs.handle, rhs.handle, _0, allow_tf32),
                        ret_ty)
        return cast(ret, ret_scalar_ty, builder)
    ret_ty = tl.block_type(ret_scalar_ty, [M, N])
    if acc is None:
        acc_handle = builder.create_splat(_0, [M, N])
    else:
        acc_handle = acc.handle
        assert acc.type == ret_ty

    # max_num_imprecise_acc only applies to fp8 -> fp32 dot on sm_90
    if not (_is_cuda(builder.arch) and builder.arch == 90 and lhs.dtype.is_fp8() and rhs.dtype.is_fp8() and ret_scalar_ty.is_fp32()):
        max_num_imprecise_acc = 0
    if max_num_imprecise_acc is None:
        max_num_imprecise_acc = 2**30

    return tl.tensor(builder.create_dot(lhs.handle, rhs.handle, acc_handle, allow_tf32, max_num_imprecise_acc),
                     ret_ty)


# ===----------------------------------------------------------------------===//
#                               Indexing
# ===----------------------------------------------------------------------===//

def where(condition: tl.tensor,
          x: tl.tensor,
          y: tl.tensor,
          builder: ir.builder) -> tl.tensor:
    condition = cast(condition, tl.int1, builder)
    if condition.type.is_block():
        condition, x = broadcast_impl_value(condition, x, builder)
        x, y = broadcast_impl_value(x, y, builder)
        condition, x = broadcast_impl_value(condition, x, builder)

    x, y = binary_op_type_checking_impl(x, y, builder, True, True)
    if not condition.type.is_block():
        condition, _ = broadcast_impl_value(condition, x, builder)
    ret_ty = x.type
    return tl.tensor(builder.create_select(condition.handle, x.handle, y.handle), ret_ty)

# ===----------------------------------------------------------------------===//
#                               Reduction
# ===----------------------------------------------------------------------===


def reduction(
    inputs: Sequence[tl.tensor], axis: int, region_builder_fn, builder: ir.builder
) -> Tuple[tl.tensor, ...]:
    if axis is None:
        new_inputs = []
        for i in range(len(inputs)):
            new_shape = [inputs[i].numel.value]
            new_inputs.append(view(inputs[i], new_shape, builder))
        inputs = tuple(new_inputs)
        axis = 0
    # get result shape
    shape = inputs[0].type.shape
    ret_shape = [s for i, s in enumerate(shape) if i != axis]
    for t in inputs:
        assert t.type.shape == shape

    def wrap_tensor(x, scalar_ty):
        if ret_shape:
            res_ty = tl.block_type(scalar_ty, ret_shape)
        else:
            # 0d-tensor -> scalar
            res_ty = scalar_ty
        return tl.tensor(x, res_ty)

    reduce_op = builder.create_reduce([t.handle for t in inputs], axis)
    region_builder_fn(reduce_op)
    reduce_op.verify()

    return tuple(
        wrap_tensor(reduce_op.get_result(i), inputs[i].type.scalar)
        for i in range(len(inputs))
    )


# ===----------------------------------------------------------------------===
#                               Associative Scan
# ===----------------------------------------------------------------------===


def associative_scan(
    inputs: Sequence[tl.tensor], axis: int, region_builder_fn, builder: ir.builder
) -> Tuple[tl.tensor, ...]:
    if len(inputs) != 1:
        raise ValueError("Current implementation only support single tensor input")
    shape = inputs[0].type.shape

    def wrap_tensor(x, scalar_ty):
        res_ty = tl.block_type(scalar_ty, shape)
        return tl.tensor(x, res_ty)

    scan_op = builder.create_scan([t.handle for t in inputs], axis)
    region_builder_fn(scan_op)
    scan_op.verify()

    return tuple(
        wrap_tensor(scan_op.get_result(i), inputs[i].type.scalar)
        for i in range(len(inputs))
    )


# ===----------------------------------------------------------------------===
#                               Math
# ===----------------------------------------------------------------------===

def _check_dtype(dtypes: List[str]) -> T:
    """
    We're following libdevice's convention to check accepted data types for math functions.
    It is not a good practice to support all data types as accelerators/GPUs don't support
    many float16 and bfloat16 math operations.
    We should let the users know that they are using and invoke explicit cast to convert
    the data type to the supported one.
    """
    def wrapper(fn):
        @wraps(fn)
        def check(*args, **kwargs):
            # concatenate args and kwargs
            all_args = list(args) + list(kwargs.values())
            for arg in [a for a in all_args if isinstance(a, tl.tensor)]:
                if arg.type.scalar.name not in dtypes:
                    raise ValueError(f"Expected dtype {dtypes} but got {arg.type.scalar.name}")
            return fn(*args, **kwargs)
        return check

    return wrapper


def umulhi(x: tl.tensor, y: tl.tensor, builder: ir.builder) -> tl.tensor:
    x, y = binary_op_type_checking_impl(x, y, builder)
    # FIXME(Keren): not portable, should be fixed
    from . import math
    return math.mulhi(x, y, _builder=builder)


@_check_dtype(dtypes=["fp32", "fp64"])
def floor(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    # FIXME(Keren): not portable, should be fixed
    from . import math
    return math.floor(x, _builder=builder)


@_check_dtype(dtypes=["fp32", "fp64"])
def exp(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_exp(x.handle), x.type)


@_check_dtype(dtypes=["fp32", "fp64"])
def log(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_log(x.handle), x.type)


@_check_dtype(dtypes=["fp32", "fp64"])
def cos(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_cos(x.handle), x.type)


@_check_dtype(dtypes=["fp32", "fp64"])
def sin(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_sin(x.handle), x.type)


@_check_dtype(dtypes=["fp32", "fp64"])
def sqrt(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_sqrt(x.handle), x.type)


def abs(x: tl.tensor, builder: ir.builder) -> tl.tensor:
    dtype = x.dtype
    if dtype.is_floating():
        return tl.tensor(builder.create_fabs(x.handle), x.type)
    elif dtype.is_int_signed():
        return tl.tensor(builder.create_iabs(x.handle), x.type)
    elif dtype.is_int_unsigned():
        return x  # no-op
    else:
        assert False, f"Unexpected dtype {dtype}"


##


def multiple_of(x: tl.tensor, values: List[int]) -> tl.tensor:
    if max(1, len(x.shape)) != len(values):
        raise ValueError("Shape of input to multiple_of does not match the length of values")
    x.handle.set_attr("tt.divisibility", ir.make_attr(values, x.handle.get_context()))
    return x


def max_contiguous(x: tl.tensor, values: List[int]) -> tl.tensor:
    if len(x.shape) != len(values):
        raise ValueError("Shape of input to max_contiguous does not match the length of values")
    x.handle.set_attr("tt.contiguity", ir.make_attr(values, x.handle.get_context()))
    return x


def max_constancy(x: tl.tensor, values: List[int]) -> tl.tensor:
    if len(x.shape) != len(values):
        raise ValueError("Shape of input to max_constancy does not match the length of values")
    x.handle.set_attr("tt.constancy", ir.make_attr(values, x.handle.get_context()))
    return x


def debug_barrier(builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_barrier(), tl.void)


def device_print(prefix: str, args: List[tl.tensor], builder: ir.builder) -> tl.tensor:
    new_args = []
    for arg in args:
        new_args.append(arg.handle)
    return tl.tensor(builder.create_print(prefix, new_args), tl.void)


def device_assert(cond: tl.tensor, msg: str, file_name: str, func_name, lineno: int, builder: ir.builder) -> tl.tensor:
    cond_ty = cond.type
    if not cond_ty.is_block():
        cond_ty = tl.block_type(cond_ty.scalar, (1,))
        cond = tl.tensor(builder.create_splat(cond.handle, (1,)), cond_ty)
    return tl.tensor(builder.create_assert(cond.handle, msg, file_name, func_name, lineno), tl.void)


def _convert_elem_to_ir_value(builder, elem, require_i64):
    if isinstance(elem, int):
        elem = tl.constexpr(elem)
    if isinstance(elem, tl.constexpr):
        return builder.get_int64(elem.value) if require_i64 else builder.get_int32(elem.value)
    elif isinstance(elem, tl.tensor):
        assert elem.numel.value == 1, "Expected a scalar in shape/strides/offsets"
        assert elem.dtype.is_int(), "Expected an integer scalar type in shape/strides/offsets"
        if elem.dtype != tl.int64 and require_i64:
            return builder.create_int_cast(elem.handle, builder.get_int64_ty(), elem.dtype.is_int_signed())
        elif elem.dtype != tl.int32:
            return builder.create_int_cast(elem.handle, builder.get_int32_ty(), elem.dtype.is_int_signed())
        return elem.handle
    assert False, f"Unsupported element type in shape/strides/offsets: {type(elem)}"


def _convert_to_ir_values(builder, list_like, require_i64=True):
    if hasattr(list_like, "__iter__"):
        return [_convert_elem_to_ir_value(builder, elem, require_i64) for elem in list_like]
    return [_convert_elem_to_ir_value(builder, list_like, require_i64)]


def make_block_ptr(base: tl.tensor, shape, strides, offsets, block_shape, order, builder: ir.builder) -> tl.tensor:
    # Convert dynamic arguments to IR values
    # NOTES(Chenggang): current `shape/strides` are `int64_t`, while `offsets/block_shape` are `int32_t`
    shape = _convert_to_ir_values(builder, shape)
    strides = _convert_to_ir_values(builder, strides)
    offsets = _convert_to_ir_values(builder, offsets, require_i64=False)

    # Check `base` type
    if not base.type.is_ptr() or base.type.element_ty.is_block():
        raise ValueError("Expected `base` to be a pointer type (but not a block pointer type or others)")

    # Treat `pointer_type<tl.int1>` as `pointer_type<tl.int8>`
    if base.type.element_ty == tl.int1:
        base = cast(base, tl.pointer_type(tl.int8, base.type.address_space), builder)

    # Check whether `block_shape` is static
    if not hasattr(block_shape, "__iter__"):
        block_shape = [block_shape]
    block_shape = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in block_shape]
    assert all([isinstance(elem, int) and -2**31 <= elem < 2**31 for elem in block_shape]), \
        "Expected a list of constant integers (`int32_t` range) in `block_shape`"

    # Check `order`
    if not hasattr(order, "__iter__"):
        order = [order]
    order = [elem.value if isinstance(elem, tl.constexpr) else elem for elem in order]
    assert sorted(order) == list(range(len(order))), "Expected a permutation of (0, 1, ..., len(order)-1) in order"

    # Must have same length
    assert all([len(block_shape) == len(list_like) for list_like in [shape, strides, offsets, order]]), \
        "Expected shape/strides/offsets/block_shape to have the same length"

    # Build value, the type is:
    #   `pointer_type<blocked<shape, element_type>>` in Python
    #   `tt.ptr<tensor<shape, element_type>>` in MLIR
    handle = builder.create_make_block_ptr(base.handle, shape, strides, offsets, block_shape, order)
    return tl.tensor(handle, tl.pointer_type(tl.block_type(base.type.element_ty, block_shape)))


def advance(base: tl.tensor, offsets, builder: ir.builder) -> tl.tensor:
    # Convert dynamic offsets to IR values
    offsets = _convert_to_ir_values(builder, offsets, require_i64=False)

    # Advanced block pointer type is the same as before
    return tl.tensor(builder.create_advance(base.handle, offsets), base.type)
