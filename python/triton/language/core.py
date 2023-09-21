from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Callable, List, Sequence, TypeVar

from .._C.libtriton.triton import ir
from . import semantic

T = TypeVar('T')

TRITON_MAX_TENSOR_NUMEL = 131072

TRITON_BUILTIN = "__triton_builtin__"


def builtin(fn: T) -> T:
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError(
                "Did you forget to add @triton.jit ? "
                "(`_builder` argument must be provided outside of JIT functions.)"
            )
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)

    return wrapper


def is_builtin(fn) -> bool:
    """Is this a registered triton builtin function?"""
    return getattr(fn, TRITON_BUILTIN, False)


def _to_tensor(x, builder):
    if isinstance(x, bool):
        return tensor(builder.get_int1(x), int1)
    # Note: compile-time const integers are represented by unsigned values
    elif isinstance(x, int):
        if -2**31 <= x < 2**31:
            return tensor(builder.get_int32(x), int32)
        elif 2**31 <= x < 2**32:
            return tensor(builder.get_int32(x), uint32)
        elif -2**63 <= x < 2**63:
            return tensor(builder.get_int64(x), int64)
        elif 2**63 <= x < 2**64:
            return tensor(builder.get_int64(x), uint64)
        else:
            raise RuntimeError(f'Nonrepresentable integer {x}.')
    elif isinstance(x, float):
        min_float32 = 2 ** -126
        max_float32 = (2 - 2**-23) * 2**127
        abs_x = __builtins__['abs'](x)
        if abs_x == float("inf") or\
           abs_x == 0.0 or \
           x != x or \
           min_float32 <= abs_x <= max_float32:
            return tensor(builder.get_fp32(x), float32)
        else:
            return tensor(builder.get_fp64(x), float64)

    elif isinstance(x, constexpr):
        return _to_tensor(x.value, builder)
    elif isinstance(x, tensor):
        return x
    assert False, f"cannot convert {x} of type {type(x)} to tensor"


class dtype:
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8e4b15', 'fp8e4b15x4', 'fp8e4nv', 'fp8e5', 'fp16', 'bf16', 'fp32', 'fp64']
    STANDARD_FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    def __init__(self, name):
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES, name
        if name in dtype.SINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.SIGNED
            self.int_bitwidth = int(name.split('int')[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.UINT_TYPES:
            self.int_signedness = dtype.SIGNEDNESS.UNSIGNED
            self.int_bitwidth = int(name.split('int')[-1])
            self.primitive_bitwidth = self.int_bitwidth
        elif name in dtype.FP_TYPES:
            if name == 'fp8e4b15':
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
                self.exponent_bias = 15
            elif name == 'fp8e4b15x4':
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
                self.exponent_bias = 15
            elif name == 'fp8e4nv':
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
                self.exponent_bias = 7
            elif name == 'fp8e5':
                self.fp_mantissa_width = 2
                self.primitive_bitwidth = 8
                self.exponent_bias = 15
            elif name == 'fp16':
                self.fp_mantissa_width = 10
                self.primitive_bitwidth = 16
                self.exponent_bias = 15
            elif name == 'bf16':
                self.fp_mantissa_width = 7
                self.primitive_bitwidth = 16
                self.exponent_bias = 127
            elif name == 'fp32':
                self.fp_mantissa_width = 23
                self.primitive_bitwidth = 32
                self.exponent_bias = 127
            elif name == 'fp64':
                self.fp_mantissa_width = 53
                self.primitive_bitwidth = 64
                self.exponent_bias = 1023
            else:
                raise RuntimeError(f'Unsupported floating-point type {name}')
        elif name == 'void':
            self.primitive_bitwidth = 0

    def is_fp8(self):
        return 'fp8' in self.name

    def is_fp8e4nv(self):
        return self.name == 'fp8e4nv'

    def is_fp8e4b15(self):
        return self.name == 'fp8e4b15'

    def is_fp8e4b15x4(self):
        return self.name == 'fp8e4b15x4'

    def is_fp8e5(self):
        return self.name == 'fp8e5'

    def is_fp16(self):
        return self.name == 'fp16'

    def is_bf16(self):
        return self.name == 'bf16'

    def is_fp32(self):
        return self.name == 'fp32'

    def is_fp64(self):
        return self.name == 'fp64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def is_floating(self):
        return self.name in dtype.FP_TYPES

    def is_standard_floating(self):
        return self.name in dtype.STANDARD_FP_TYPES

    def is_int_signed(self):
        return self.name in dtype.SINT_TYPES

    def is_int_unsigned(self):
        return self.name in dtype.UINT_TYPES

    def is_int(self):
        return self.name in dtype.SINT_TYPES + dtype.UINT_TYPES

    def is_bool(self):
        return self.is_int1()

    @staticmethod
    def is_dtype(type_str):
        return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES + dtype.OTHER_TYPES

    @staticmethod
    def is_void():
        raise RuntimeError("Not implemented")

    @staticmethod
    def is_block():
        return False

    @staticmethod
    def is_ptr():
        return False

    def __eq__(self, other: dtype):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other: dtype):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name,))

    @property
    def scalar(self):
        return self

    def to_ir(self, builder: ir.builder) -> ir.type:
        if self.name == 'void':
            return builder.get_void_ty()
        elif self.name == 'int1':
            return builder.get_int1_ty()
        elif self.name in ('int8', 'uint8'):
            return builder.get_int8_ty()
        elif self.name in ('int16', 'uint16'):
            return builder.get_int16_ty()
        elif self.name in ('int32', 'uint32'):
            return builder.get_int32_ty()
        elif self.name in ('int64', 'uint64'):
            return builder.get_int64_ty()
        elif self.name == 'fp8e5':
            return builder.get_fp8e5_ty()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_ty()
        elif self.name == 'fp8e4b15':
            return builder.get_fp8e4b15_ty()
        elif self.name == 'fp8e4b15x4':
            return builder.get_fp8e4b15x4_ty()
        elif self.name == 'fp16':
            return builder.get_half_ty()
        elif self.name == 'bf16':
            return builder.get_bf16_ty()
        elif self.name == 'fp32':
            return builder.get_float_ty()
        elif self.name == 'fp64':
            return builder.get_double_ty()
        raise ValueError(f'fail to convert {self} to ir type')

    def __str__(self):
        return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        return f'triton.language.{self.name}'


class pointer_type(dtype):
    def __init__(self, element_ty: dtype, address_space: int = 1):
        if not isinstance(element_ty, dtype):
            raise TypeError('element_ty is a {type(element_ty).__name__}.')
        self.element_ty = element_ty
        self.address_space = address_space

        self.name = self.__str__()

    def to_ir(self, builder: ir.builder) -> ir.pointer_type:
        return builder.get_ptr_ty(self.element_ty.to_ir(builder), 1)

    def __str__(self):
        return f'pointer<{self.element_ty}>'

    def __repr__(self):
        return self.__str__()

    def is_ptr(self):
        return True

    def __eq__(self, other: pointer_type) -> bool:
        if not isinstance(other, pointer_type):
            return False
        return self.element_ty == other.element_ty and self.address_space == other.address_space

    def __ne__(self, other: pointer_type) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self):
        return self


class block_type(dtype):
    def __init__(self, element_ty: dtype, shape: List):
        self.element_ty = element_ty

        # Note that block_type's shape is a list of int
        # while tensor's shape is a list of constexpr.

        # shape can be empty ([]) when an input is a 0D tensor.
        if not shape:
            raise TypeError('0d block_type is forbidden')
        if isinstance(shape[0], constexpr):
            shape = [s.value for s in shape]

        self.shape = shape
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        if self.numel > TRITON_MAX_TENSOR_NUMEL:
            raise ValueError(f"numel ({self.numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")

        self.name = self.__str__()

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):
        return f'<{self.shape}, {self.element_ty}>'

    def __repr__(self):
        return self.__str__()

    def is_block(self):
        return True

    def get_block_shapes(self) -> List[int]:
        return self.shape

    def __eq__(self, other: block_type) -> bool:
        if not isinstance(other, block_type):
            return False
        return self.element_ty == other.element_ty and self.shape == other.shape

    def __ne__(self, other: block_type) -> bool:
        return not self.__eq__(other)

    @property
    def scalar(self):
        return self.element_ty


class function_type(dtype):
    def __init__(self, ret_types: List[dtype], param_types: List[dtype]) -> None:
        self.ret_types = ret_types
        self.param_types = param_types

    def __str__(self):
        return f'fn ({self.param_types}) -> {self.ret_types}'

    def to_ir(self, builder: ir.builder):
        ir_param_types = [ty.to_ir(builder) for ty in self.param_types]
        ret_types = [ret_type.to_ir(builder) for ret_type in self.ret_types]
        return builder.get_function_ty(ir_param_types, ret_types)


# scalar types
void = dtype('void')
int1 = dtype('int1')
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float8e5 = dtype('fp8e5')
float8e4nv = dtype('fp8e4nv')
float8e4b15 = dtype('fp8e4b15')
float8e4b15x4 = dtype('fp8e4b15x4')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')
# pointer types
pi32_t = pointer_type(int32)

# -----------------------
# constexpr
# -----------------------


class constexpr:
    """
    This class is used to store a value that is known at compile-time.
    """

    def __init__(self, value):
        if isinstance(value, constexpr):
            self.value = value.value
        else:
            self.value = value

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def __index__(self):
        return self.value

    def __add__(self, other):
        return constexpr(self.value + other.value)

    def __radd__(self, other):
        return constexpr(other.value + self.value)

    def __sub__(self, other):
        return constexpr(self.value - other.value)

    def __rsub__(self, other):
        return constexpr(other.value - self.value)

    def __mul__(self, other):
        return constexpr(self.value * other.value)

    def __mod__(self, other):
        return constexpr(self.value % other.value)

    def __rmul__(self, other):
        return constexpr(other.value * self.value)

    def __truediv__(self, other):
        return constexpr(self.value / other.value)

    def __rtruediv__(self, other):
        return constexpr(other.value / self.value)

    def __floordiv__(self, other):
        return constexpr(self.value // other.value)

    def __rfloordiv__(self, other):
        return constexpr(other.value // self.value)

    def __gt__(self, other):
        return constexpr(self.value > other.value)

    def __rgt__(self, other):
        return constexpr(other.value > self.value)

    def __ge__(self, other):
        return constexpr(self.value >= other.value)

    def __rge__(self, other):
        return constexpr(other.value >= self.value)

    def __lt__(self, other):
        return constexpr(self.value < other.value)

    def __rlt__(self, other):
        return constexpr(other.value < self.value)

    def __le__(self, other):
        return constexpr(self.value <= other.value)

    def __rle__(self, other):
        return constexpr(other.value <= self.value)

    def __eq__(self, other):
        return constexpr(self.value == other.value)

    def __ne__(self, other):
        return constexpr(self.value != other.value)

    def __bool__(self):
        return bool(self.value)

    def __neg__(self):
        return constexpr(-self.value)

    def __and__(self, other):
        return constexpr(self.value & other.value)

    def logical_and(self, other):
        return constexpr(self.value and other.value)

    def __or__(self, other):
        return constexpr(self.value | other.value)

    def __xor__(self, other):
        return constexpr(self.value ^ other.value)

    def logical_or(self, other):
        return constexpr(self.value or other.value)

    def __pos__(self):
        return constexpr(+self.value)

    def __invert__(self):
        return constexpr(~self.value)

    def __pow__(self, other):
        return constexpr(self.value ** other.value)

    def __rshift__(self, other):
        return constexpr(self.value >> other.value)

    def __lshift__(self, other):
        return constexpr(self.value << other.value)

    def __not__(self):
        return constexpr(not self.value)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)


class tensor:
    def __init__(self, handle, type: dtype):
        # IR handle
        self.handle = handle
        # Block shape
        self.shape = type.shape if type.is_block() else ()
        self.numel = 1
        for s in self.shape:
            self.numel *= s
        self.numel = constexpr(self.numel)
        self.type = type  # Tensor type (can be block_type)
        # Following the practice in pytorch, dtype is scalar type
        self.dtype = type.scalar
        self.shape = [constexpr(s) for s in self.shape]

    def __str__(self) -> str:
        # ex. "float32[16, 32]"
        return str(self.dtype) + '[' + ', '.join(str(s) for s in self.shape) + ']'

    @builtin
    def __add__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.add(self, other, _builder)

    @builtin
    def __radd__(self, other, _builder=None):
        return self.__add__(other, _builder=_builder)

    @builtin
    def __sub__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.sub(self, other, _builder)

    @builtin
    def __rsub__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.sub(other, self, _builder)

    @builtin
    def __mul__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.mul(self, other, _builder)

    @builtin
    def __rmul__(self, other, _builder=None):
        return self.__mul__(other, _builder=_builder)

    @builtin
    def __truediv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.truediv(self, other, _builder)

    @builtin
    def __rtruediv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.truediv(other, self, _builder)

    @builtin
    def __floordiv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.floordiv(self, other, _builder)

    @builtin
    def __rfloordiv__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.floordiv(other, self, _builder)

    @builtin
    def __mod__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.mod(self, other, _builder)

    @builtin
    def __rmod__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.mod(other, self, _builder)

    # unary operators
    @builtin
    def __neg__(self, _builder=None):
        return semantic.minus(self, _builder)

    @builtin
    def __invert__(self, _builder=None):
        return semantic.invert(self, _builder)

    # bitwise operators

    @builtin
    def __and__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.and_(self, other, _builder)

    @builtin
    def __rand__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.and_(other, self, _builder)

    @builtin
    def __or__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.or_(self, other, _builder)

    @builtin
    def __ror__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.or_(other, self, _builder)

    @builtin
    def __xor__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.xor_(self, other, _builder)

    @builtin
    def __rxor__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.xor_(other, self, _builder)

    @builtin
    def __lshift__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.shl(self, other, _builder)

    @builtin
    def __rlshift__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.shl(other, self, _builder)

    @builtin
    def __rshift__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        if self.dtype.is_int_signed():
            return semantic.ashr(self, other, _builder)
        else:
            return semantic.lshr(self, other, _builder)

    @builtin
    def __rrshift__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        if self.dtype.is_int_signed():
            return semantic.ashr(other, self, _builder)
        else:
            return semantic.lshr(other, self, _builder)

    # >
    @builtin
    def __gt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_than(self, other, _builder)

    @builtin
    def __rgt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_than(other, self, _builder)

    # >=
    @builtin
    def __ge__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_equal(self, other, _builder)

    @builtin
    def __rge__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.greater_equal(other, self, _builder)

    # <
    @builtin
    def __lt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_than(self, other, _builder)

    @builtin
    def __rlt__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_than(other, self, _builder)

    # <=
    @builtin
    def __le__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_equal(self, other, _builder)

    @builtin
    def __rle__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.less_equal(other, self, _builder)

    # ==
    @builtin
    def __eq__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.equal(self, other, _builder)

    @builtin
    def __ne__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.not_equal(self, other, _builder)

    @builtin
    def logical_and(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.logical_and(self, other, _builder)

    @builtin
    def logical_or(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.logical_or(self, other, _builder)

    # note: __not__ isn't actually a magic method in python
    # but it's ok because our ASTVisitor handles it
    @builtin
    def __not__(self, _builder=None):
        return semantic.not_(self, _builder)

    @builtin
    def __getitem__(self, slices, _builder=None):
        if isinstance(slices, (slice, constexpr)):
            slices = [slices]
        ret = self
        for dim, sl in enumerate(slices):
            if sl is None or isinstance(sl, constexpr) and sl.value is None:
                ret = semantic.expand_dims(ret, dim, _builder)
            elif isinstance(sl, slice) and sl.start is None and sl.stop is None and sl.step is None:
                pass
            else:
                assert False, f"unsupported tensor index: {sl}"
        return ret

    @property
    def T(self):
        assert False, "Transposition must be created by the AST Visitor"

    @builtin
    def to(self, dtype, bitcast=False, _builder=None):
        if isinstance(bitcast, constexpr):
            bitcast = bitcast.value
        if bitcast:
            return semantic.bitcast(self, dtype, _builder)
        return semantic.cast(self, dtype, _builder)


# -----------------------
# SPMD Programming Model
# -----------------------
def _constexpr_to_value(v):
    if isinstance(v, constexpr):
        return v.value
    return v


@builtin
def program_id(axis, _builder=None):
    """
    Returns the id of the current program instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    # if axis == -1:
    #     pid0 = program_id(0, _builder)
    #     pid1 = program_id(1, _builder)
    #     pid2 = program_id(2, _builder)
    #     npg0 = num_programs(0, _builder)
    #     npg1 = num_programs(0, _builder)
    #     return pid0 + pid1*npg0 + pid2*npg0*npg1
    axis = _constexpr_to_value(axis)
    return semantic.program_id(axis, _builder)


@builtin
def num_programs(axis, _builder=None):
    """
    Returns the number of program instances launched along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Has to be either 0, 1 or 2.
    :type axis: int
    """
    axis = _constexpr_to_value(axis)
    return semantic.num_programs(axis, _builder)


# -----------------------
# Block Initialization
# -----------------------


@builtin
def arange(start, end, _builder=None):
    """
    Returns contiguous values within the left-closed and right-open interval [:code:`start`, :code:`end`). \
    End - Start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = 131072

    :param start: Start of the interval. Must be a power of two.
    :type start: int32
    :param end: End of the interval. Must be a power of two > start.
    :type end: int32
    """
    start = _constexpr_to_value(start)
    end = _constexpr_to_value(end)
    return semantic.arange(start, end, _builder)


def _shape_check_impl(shape):
    shape = _constexpr_to_value(shape)
    for i, d in enumerate(shape):
        if isinstance(d, int):
            d = constexpr(d)
        if not isinstance(d, constexpr):
            raise TypeError(f"Shape element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    return [_constexpr_to_value(x) for x in shape]


@builtin
def full(shape, value, dtype, _builder=None):
    """
    Returns a tensor filled with the scalar value for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :value value: A scalar value to fill the array with
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    shape = _shape_check_impl(shape)
    value = _constexpr_to_value(value)
    dtype = _constexpr_to_value(dtype)
    return semantic.full(shape, value, dtype, _builder)


# -----------------------
# Shape Manipulation
# -----------------------


@builtin
def broadcast(input, other, _builder=None):
    """
    Tries to broadcast the two given blocks to a common compatible shape.

    :param input: The first input tensor.
    :type input: Block
    :param other: The second input tensor.
    :type other: Block
    """
    return semantic.broadcast_impl_value(input, other, _builder)


@builtin
def broadcast_to(input, shape, _builder=None):
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape: Tuple[int]
    """
    shape = _shape_check_impl(shape)
    return semantic.broadcast_impl_shape(input, shape, _builder)


@builtin
def trans(input, _builder=None):
    """
    Returns a transposed tensor.

    :param input: The input tensor.
    :type input:
    """
    return semantic.trans(input, _builder)


@builtin
def cat(input, other, can_reorder=False, _builder=None):
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input:
    :param other: The second input tensor.
    :type other:
    :param reorder: Compiler hint. If true, the compiler is
    allowed to reorder elements while concatenating inputs.
    Only use if the order does not matter (e.g., result is
    only used in reduction ops)
    """
    return semantic.cat(input, other, can_reorder, _builder)


@builtin
def view(input, shape, _builder=None):
    """
    Returns a tensor with the same elements as `input` but a different shape.
    The order of the elements may not be preserved.

    :param input: The input tensor.
    :type input:
    :param shape: The desired shape.
    :type shape: Tuple[int]

    """
    shape = _shape_check_impl(shape)
    return semantic.view(input, shape, _builder)


@builtin
def reshape(input, shape, _builder=None):
    """
    Returns a tensor with the same number of elements as input but with the
    provided shape.

    :param input: The input tensor.
    :type input:
    :param shape: The new shape.
    :type shape: Tuple[int]
    """
    shape = _shape_check_impl(shape)
    return semantic.reshape(input, shape, _builder)


def _wrap_axis(axis, ndim):
    if not (-ndim <= axis < ndim):
        raise ValueError(f"invalid axis {axis}. Expected {-ndim} <= axis < {ndim}")

    return axis if axis >= 0 else axis + ndim


@builtin
def expand_dims(input, axis, _builder=None):
    """
    Expand the shape of a tensor, by inserting new length-1 dimensions.

    Axis indices are with respect to the resulting tensor, so
    ``result.shape[axis]`` will be 1 for each axis.

    :param input: The input tensor.
    :type input: tl.tensor
    :param axis: The indices to add new axes
    :type axis: int | Sequence[int]

    """
    axis = _constexpr_to_value(axis)
    axes = list(axis) if isinstance(axis, Sequence) else [axis]
    new_ndim = len(input.shape) + len(axes)
    axes = [_wrap_axis(_constexpr_to_value(d), new_ndim) for d in axes]

    if len(set(axes)) != len(axes):
        raise ValueError(f"expand_dims recieved duplicate axes, normalized axes = {axes}")

    ret = input
    for a in sorted(axes):
        ret = semantic.expand_dims(ret, a, _builder)
    return ret

# -----------------------
# Linear Algebra
# -----------------------


@builtin
def dot(input, other, acc=None, allow_tf32=True, max_num_imprecise_acc=None, out_dtype=float32, _builder=None):
    """
    Returns the matrix product of two blocks.

    The two blocks must be two-dimensional and have compatible inner dimensions.

    :param input: The first tensor to be multiplied.
    :type input: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    """
    allow_tf32 = _constexpr_to_value(allow_tf32)
    out_dtype = _constexpr_to_value(out_dtype)
    return semantic.dot(input, other, acc, allow_tf32, max_num_imprecise_acc, out_dtype, _builder)


# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@builtin
def load(pointer, mask=None, other=None, boundary_check=tuple(), padding_option="", cache_modifier="",
         eviction_policy="", volatile=False, _builder=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:
        (1) `pointer` could be a single element pointer, then a scalar will be loaded
            - `mask` and `other` must be scalar too
            - `other` is implicitly typecast to `pointer.dtype.element_ty`
            - `boundary_check` and `padding_option` must be empty
        (2) `pointer` could be element-wise tensor of pointers, in which case:
            - `mask` and `other` are implicitly broadcast to `pointer.shape`
            - `other` is implicitly typecast to `pointer.dtype.element_ty`
            - `boundary_check` and `padding_option` must be empty
        (3) `pointer` could be a block pointer defined by `make_block_ptr`, in which case:
            - `mask` and `other` must be None
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, do padding while out of bound
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    # `mask` and `other` can be constexpr
    if _constexpr_to_value(mask) is not None:
        mask = _to_tensor(mask, _builder)
    if _constexpr_to_value(other) is not None:
        other = _to_tensor(other, _builder)
    padding_option = _constexpr_to_value(padding_option)
    cache_modifier = _constexpr_to_value(cache_modifier)
    eviction_policy = _constexpr_to_value(eviction_policy)
    volatile = _constexpr_to_value(volatile)
    return semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                         volatile, _builder)


@builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="", _builder=None):
    """
    Store a tensor of data into memory locations defined by `pointer`:
        (1) `pointer` could be a single element pointer, then a scalar will be stored
            - `mask` must be scalar too
            - `boundary_check` and `padding_option` must be empty
        (2) `pointer` could be element-wise tensor of pointers, in which case:
            - `mask` is implicitly broadcast to `pointer.shape`
            - `boundary_check` must be empty
        (3) or `pointer` could be a block pointer defined by `make_block_ptr`, in which case:
            - `mask` must be None
            - `boundary_check` can be specified to control the behavior of out-of-bound access
    `value` is implicitly broadcast to `pointer.shape` and typecast to `pointer.dtype.element_ty`.

    :param pointer: The memory location where the elements of `value` are stored
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param value: The tensor of elements to be stored
    :type value: Block
    :param mask: If `mask[idx]` is false, do not store `value[idx]` at `pointer[idx]`
    :type mask: Block of triton.int1, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    """
    # `value` can be constexpr
    value = _to_tensor(value, _builder)
    if _constexpr_to_value(mask) is not None:
        mask = _to_tensor(mask, _builder)
    cache_modifier = _constexpr_to_value(cache_modifier)
    eviction_policy = _constexpr_to_value(eviction_policy)
    return semantic.store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy, _builder)


@builtin
def make_block_ptr(base: tensor, shape, strides, offsets, block_shape, order, _builder=None):
    """
    Returns a pointer to a block in a parent tensor

    :param base: The base pointer to the parent tensor
    :param shape: The shape of the parent tensor
    :param strides: The strides of the parent tensor
    :param offsets: The offsets to the block
    :param block_shape: The shape of the block
    :param order: The order of the original data format
    """
    return semantic.make_block_ptr(base, shape, strides, offsets, block_shape, order, _builder)


@builtin
def advance(base: tensor, offsets, _builder=None):
    """
    Advance a block pointer

    :param base: the block pointer to advance
    :param offsets: the offsets to advance, a tuple by dimension
    """
    return semantic.advance(base, offsets, _builder)

# -----------------------
# Atomic Memory Operations
# -----------------------


@builtin
def atomic_load(pointer, sem=None, _builder=None):
    """Performs an atomic load at the memory location specified by :code:`pointer`.

    Memory semantics can be "acquire", or "relaxed". See :func:`atomic_store`
    for details.

    :param pointer: The memory locations to load.
    :type pointer: Scalar of dtype=triton.PointerDType
    :param sem: Memory consistency semantics (default: "acquire")
    :type sem: str

    """
    pointer = _to_tensor(pointer, _builder)
    sem = _constexpr_to_value(sem)
    return semantic.atomic_load(pointer, sem, _builder)


@builtin
def atomic_store(pointer, value, sem=None, _builder=None):
    """Performs an atomic store at the memory location specified by :code:`pointer`.

    Memory semantics can be "release", or "relaxed":
    * "release" means any reads and writes made by the current triton program id
      will be visible to any corresponding program that reads the value by
      :code:`tl.atomic_load`` with "acquire" semantics.
    * "relaxed" implies no additional memory consistency, and only the location
      being written will obey a causal order. e.g. previous writes made by this
      program may *NOT* be visible to a program reading the value written.

    For "relaxed", only the memory
    location being stored to is guarunteed to be consistent with a causal ordering
    of the program.

    :param pointer: The memory locations to load.
    :type pointer: Scalar of dtype=triton.PointerDType
    :param value: The value to store.
    :type value: Scalar of dtype=`pointer.dtype.element_ty`
    :param sem: Memory consistency semantics (default: "release")
    :type sem: str

    """
    pointer = _to_tensor(pointer, _builder)
    value = _to_tensor(value, _builder)
    sem = _constexpr_to_value(sem)
    return semantic.atomic_store(pointer, value, sem, _builder)


def _add_atomic_docstr(name: str) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to compare-and-swap.
    :type pointer: Block of dtype=triton.PointerDType
    :param cmp: The values expected to be found in the atomic object
    :type cmp: Block of dtype=`pointer.dtype.element_ty`
    :param val: The values to copy in case the expected value matches the contained value.
    :type val: Block of dtype=`pointer.dtype.element_ty`
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@builtin
@_add_atomic_docstr("compare-and-swap")
def atomic_cas(pointer, cmp, val, sem=None, _builder=None):
    cmp = _to_tensor(cmp, _builder)
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    return semantic.atomic_cas(pointer, cmp, val, sem, _builder)


@builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, sem=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    return semantic.atomic_xchg(pointer, val, mask, sem, _builder)


@builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, sem=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    return semantic.atomic_add(pointer, val, mask, sem, _builder)


@builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, sem=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    return semantic.atomic_max(pointer, val, mask, sem, _builder)


@builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, sem=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    return semantic.atomic_min(pointer, val, mask, sem, _builder)


@builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, sem=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    return semantic.atomic_and(pointer, val, mask, sem, _builder)


@builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, sem=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    return semantic.atomic_or(pointer, val, mask, sem, _builder)


@builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, sem=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    return semantic.atomic_xor(pointer, val, mask, sem, _builder)


# -----------------------
# Conditioning
# -----------------------

@builtin
def where(condition, x, y, _builder=None):
    """
    Returns a tensor of elements from either :code:`x` or :code:`y`, depending on :code:`condition`.

    Note that :code:`x` and :code:`y` are always evaluated regardless of the value of :code:`condition`.

    If you want to avoid unintended memory operations, use the :code:`mask` arguments in `triton.load` and `triton.store` instead.

    The shape of :code:`x` and :code:`y` are both broadcast to the shape of :code:`condition`.
    :code:`x` and :code:`y` must have the same data type.

    :param condition: When True (nonzero), yield x, otherwise yield y.
    :type condition: Block of triton.bool
    :param x: values selected at indices where condition is True.
    :param y: values selected at indices where condition is False.
    """
    condition = _to_tensor(condition, _builder)
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    return semantic.where(condition, x, y, _builder)


# -----------------------
# Math
# -----------------------

@builtin
def umulhi(x, y, _builder=None):
    """
    Returns the most significant 32 bits of the product of x and y.

    :param x: the input tensor
    :type x: int32
    :param y: the input tensor
    :type y: int32
    """
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    return semantic.umulhi(x, y, _builder)


@builtin
def fdiv(x, y, ieee_rounding=False, _builder=None):
    """
    Returns a floating-point resultant tensor of dividing x by y.

    :param x: the input numerator value.
    :param y: the input denominator value.
    :param ieee_rounding: To follow IEEE-754 floating point number
    rounding mechanism
    :type ieee_rounding: bool
    """
    ieee_rounding = _constexpr_to_value(ieee_rounding)
    return semantic.fdiv(x, y, ieee_rounding, _builder)


def _add_math_1arg_docstr(name: str) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Computes the element-wise {name} of :code:`x`.

    :param x: the input values
    :type x: Block
    """
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@builtin
@_add_math_1arg_docstr("exponential")
def exp(x, _builder=None):
    return semantic.exp(x, _builder)


@builtin
@_add_math_1arg_docstr("natural logarithm")
def log(x, _builder=None):
    return semantic.log(x, _builder)


@builtin
@_add_math_1arg_docstr("cosine")
def cos(x, _builder=None):
    return semantic.cos(x, _builder)


@builtin
@_add_math_1arg_docstr("sine")
def sin(x, _builder=None):
    return semantic.sin(x, _builder)


@builtin
@_add_math_1arg_docstr("square root")
def sqrt(x, _builder=None):
    return semantic.sqrt(x, _builder)


@builtin
@_add_math_1arg_docstr("absolute value")
def abs(x, _builder=None):
    return semantic.abs(x, _builder)


# -----------------------
# Reductions
# -----------------------

def _add_reduction_docstr(name: str, return_indices_arg: str = None, tie_break_arg: str = None) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :param axis: the dimension along which the reduction should be done"""
        if return_indices_arg is not None:
            docstr += f"""
    :param {return_indices_arg}: if true, return index corresponding to the {name} value"""
        if tie_break_arg is not None:
            docstr += f"""
    :param {tie_break_arg}: if true, return the left-most indices in case of ties for values that aren't NaN"""

        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@contextmanager
def _insertion_guard(builder):
    ip = builder.get_insertion_point()
    yield
    builder.restore_insertion_point(ip)


@builtin
def reduce(input, axis, combine_fn, _builder=None, _generator=None):
    """Applies the combine_fn to all elements in :code:`input` tensors along the provided :code:`axis`

    :param input: the input tensor, or tuple of tensors
    :param axis: the dimension along which the reduction should be done
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)

    """
    if isinstance(input, tensor):
        return reduce((input,), axis, combine_fn,
                      _builder=_builder, _generator=_generator)[0]

    def make_combine_region(reduce_op):
        in_scalar_tys = [t.type.scalar for t in input]
        prototype = function_type(in_scalar_tys, in_scalar_tys * 2)

        region = reduce_op.get_region(0)
        with _insertion_guard(_builder):
            param_types = [ty.to_ir(_builder) for ty in prototype.param_types]
            block = _builder.create_block_with_parent(region, param_types)
            args = [tensor(block.arg(i), ty)
                    for i, ty in enumerate(prototype.param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            _builder.create_reduce_ret(*handles)
    if axis is not None:
        axis = _constexpr_to_value(axis)
    return semantic.reduction(input, axis, make_combine_region, _builder)


@builtin
def _promote_reduction_input(t, _builder=None):
    scalar_ty = t.type.scalar

    # hardware doesn't support FMAX, FMIN, CMP for bfloat16
    if scalar_ty is bfloat16:
        return t.to(float32, _builder=_builder)

    return t


@builtin
def _reduce_with_indices(input, axis, combine_fn, _builder=None, _generator=None):
    axis = _constexpr_to_value(axis)
    n = input.shape[axis]
    index = arange(0, n, _builder=_builder)

    if len(input.shape) > 1:
        # Broadcast index across the non-reduced axes
        axes_to_expand = [constexpr(d) for d in range(len(input.shape))]
        del axes_to_expand[axis]
        index = expand_dims(index, axes_to_expand, _builder=_builder)
        index = broadcast_to(index, input.shape, _builder=_builder)

    rvalue, rindices = reduce((input, index), axis, combine_fn,
                              _builder=_builder, _generator=_generator)
    return rvalue, rindices


# -----------------------
# Scans
# -----------------------

def _add_scan_docstr(name: str, return_indices_arg: str = None, tie_break_arg: str = None) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :param axis: the dimension along which the scan should be done"""
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@builtin
def associative_scan(input, axis, combine_fn, _builder=None, _generator=None):
    """Applies the combine_fn to each elements with a carry in :code:`input` tensors along the provided :code:`axis` and update the carry

    :param input: the input tensor, or tuple of tensors
    :param axis: the dimension along which the reduction should be done
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)

    """
    if isinstance(input, tensor):
        return associative_scan((input,), axis, combine_fn,
                                _builder=_builder, _generator=_generator)[0]

    def make_combine_region(scan_op):
        in_scalar_tys = [t.type.scalar for t in input]
        prototype = function_type(in_scalar_tys, in_scalar_tys * 2)

        region = scan_op.get_region(0)
        with _insertion_guard(_builder):
            param_types = [ty.to_ir(_builder) for ty in prototype.param_types]
            block = _builder.create_block_with_parent(region, param_types)
            args = [tensor(block.arg(i), ty)
                    for i, ty in enumerate(prototype.param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            _builder.create_scan_ret(*handles)
    axis = _constexpr_to_value(axis)
    return semantic.associative_scan(input, axis, make_combine_region, _builder)

# -----------------------
# Compiler Hint Ops
# -----------------------


@builtin
def debug_barrier(_builder=None):
    '''
    Insert a barrier to synchronize all threads in a block.
    '''
    return semantic.debug_barrier(_builder)


@builtin
def multiple_of(input, values, _builder=None):
    """
    Let the compiler knows that the values in :code:`input` are all multiples of :code:`value`.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return semantic.multiple_of(input, values)


@builtin
def max_contiguous(input, values, _builder=None):
    """
    Let the compiler knows that the `value` first values in :code:`input` are contiguous.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return semantic.max_contiguous(input, values)


@builtin
def max_constancy(input, values, _builder=None):
    """
    Let the compiler knows that the `value` first values in :code:`input` are constant.

    e.g. if :code:`values` is [4], then each group of 4 values in :code:`input` should all be equal,
    for example [0, 0, 0, 0, 1, 1, 1, 1].
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return semantic.max_constancy(input, values)
# -----------------------
# Debugging functions
# -----------------------


@builtin
def static_print(*values, sep: str = " ", end: str = "\n", file=None, flush=False, _builder=None):
    '''
    Print the values at compile time.  The parameters are the same as the builtin :code:`print`.

    NOTE: Calling the Python builtin :code:`print` is not the same as calling this, it instead maps to :code:`device_print`,
    which has special requirements for the arguments.

    .. highlight:: python
    .. code-block:: python

        tl.static_print(f"{BLOCK_SIZE=}")
    '''
    pass


@builtin
def static_assert(cond, msg="", _builder=None):
    '''
    Assert the condition at compile time.  Does not require that the :code:`TRITON_DEBUG` environment variable
    is set.

    .. highlight:: python
    .. code-block:: python

        tl.static_assert(BLOCK_SIZE == 1024)
    '''
    pass


@builtin
def device_print(prefix, *args, _builder=None):
    '''
    Print the values at runtime from the device.  String formatting does not work for runtime values, so you should
    provide the values you want to print as arguments.  The first value must be a string, all following values must
    be scalars or tensors.

    Calling the Python builtin :code:`print` is the same as calling this function, and the requirements for the arguments will match
    this function (not the normal requirements for :code:`print`).

    .. highlight:: python
    .. code-block:: python

        tl.device_print("pid", pid)
        print("pid", pid)

    :param prefix: a prefix to print before the values. This is required to be a string literal.
    :param args: the values to print. They can be any tensor or scalar.
    '''
    import string
    prefix = _constexpr_to_value(prefix)
    assert isinstance(prefix, str), f"{prefix} is not string"
    b_ascii = True
    for ch in prefix:
        if ch not in string.printable:
            b_ascii = False
            break
    assert b_ascii, f"{prefix} is not an ascii string"
    new_args = []
    for arg in args:
        new_args.append(_to_tensor(arg, _builder))
    return semantic.device_print(prefix, new_args, _builder)


@builtin
def device_assert(cond, msg="", _builder=None):
    '''
    Assert the condition at runtime from the device.  Requires that the environment variable :code:`TRITON_DEBUG`
    is set to a value besides :code:`0` in order for this to have any effect.

    Using the Python :code:`assert` statement is the same as calling this function, except that the second argument
    must be provided and must be a string, e.g. :code:`assert pid == 0, "pid != 0"`.  The environment variable must
    be set for this :code:`assert` statement to have any effect.

    .. highlight:: python
    .. code-block:: python

        tl.device_assert(pid == 0)
        assert pid == 0, f"pid != 0"

    :param cond: the condition to assert. This is required to be a boolean tensor.
    :param msg: the message to print if the assertion fails. This is required to be a string literal.
    '''
    msg = _constexpr_to_value(msg)
    import inspect
    frame = inspect.currentframe()
    module = inspect.getmodule(frame)
    # The triton function module doesn't have the name attribute.
    # We use this trick to find the caller.
    while hasattr(module, "__name__"):
        frame = frame.f_back
        module = inspect.getmodule(frame)
    lineno = 0
    func_name = 'unknown'
    file_name = 'unknown'
    if frame is not None:
        func_name = frame.f_code.co_name
        file_name = frame.f_back.f_code.co_filename
        # TODO: The line number currently indicates the line
        # where the triton function is called but not where the
        # device_assert is called. Need to enhance this.
        lineno = frame.f_back.f_lineno
    return semantic.device_assert(_to_tensor(cond, _builder), msg, file_name, func_name, lineno, _builder)


@builtin
def inline_asm_elementwise(asm: str, constraints: str, args: list, dtype, is_pure: bool, pack: int, _builder=None):
    '''
        Execute the inline assembly to a packed of elements of the tensor
        :param asm: assembly to be inlined, it has to match the target assembly format
        :param constraints: string representing the mapping of operands to register
        :param args: the arguments of the operation
        :param dtype: the element type of the returned variable
        :param is_pure: whether the operation is pure
        :param pack: the number of elements to be processed by one instance of inline assembly
        :param _builder: the builder
        :return: the return value of the function
    '''
    dispatch_args = args.copy()
    asm = _constexpr_to_value(asm)
    constraints = _constexpr_to_value(constraints)
    pack = _constexpr_to_value(pack)
    is_pure = _constexpr_to_value(is_pure)
    ret_shape = None
    arg_types = []
    res_ty = dtype
    for i in range(len(dispatch_args)):
        dispatch_args[i] = _to_tensor(dispatch_args[i], _builder)
        arg_types.append(dispatch_args[i].dtype)
    if len(arg_types) > 0:
        arg_types = tuple(arg_types)
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for i, item in enumerate(dispatch_args):
            _, broadcast_arg = semantic.binary_op_type_checking_impl(
                item, broadcast_arg, _builder, arithmetic_check=False)
        # Change the shape of each argument based on the broadcast shape
        for i in range(len(dispatch_args)):
            dispatch_args[i], _ = semantic.binary_op_type_checking_impl(
                dispatch_args[i], broadcast_arg, _builder, arithmetic_check=False)
        ret_shape = broadcast_arg.shape
        res_ty = block_type(dtype, ret_shape)
    call = _builder.create_inline_asm(asm, constraints, [t.handle for t in args], res_ty.to_ir(_builder), is_pure, pack)
    return tensor(call, res_ty)


# -----------------------
# Iterators
# -----------------------


class static_range:

    """
    Iterator that counts upward forever.

    .. highlight:: python
    .. code-block:: python

        @triton.jit
        def kernel(...):
            for i in tl.static_range(10):
                ...
    :note: This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
        :code:`triton.jit` functions. In addition, it also guides the compiler to unroll the loop aggressively.
    :param arg1: the start value.
    :param arg2: the end value.
    :param step: the step value.
    """

    def __init__(self, arg1, arg2=None, step=None):
        assert isinstance(arg1, constexpr)
        if step is None:
            self.step = constexpr(1)
        else:
            assert isinstance(step, constexpr)
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            assert isinstance(arg2, constexpr)
            self.start = arg1
            self.end = arg2

    def __iter__(self):
        raise RuntimeError("static_range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError("static_range can only be used in @triton.jit'd functions")


# -----------------------
# Extern functions
# -----------------------

def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, ret_shape: tuple, is_pure: bool, _builder=None):
    '''
        Dispatch a function to a library
        :param func: the function to dispatch
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param ret_shape: the shape of the return value
        :param _builder: the builder
        :return: the return value of the function
    '''
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        ret_type = arg_type_symbol_dict[arg_types][1]
        if ret_shape:
            ret_type = block_type(ret_type, ret_shape)
        return tensor(func(lib_name, lib_path, symbol, arg_list, ret_type.to_ir(_builder), is_pure), ret_type)


def extern_elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool, _builder=None):
    '''
        Dispatch an elementwise function to a library
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param is_pure: whether the function is pure
        :param _builder: the builder
        :return: the return value of the function
    '''
    dispatch_args = args.copy()
    all_scalar = True
    ret_shape = None
    arg_types = []
    for i in range(len(dispatch_args)):
        dispatch_args[i] = _to_tensor(dispatch_args[i], _builder)
        arg_types.append(dispatch_args[i].dtype)
        if dispatch_args[i].type.is_block():
            all_scalar = False
    if len(arg_types) > 0:
        arg_types = tuple(arg_types)
        arithmetic_check = True
        # If there's a type tuple that is not supported by the library, we will do arithmetic check
        if arg_types in arg_type_symbol_dict:
            arithmetic_check = False
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for i, item in enumerate(dispatch_args):
            _, broadcast_arg = semantic.binary_op_type_checking_impl(
                item, broadcast_arg, _builder, arithmetic_check=arithmetic_check)
        # Change the shape of each argument based on the broadcast shape
        for i in range(len(dispatch_args)):
            dispatch_args[i], _ = semantic.binary_op_type_checking_impl(
                dispatch_args[i], broadcast_arg, _builder, arithmetic_check=arithmetic_check)
        if not all_scalar:
            ret_shape = broadcast_arg.shape
    func = getattr(_builder, "create_extern_elementwise")
    return dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, ret_shape, is_pure, _builder)


def binary_op_type_legalization(lhs, rhs, builder):
    '''
        Convert both operands to a single common type
        :param lhs: the left operand
        :param rhs: the right operand
        :param builder: the builder
    '''
    return semantic.binary_op_type_checking_impl(lhs, rhs, builder)


def extern(fn):
    """A decorator for external functions."""
    return builtin(fn)
