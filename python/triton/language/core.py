from __future__ import annotations

from warnings import warn
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
import typing
from typing import Union, Callable, List, Sequence, TypeVar, Optional
import builtins
from ..runtime.jit import jit
import inspect
import os

from .._C.libtriton import ir
from . import semantic

T = TypeVar('T')

TRITON_MAX_TENSOR_NUMEL = 1048576

TRITON_BUILTIN = "__triton_builtin__"

PropagateNan = ir.PROPAGATE_NAN


def builtin(fn: T) -> T:
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)

    return wrapper


def _tensor_member_fn(fn: T) -> T:
    """Decorator that adds this free function as a member fn on class tensor.

    When called as a member function on class tensor, the first argument to `fn`
    is `self`, i.e. the tensor object.

    If there are multiple decorators on a function, you probably want this one
    to be the highest one (i.e. furthest from the function's `def`), so it's
    applied last.

    Unfortunately you still need to add a type stub to the body of class tensor
    in order for pytype to know about it.
    """
    assert callable(fn)
    orig_sig = inspect.signature(fn)
    # Does fn take args other than _builder, _generator, and the tensor itself?
    has_args = len(orig_sig.parameters.keys() - {"_builder", "_generator"}) > 1

    if not fn.__doc__:
        fn.__doc__ = ""
    fn.__doc__ += f"""
    This function can also be called as a member function on :py:class:`tensor`,
    as :code:`x.{fn.__name__}({"..." if has_args else ""})` instead of
    :code:`{fn.__name__}(x{", ..." if has_args else ""})`.
    """

    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    # Match the signature of `fn`, but change the first arg to `self` so the
    # docs are a little less weird.
    new_params = list(orig_sig.parameters.values())
    new_params[0] = new_params[0].replace(name='self')
    new_sig = orig_sig.replace(parameters=new_params)
    wrapper.__signature__ = new_sig
    wrapper.__doc__ = f"Forwards to :py:func:`{fn.__name__}` free function"
    # If fn is a builtin, mark the wrapper as a builtin too.
    if is_builtin(fn):
        setattr(wrapper, TRITON_BUILTIN, True)

    setattr(tensor, fn.__name__, wrapper)
    return fn


def _unwrap_iterable(x):
    """Returns x[0] if x has one element and x[0] is iterable."""
    if len(x) == 1:
        # Determine whether x[0] is iterable.
        #
        # You might want to use collections.abc.Iterable instead of this
        # try/except block.  Unfortunately, this doesn't work with constexpr.
        #
        # The problem is that abc.Iterable checks for __iter__ on the *class*.
        # But we want constexpr to expose an __iter__ method if and only if the
        # wrapped *object* (i.e. self.value) is iterable.  Therefore there's no
        # right answer for whether the class constexpr defines __iter__, and
        # abc.Iterable doesn't work (at least not without some metaclass magic).
        try:
            iter(x[0])
            return x[0]
        except TypeError:
            pass

    return x


def is_builtin(fn) -> bool:
    """Is this a registered triton builtin function?"""
    return getattr(fn, TRITON_BUILTIN, False)


@builtin
def to_tensor(x, _builder=None):
    return _to_tensor(x, _builder)


def _to_tensor(x, builder, check_type: bool = True):
    if isinstance(x, bool):
        return tensor(builder.get_int1(x), int1)
    # Note: compile-time const integers are represented by unsigned values
    elif isinstance(x, int):
        if -2**31 <= x < 2**31:
            return tensor(builder.get_int32(x), int32)
        elif 2**31 <= x < 2**32:
            return tensor(builder.get_uint32(x), uint32)
        elif -2**63 <= x < 2**63:
            return tensor(builder.get_int64(x), int64)
        elif 2**63 <= x < 2**64:
            return tensor(builder.get_uint64(x), uint64)
        else:
            raise ValueError(f'Nonrepresentable integer {x}.')
    elif isinstance(x, float):
        min_float32 = 2**-126 * 2**-23
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
    if check_type:
        raise TypeError(f"cannot convert {x} of type {type(x)} to tensor")
    return x


# -----------------------
# constexpr
# -----------------------


class const:
    """
    This class is used as a type annotation to mark pointers to constant data.
    The `store` function cannot be called with a pointer to const. Constness
    is part of the pointer type and the usual Triton type consistency rules
    apply. For example you cannot have a function that returns constant pointer
    in one return statement and non-constant pointer in another.
    """
    pass


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

    # In interpreter mode, constant values are not wrapped in constexpr,
    # and therefore do not have a .value attribute.
    # As a result, from here and below, we need to call the _constexpr_to_value
    # function to obtain either constexpr.value or the value itself.
    def __add__(self, other):
        return constexpr(self.value + _constexpr_to_value(other))

    def __radd__(self, other):
        return constexpr(_constexpr_to_value(other) + self.value)

    def __sub__(self, other):
        return constexpr(self.value - _constexpr_to_value(other))

    def __rsub__(self, other):
        return constexpr(_constexpr_to_value(other) - self.value)

    def __mul__(self, other):
        return constexpr(self.value * _constexpr_to_value(other))

    def __mod__(self, other):
        return constexpr(self.value % _constexpr_to_value(other))

    def __rmul__(self, other):
        return constexpr(_constexpr_to_value(other) * self.value)

    def __truediv__(self, other):
        return constexpr(self.value / _constexpr_to_value(other))

    def __rtruediv__(self, other):
        return constexpr(_constexpr_to_value(other) / self.value)

    def __floordiv__(self, other):
        return constexpr(self.value // _constexpr_to_value(other))

    def __rfloordiv__(self, other):
        return constexpr(_constexpr_to_value(other) // self.value)

    def __gt__(self, other):
        return constexpr(self.value > _constexpr_to_value(other))

    def __rgt__(self, other):
        return constexpr(_constexpr_to_value(other) > self.value)

    def __ge__(self, other):
        return constexpr(self.value >= _constexpr_to_value(other))

    def __rge__(self, other):
        return constexpr(_constexpr_to_value(other) >= self.value)

    def __lt__(self, other):
        return constexpr(self.value < _constexpr_to_value(other))

    def __rlt__(self, other):
        return constexpr(_constexpr_to_value(other) < self.value)

    def __le__(self, other):
        return constexpr(self.value <= _constexpr_to_value(other))

    def __rle__(self, other):
        return constexpr(_constexpr_to_value(other) <= self.value)

    def __eq__(self, other):
        return constexpr(self.value == _constexpr_to_value(other))

    def __ne__(self, other):
        return constexpr(self.value != _constexpr_to_value(other))

    def __bool__(self):
        return bool(self.value)

    def __neg__(self):
        return constexpr(-self.value)

    def __and__(self, other):
        return constexpr(self.value & _constexpr_to_value(other))

    def logical_and(self, other):
        return constexpr(self.value and _constexpr_to_value(other))

    def __or__(self, other):
        return constexpr(self.value | _constexpr_to_value(other))

    def __xor__(self, other):
        return constexpr(self.value ^ _constexpr_to_value(other))

    def logical_or(self, other):
        return constexpr(self.value or _constexpr_to_value(other))

    def __pos__(self):
        return constexpr(+self.value)

    def __invert__(self):
        return constexpr(~self.value)

    def __pow__(self, other):
        return constexpr(self.value**_constexpr_to_value(other))

    def __rpow__(self, other):
        return constexpr(_constexpr_to_value(other)**self.value)

    def __rshift__(self, other):
        return constexpr(self.value >> _constexpr_to_value(other))

    def __lshift__(self, other):
        return constexpr(self.value << _constexpr_to_value(other))

    def __not__(self):
        return constexpr(not self.value)

    def __iter__(self):
        return iter(self.value)

    def __call__(self, *args, **kwds):
        return self.value(*args, **kwds)


CONSTEXPR_0 = constexpr(0)


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, constexpr) else o


def check_bit_width(value, shift_value):
    if isinstance(value, tensor) and isinstance(shift_value, constexpr):
        bitwidth = value.type.scalar.primitive_bitwidth
        if shift_value.value >= bitwidth:
            warn(
                f"Value {shift_value.value} exceeds the maximum bitwidth ({bitwidth}) for type '{value.dtype}'. This may result in undefined behavior."
            )


# -----------------------
# dtype
# -----------------------


class dtype:
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['int1', 'uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp8e4b15', 'fp8e4nv', 'fp8e4b8', 'fp8e5', 'fp8e5b16', 'fp16', 'bf16', 'fp32', 'fp64']
    STANDARD_FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']
    OTHER_TYPES = ['void']

    class SIGNEDNESS(Enum):
        SIGNED = 0
        UNSIGNED = 1

    def __init__(self, name):
        name = _unwrap_if_constexpr(name)
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
            elif name == 'fp8e4nv':
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
                self.exponent_bias = 7
            elif name == 'fp8e4b8':
                self.fp_mantissa_width = 3
                self.primitive_bitwidth = 8
                self.exponent_bias = 8
            elif name == 'fp8e5':
                self.fp_mantissa_width = 2
                self.primitive_bitwidth = 8
                self.exponent_bias = 15
            elif name == 'fp8e5b16':
                self.fp_mantissa_width = 2
                self.primitive_bitwidth = 8
                self.exponent_bias = 16
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

    def is_fp8e4b8(self):
        return self.name == 'fp8e4b8'

    def is_fp8e4b15(self):
        return self.name == 'fp8e4b15'

    def is_fp8e5(self):
        return self.name == 'fp8e5'

    def is_fp8e5b16(self):
        return self.name == 'fp8e5b16'

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

    @staticmethod
    def is_const():
        return False

    def __eq__(self, other: dtype):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other: dtype):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, ))

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
        elif self.name == 'fp8e5b16':
            return builder.get_fp8e5b16_ty()
        elif self.name == 'fp8e4nv':
            return builder.get_fp8e4nv_ty()
        elif self.name == 'fp8e4b8':
            return builder.get_fp8e4b8_ty()
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

    def __str__(self):
        return self.name

    def codegen_name(self):
        if self.name.startswith("fp"):
            return "float" + self.name[2:]
        elif self.name.startswith("bf"):
            return "bfloat" + self.name[2:]
        else:
            return self.name

    @property
    def cache_key_part(self) -> str:
        """See cache_key_part() in triton.cc."""
        return self.name

    def __repr__(self):
        """Output of repr needs to be an evaluatable expression"""
        return f'triton.language.{self.codegen_name()}'


# Some functions have a param named `dtype`, which shadows the `dtype` class.
# We can't change the param name because it is part of function's public API.
# Declare an alias so those functions can still reference the dtype class.
_DtypeClass = dtype


class pointer_type(dtype):

    def __init__(self, element_ty: dtype, address_space: int = 1):
        element_ty = _unwrap_if_constexpr(element_ty)
        if not isinstance(element_ty, dtype):
            raise TypeError(f'element_ty has type `{type(element_ty).__name__}`; expected `dtype`.')
        self.element_ty = element_ty
        self.address_space = address_space

        self.name = f'pointer<{element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.pointer_type:
        return builder.get_ptr_ty(self.element_ty.to_ir(builder), 1)

    def __str__(self):
        return self.name

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


class const_pointer_type(pointer_type):

    def __init__(self, element_ty: dtype, address_space: int = 1):
        super().__init__(element_ty, address_space)

    def __str__(self):
        return f'const_pointer<{self.element_ty}>'

    def is_const(self):
        return True

    def __eq__(self, other) -> bool:
        if not isinstance(other, const_pointer_type):
            return False
        return self.element_ty == other.element_ty and self.address_space == other.address_space


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

        self.name = f'<{self.shape}, {self.element_ty}>'

    def to_ir(self, builder: ir.builder) -> ir.block_type:
        return builder.get_block_ty(self.element_ty.to_ir(builder), self.shape)

    def __str__(self):
        return self.name

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
float8e5b16 = dtype('fp8e5b16')
float8e4nv = dtype('fp8e4nv')
float8e4b8 = dtype('fp8e4b8')
float8e4b15 = dtype('fp8e4b15')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')
# pointer types
pi32_t = pointer_type(int32)


def get_int_dtype(bitwidth: int, signed: bool) -> dtype:
    if bitwidth == 1:
        return int1
    elif bitwidth == 8 and signed:
        return int8
    elif bitwidth == 8 and not signed:
        return uint8
    elif bitwidth == 16 and signed:
        return int16
    elif bitwidth == 16 and not signed:
        return uint16
    elif bitwidth == 32 and signed:
        return int32
    elif bitwidth == 32 and not signed:
        return uint32
    elif bitwidth == 64 and signed:
        return int64
    elif bitwidth == 64 and not signed:
        return uint64
    else:
        raise ValueError(f'Unsupported bitwidth {bitwidth} and signedness {signed}')


# -----------------------
# tensor
# -----------------------


class tensor:
    """Represents an N-dimensional array of values or pointers.

    :code:`tensor` is the fundamental data structure in Triton programs.  Most
    functions in :py:mod:`triton.language` operate on and return tensors.

    Most of the named member functions here are duplicates of the free functions
    in :code:`triton.language`.  For example, :code:`triton.language.sqrt(x)` is
    equivalent to :code:`x.sqrt()`.

    :code:`tensor` also defines most of the magic/dunder methods, so you can
    write :code:`x+y`, :code:`x << 2`, etc.

    .. rubric:: Constructors
    ..
       For some reason Sphinx includes __init__ before printing the full table
       of methods.  Not what I want, but I can't figure out how to fix it.  Give
       it its own section so it looks intentional. :)
    """

    def __init__(self, handle, type: dtype):
        """Not called by user code."""
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
        check_bit_width(self, other)
        other = _to_tensor(other, _builder)
        return semantic.shl(self, other, _builder)

    @builtin
    def __rlshift__(self, other, _builder=None):
        check_bit_width(other, self)
        other = _to_tensor(other, _builder)
        return semantic.shl(other, self, _builder)

    @builtin
    def __rshift__(self, other, _builder=None):
        check_bit_width(self, other)
        other = _to_tensor(other, _builder)
        if self.dtype.is_int_signed():
            return semantic.ashr(self, other, _builder)
        else:
            return semantic.lshr(self, other, _builder)

    @builtin
    def __rrshift__(self, other, _builder=None):
        check_bit_width(other, self)
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
    def __req__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.equal(other, self, _builder)

    @builtin
    def __ne__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.not_equal(self, other, _builder)

    @builtin
    def __rne__(self, other, _builder=None):
        other = _to_tensor(other, _builder)
        return semantic.not_equal(other, self, _builder)

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
        if isinstance(slices, (slice, constexpr)) or slices is None:
            slices = [slices]
        ret = self
        for dim, sl in enumerate(slices):
            if sl is None or isinstance(sl, constexpr) and sl.value is None:
                ret = semantic.expand_dims(ret, dim, _builder)
            elif isinstance(sl, slice) and sl.start is None and sl.stop is None and sl.step is None:
                pass
            else:
                raise ValueError(f"unsupported tensor index: {sl}")
        return ret

    @property
    def T(self):
        """Transposes a 2D tensor."""
        assert False, "Transposition must be created by the AST Visitor"

    @builtin
    def to(self, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _builder=None):
        """
        Alias for :py:func:`tensor.cast`.
        """
        # Triton doesn't like core functions calling other core functions, so we
        # just copy-paste the implementation of cast here.  It's not too bad.
        dtype = _unwrap_if_constexpr(dtype)
        bitcast = _unwrap_if_constexpr(bitcast)
        if bitcast:
            return semantic.bitcast(self, dtype, _builder)
        return semantic.cast(self, dtype, _builder, fp_downcast_rounding)

    # Type stubs for functions added by the _tensor_member_fn decorator.
    # (Unfortunately these can't be created automatically.)
    #
    # We couldn't write these definitions out even if we wanted to, because some
    # of these functions are defined in standard.py.
    def broadcast_to(self, *shape) -> tensor:
        ...

    def trans(self, *dims) -> tensor:
        ...

    def permute(self, *dims) -> tensor:
        ...

    def split(self) -> tuple[tensor, tensor]:
        ...

    def view(self, *shape) -> tensor:
        ...

    def reshape(self, *shape) -> tensor:
        ...

    def expand_dims(self, axis) -> tensor:
        ...

    def cast(self, dtype, fp_downcast_rounding=None, bitcast=False) -> tensor:
        ...

    def store(self, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="") -> tensor:
        ...

    def advance(self, offsets) -> tensor:
        ...

    def atomic_cas(self, cmp, val, sem=None, scope=None) -> tensor:
        ...

    def atomic_xchg(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_add(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_max(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_min(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_and(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_or(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def atomic_xor(self, val, mask=None, sem=None, scope=None) -> tensor:
        ...

    def exp(self) -> tensor:
        ...

    def log(self) -> tensor:
        ...

    def cos(self) -> tensor:
        ...

    def sin(self) -> tensor:
        ...

    def sqrt(self) -> tensor:
        ...

    def rsqrt(self) -> tensor:
        ...

    def abs(self) -> tensor:
        ...

    def reduce(self, axis, combine_fn, keep_dims=False) -> tensor:
        ...

    def associative_scan(self, axis, combine_fn, reverse=False) -> tensor:
        ...

    def histogram(self, num_bins) -> tensor:
        ...

    def cdiv(self, div) -> tensor:
        ...

    def sigmoid(self) -> tensor:
        ...

    def softmax(self, ieee_rounding=False) -> tensor:
        ...

    def ravel(self) -> tensor:
        ...

    def max(self, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def argmax(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def min(self, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def argmin(self, axis, tie_break_left=True, keep_dims=False) -> tensor:
        ...

    def sum(self, axis=None, keep_dims=False) -> tensor:
        ...

    def xor_sum(self, axis=None, keep_dims=False) -> tensor:
        ...

    def cumsum(self, axis=0, reverse=False) -> tensor:
        ...

    def cumprod(self, axis=0, reverse=False) -> tensor:
        ...

    def sort(self, dim: constexpr = None, descending: constexpr = CONSTEXPR_0) -> tensor:
        ...

    def flip(self, dim=None) -> tensor:
        ...


def get_bool_env_var(var_name):
    v = os.getenv(var_name, "0")
    return v == "1" or v == "true" or v == "on"


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

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
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

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    axis = _constexpr_to_value(axis)
    return semantic.num_programs(axis, _builder)


# -----------------------
# Block Initialization
# -----------------------


@builtin
def arange(start, end, _builder=None):
    start = _constexpr_to_value(start)
    end = _constexpr_to_value(end)
    return semantic.arange(start, end, _builder)


arange.__doc__ = f"""
    Returns contiguous values within the half-open interval :code:`[start,
    end)`.  :code:`end - start` must be less than or equal to
    :code:`TRITON_MAX_TENSOR_NUMEL = {TRITON_MAX_TENSOR_NUMEL}`

    :param start: Start of the interval. Must be a power of two.
    :type start: int32
    :param end: End of the interval. Must be a power of two greater than
        :code:`start`.
    :type end: int32
"""


def _shape_check_impl(shape):
    shape = _constexpr_to_value(shape)
    for i, d in enumerate(shape):
        if isinstance(d, int):
            d = constexpr(d)
        if not isinstance(d, constexpr):
            raise TypeError(f"Shape element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
        if d.value & (d.value - 1) != 0:
            raise ValueError(f"Shape element {i} must be a power of 2")
    return [_constexpr_to_value(x) for x in shape]


@builtin
def full(shape, value, dtype, _builder=None):
    """
    Returns a tensor filled with the scalar value for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param value: A scalar value to fill the array with
    :type value: scalar
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


@_tensor_member_fn
@builtin
def broadcast_to(input, *shape, _builder=None):
    """
    Tries to broadcast the given tensor to a new :code:`shape`.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.
    :type shape:

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        broadcast_to(x, (32, 32))
        broadcast_to(x, 32, 32)
    """
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return semantic.broadcast_impl_shape(input, shape, _builder)


@_tensor_member_fn
@builtin
def trans(input: tensor, *dims, _builder=None):
    """
    Permutes the dimensions of a tensor.

    If the parameter :code:`dims` is not specified, the function defaults to a (1,0) permutation,
    effectively transposing a 2D tensor.

    :param input: The input tensor.
    :param dims: The desired ordering of dimensions.  For example,
        :code:`(2, 1, 0)` reverses the order dims in a a 3D tensor.

    :code:`dims` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        trans(x, (2, 1, 0))
        trans(x, 2, 1, 0)

    :py:func:`permute` is equivalent to this function, except it doesn't
    have the special case when no permutation is specified.
    """
    if not dims:
        dims = (1, 0)
    return semantic.permute(input, dims, _builder)


@_tensor_member_fn
@builtin
def permute(input, *dims, _builder=None):
    """
    Permutes the dimensions of a tensor.

    :param input: The input tensor.
    :type input: Block
    :param dims: The desired ordering of dimensions.  For example,
        :code:`(2, 1, 0)` reverses the order dims in a a 3D tensor.

    :code:`dims` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        permute(x, (2, 1, 0))
        permute(x, 2, 1, 0)

    :py:func:`trans` is equivalent to this function, except when
    :code:`dims` is empty, it tries to do a (1,0) permutation.
    """
    dims = _unwrap_iterable(dims)
    return semantic.permute(input, dims, _builder)


@builtin
def cat(input, other, can_reorder=False, _builder=None):
    """
    Concatenate the given blocks

    :param input: The first input tensor.
    :type input: Tensor
    :param other: The second input tensor.
    :type other: Tensor
    :param reorder: Compiler hint. If true, the compiler is
        allowed to reorder elements while concatenating inputs.  Only use if the
        order does not matter (e.g., result is only used in reduction ops).
        Current implementation of `cat` supports only can_reorder=True.
    """
    return semantic.cat(input, other, can_reorder, _builder)


@builtin
def join(a, b, _builder=None):
    """
    Join the given tensors in a new, minor dimension.

    For example, given two tensors of shape (4,8), produces a new tensor of
    shape (4,8,2).  Given two scalars, returns a tensor of shape (2).

    The two inputs are broadcasted to be the same shape.

    If you want to join more than two elements, you can use multiple calls to
    this function.  This reflects the constraint in Triton that tensors must
    have power-of-two sizes.

    join is the inverse of split.

    :param a: The first input tensor.
    :type a: Tensor
    :param b: The second input tensor.
    :type b: Tensor
    """
    return semantic.join(a, b, _builder)


@jit
def _take_first(a, b):
    return a


@_tensor_member_fn
@builtin
def split(a, _builder=None, _generator=None) -> tuple[tensor, tensor]:
    """
    Split a tensor in two along its last dim, which must have size 2.

    For example, given a tensor of shape (4,8,2), produces two tensors of shape
    (4,8).  Given a tensor of shape (2), returns two scalars.

    If you want to split into more than two pieces, you can use multiple calls
    to this function (probably plus calling reshape).  This reflects the
    constraint in Triton that tensors must have power-of-two sizes.

    split is the inverse of join.

    :param a: The tensor to split.
    :type a: Tensor
    """
    # If len(a.shape) == 1, i.e. a.shape == [2], we should return two scalars.
    # But semantic.split can only handle returning tensors.  Work around this by
    # expanding the input to shape [1,2] and then reducing the result.
    was_rank_1 = len(a.shape) == 1
    if was_rank_1:
        a = semantic.expand_dims(a, 0, _builder)

    out_lhs, out_rhs = semantic.split(a, _builder)

    if was_rank_1:
        # Currently `reduce` is the best way to convert a tensor of shape [1] to a scalar.
        out_lhs = typing.cast(tensor, reduce(out_lhs, None, _take_first, _builder=_builder, _generator=_generator))
        out_rhs = typing.cast(tensor, reduce(out_rhs, None, _take_first, _builder=_builder, _generator=_generator))

    return out_lhs, out_rhs


@_tensor_member_fn
@builtin
def view(input, *shape, _builder=None):
    """
    Returns a tensor with the same elements as `input` but a different shape.
    The order of the elements may not be preserved.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        view(x, (32, 32))
        view(x, 32, 32)
    """
    warn("view is deprecated, please use reshape with can_reorder being true.")
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return semantic.reshape(input, shape, can_reorder=True, builder=_builder)


@_tensor_member_fn
@builtin
def reshape(input, *shape, can_reorder=False, _builder=None):
    """
    Returns a tensor with the same number of elements as input but with the
    provided shape.

    :param input: The input tensor.
    :type input: Block
    :param shape: The new shape.

    :code:`shape ` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        reshape(x, (32, 32))
        reshape(x, 32, 32)
    """
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return semantic.reshape(input, shape, can_reorder, _builder)


def _wrap_axis(axis, ndim):
    if not (-ndim <= axis < ndim):
        raise ValueError(f"invalid axis {axis}. Expected {-ndim} <= axis < {ndim}")

    return axis if axis >= 0 else axis + ndim


@_tensor_member_fn
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
    input = _to_tensor(input, _builder)
    axis = _constexpr_to_value(axis)
    axes = list(axis) if isinstance(axis, Sequence) else [axis]
    new_ndim = len(input.shape) + len(axes)
    axes = [_wrap_axis(_constexpr_to_value(d), new_ndim) for d in axes]

    if len(set(axes)) != len(axes):
        raise ValueError(f"expand_dims received duplicate axes, normalized axes = {axes}")

    ret = input
    for a in sorted(axes):
        ret = semantic.expand_dims(ret, a, _builder)
    return ret


@_tensor_member_fn
@builtin
def cast(input, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _builder=None):
    """
    Casts a tensor to the given :code:`dtype`.

    :param dtype: The target data type.
    :param fp_downcast_rounding: The rounding mode for downcasting
        floating-point values.  This parameter is only used when self is a
        floating-point tensor and dtype is a floating-point type with a
        smaller bitwidth. Supported values are :code:`"rtne"` (round to
        nearest, ties to even) and :code:`"rtz"` (round towards zero).
    :param bitcast: If true, the tensor is bitcasted to the given
        :code:`dtype`, instead of being numerically casted.
    """
    input = _to_tensor(input, _builder)
    if isinstance(bitcast, constexpr):
        bitcast = bitcast.value
    if bitcast:
        return semantic.bitcast(input, dtype, _builder)
    return semantic.cast(input, dtype, _builder, fp_downcast_rounding)


# -----------------------
# Linear Algebra
# -----------------------


@builtin
def dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=float32,
        _builder=None):
    """
    Returns the matrix product of two blocks.

    The two blocks must both be two-dimensional or three-dimensional and have compatible inner dimensions.
    For three-dimensional blocks, `tl.dot` performs the batched matrix product,
    where the first dimension of each block represents the batch dimension.

    :param input: The first tensor to be multiplied.
    :type input: 2D or 3D tensor of scalar-type in {:code:`int8`, :code: `float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D or 3D tensor of scalar-type in {:code:`int8`, :code: `float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param acc: The accumulator tensor. If not None, the result is added to this tensor.
    :type acc: 2D or 3D tensor of scalar-type in {:code:`float16`, :code:`float32`, :code:`int32`}
    :param input_precision: How to exercise the Tensor Cores for f32 x f32. If
      the device does not have Tensor Cores or the inputs are not of dtype f32,
      this option is ignored. For devices that do have tensor cores, the
      default precision is tf32.
    :type input_precision: string. Available options for nvidia: :code:`"tf32"`, :code:`"tf32x3"`, :code:`"ieee"`. Default: :code:`"tf32"`. Avaliable options for amd: :code:`"ieee"`.
    :param allow_tf32: *Deprecated.* If true, input_precision is set to "tf32".
      Only one of :code:`input_precision` and :code:`allow_tf32` can be
      specified (i.e. at least one must be :code:`None`).
    """
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is None:
        supports_tf32 = _builder and "tf32" in _builder.options.allowed_dot_input_precisions
        default_precision = "tf32" if (supports_tf32 and (allow_tf32 or allow_tf32 is None)) else "ieee"
        input_precision = os.getenv("TRITON_F32_DEFAULT", default_precision)

    input_precision = _constexpr_to_value(input_precision)
    out_dtype = _constexpr_to_value(out_dtype)
    max_num_imprecise_acc = _constexpr_to_value(max_num_imprecise_acc)
    return semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype, _builder)


# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, _builder=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

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
    :type cache_modifier: str, optional, should be one of {"", "ca", "cg"}, where "ca" stands for
        cache at all levels and "cg" stands for cache at global level (cache in L2 and below, not L1), see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    # `mask` and `other` can be constexpr
    mask = _constexpr_to_value(mask)
    other = _constexpr_to_value(other)
    if mask is not None:
        mask = _to_tensor(mask, _builder)
    if other is not None:
        other = _to_tensor(other, _builder)
    padding_option = _constexpr_to_value(padding_option)
    cache_modifier = _constexpr_to_value(cache_modifier)
    eviction_policy = _constexpr_to_value(eviction_policy)
    volatile = _constexpr_to_value(volatile)
    return semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                         volatile, _builder)


@builtin
def _experimental_descriptor_load(desc_pointer, offsets, shape, dtype, _builder=None):
    """
    Experimental feature to access TMA descriptors loads. This is an escape hatch to easily exercise TTGIR operations.
    This will be removed in the future and shouldn't be used in production code.

    This loads a tensor of data based on the descriptor and offsets.
    """
    type = block_type(dtype, shape)
    return semantic.descriptor_load(desc_pointer, offsets, "", "", type, _builder)


@builtin
def _experimental_descriptor_store(desc_pointer, value, offsets, _builder=None):
    """
    Experimental feature to access TMA descriptors stores. This is an escape hatch to easily exercise TTGIR operations.
    This will be removed in the future and shouldn't be used in production code.

    This stores a tensor of data based on the descriptor and offsets.
    """
    return semantic.descriptor_store(desc_pointer, value, offsets, _builder)


@_tensor_member_fn
@builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="", _builder=None):
    """
    Store a tensor of data into memory locations defined by `pointer`.

        (1) If `pointer` is a single element pointer, a scalar is stored.  In
            this case:

            - `mask` must also be scalar, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional block is stored.  In this case:

            - `mask` is implicitly broadcast to `pointer.shape`, and
            - `boundary_check` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a block
            of data is stored.  In this case:

            - `mask` must be None, and
            - `boundary_check` can be specified to control the behavior of out-of-bound access.

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
    :type cache_modifier: str, optional, should be one of {"", ".wb", ".cg", ".cs", ".wt"}, where ".wb" stands for
        cache write-back all coherent levels, ".cg" stands for cache global, ".cs" stands for cache streaming, ".wt"
        stands for cache write-through, see `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional, should be one of {"", "evict_first", "evict_last"}
    """
    # `value` can be constexpr
    value = _to_tensor(value, _builder)
    mask = _constexpr_to_value(mask)
    if mask is not None:
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


@_tensor_member_fn
@builtin
def advance(base, offsets, _builder=None):
    """
    Advance a block pointer

    :param base: the block pointer to advance
    :param offsets: the offsets to advance, a tuple by dimension
    """
    return semantic.advance(base, offsets, _builder)


# -----------------------
# Atomic Memory Operations
# -----------------------


def _add_atomic_docstr(name: str, has_cmp: bool = False) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = f"""
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to operate on
    :type pointer: Block of dtype=triton.PointerDType"""
        if has_cmp:
            docstr += """
    :param cmp: The values expected to be found in the atomic object
    :type cmp: Block of dtype=pointer.dtype.element_ty"""
        docstr += """
    :param val: The values with which to perform the atomic operation
    :type val: Block of dtype=pointer.dtype.element_ty
    :param sem: Specifies the memory semantics for the operation. Acceptable values are "acquire",
        "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided,
        the function defaults to using "acq_rel" semantics.
    :type sem: str, optional
    :param scope: Defines the scope of threads that observe the synchronizing effect of the atomic operation.
        Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
    :type scope: str, optional
    """
        func.__doc__ = docstr
        return func

    return _decorator


@_tensor_member_fn
@builtin
@_add_atomic_docstr("compare-and-swap", has_cmp=True)
def atomic_cas(pointer, cmp, val, sem=None, scope=None, _builder=None):
    cmp = _to_tensor(cmp, _builder)
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    return semantic.atomic_cas(pointer, cmp, val, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_xchg(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_add(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_max(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_min(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_and(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_or(pointer, val, mask, sem, scope, _builder)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, sem=None, scope=None, _builder=None):
    val = _to_tensor(val, _builder)
    sem = _constexpr_to_value(sem)
    scope = _constexpr_to_value(scope)
    mask = _constexpr_to_value(mask)
    return semantic.atomic_xor(pointer, val, mask, sem, scope, _builder)


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
def minimum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _builder=None):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param x: the first input tensor
    :type x: Block
    :param y: the second input tensor
    :type y: Block
    :param propagate_nan: whether to propagate NaN values.
    :type propagate_nan: tl.PropagateNan

    .. seealso:: :class:`tl.PropagateNan`
    """
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    x = _promote_bfloat16_to_float32(x, _builder=_builder)
    y = _promote_bfloat16_to_float32(y, _builder=_builder)
    propagate_nan = _constexpr_to_value(propagate_nan)
    return semantic.minimum(x, y, propagate_nan, _builder)


@builtin
def maximum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _builder=None):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param x: the first input tensor
    :type x: Block
    :param y: the second input tensor
    :type y: Block
    :param propagate_nan: whether to propagate NaN values.
    :type propagate_nan: tl.PropagateNan

    .. seealso:: :class:`tl.PropagateNan`
    """
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    x = _promote_bfloat16_to_float32(x, _builder=_builder)
    y = _promote_bfloat16_to_float32(y, _builder=_builder)
    propagate_nan = _constexpr_to_value(propagate_nan)
    return semantic.maximum(x, y, propagate_nan, _builder)


@builtin
def clamp(x, min, max, propagate_nan: constexpr = PropagateNan.NONE, _builder=None):
    """
    Clamps the input tensor :code:`x` within the range [min, max].
    Behavior when :code:`min` > :code:`max` is undefined.

    :param x: the input tensor
    :type x: Block
    :param min: the lower bound for clamping
    :type min: Block
    :param max: the upper bound for clamping
    :type max: Block
    :param propagate_nan: whether to propagate NaN values. Applies only to the :code:`x` tensor.
        If either :code:`min` or :code:`max` is NaN, the result is undefined.
    :type propagate_nan: tl.PropagateNan

    .. seealso:: :class:`tl.PropagateNan`
    """
    x = _to_tensor(x, _builder)
    min = _to_tensor(min, _builder)
    max = _to_tensor(max, _builder)
    x = _promote_bfloat16_to_float32(x, _builder=_builder)
    min = _promote_bfloat16_to_float32(min, _builder=_builder)
    max = _promote_bfloat16_to_float32(max, _builder=_builder)

    propagate_nan = _constexpr_to_value(propagate_nan)

    return semantic.clamp(x, min, max, propagate_nan, _builder)


# -----------------------
# Reductions
# -----------------------


def _add_reduction_docstr(name: str, return_indices_arg: str = None, tie_break_arg: str = None) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done
    :type axis: int
    :param keep_dims: if true, keep the reduced dimensions with length 1
    :type keep_dims: bool"""
        if return_indices_arg is not None:
            docstr += f"""
    :param {return_indices_arg}: if true, return index corresponding to the {name} value
    :type {return_indices_arg}: bool"""
        if tie_break_arg is not None:
            docstr += f"""
    :param {tie_break_arg}: if true, in case of a tie (i.e., multiple elements have the same {name} value), return the left-most index for values that aren't NaN
    :type {tie_break_arg}: bool"""

        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@contextmanager
def _insertion_guard(builder):
    ip = builder.get_insertion_point()
    yield
    builder.restore_insertion_point(ip)


@_tensor_member_fn
@builtin
def reduce(input, axis, combine_fn, keep_dims=False, _builder=None, _generator=None):
    """Applies the combine_fn to all elements in :code:`input` tensors along the provided :code:`axis`

    :param input: the input tensor, or tuple of tensors
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done. If None, reduce all dimensions
    :type axis: int | None
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)
    :type combine_fn: Callable
    :param keep_dims: if true, keep the reduced dimensions with length 1
    :type keep_dims: bool

    """
    if isinstance(input, tensor):
        return reduce((input, ), axis, combine_fn, keep_dims=keep_dims, _builder=_builder, _generator=_generator)[0]

    def make_combine_region(reduce_op):
        in_scalar_tys = [t.type.scalar for t in input]
        prototype = function_type(in_scalar_tys, in_scalar_tys * 2)

        region = reduce_op.get_region(0)
        with _insertion_guard(_builder):
            param_types = [ty.to_ir(_builder) for ty in prototype.param_types]
            block = _builder.create_block_with_parent(region, param_types)
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(prototype.param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            _builder.create_reduce_ret(*handles)

    def expand_ndims(t, ndims):
        for _ in builtins.range(ndims):
            t = expand_dims(t, 0, _builder=_builder)
        return t

    axis = _constexpr_to_value(axis)
    keep_dims = _constexpr_to_value(keep_dims)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    ret = semantic.reduction(input, axis, make_combine_region, _builder)
    if keep_dims:
        if axis is not None:
            ret = tuple(expand_dims(t, axis, _builder=_builder) for t in ret)
        else:
            ret = tuple(expand_ndims(t, len(input[0].shape)) for t in ret)
    return ret


@builtin
def _promote_bfloat16_to_float32(t, _builder=None):
    scalar_ty = t.type.scalar

    # hardware doesn't support FMAX, FMIN, CMP for bfloat16
    if scalar_ty is bfloat16:
        return t.to(float32, _builder=_builder)
    return t


@builtin
def _reduce_with_indices(input, axis, combine_fn, keep_dims=False, _builder=None, _generator=None):
    axis = _constexpr_to_value(axis)
    n = input.shape[axis]
    index = arange(0, n, _builder=_builder)

    if len(input.shape) > 1:
        # Broadcast index across the non-reduced axes
        axes_to_expand = [constexpr(d) for d in builtins.range(len(input.shape))]
        del axes_to_expand[axis]
        index = expand_dims(index, axes_to_expand, _builder=_builder)
        index = broadcast_to(index, input.shape, _builder=_builder)

    rvalue, rindices = reduce((input, index), axis, combine_fn, keep_dims=keep_dims, _builder=_builder,
                              _generator=_generator)
    return rvalue, rindices


# -----------------------
# Scans
# -----------------------


def _add_scan_docstr(name: str) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :type input: Tensor
    :param axis: the dimension along which the scan should be done
    :type axis: int"""
        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@_tensor_member_fn
@builtin
def associative_scan(input, axis, combine_fn, reverse=False, _builder=None, _generator=None):
    """Applies the combine_fn to each elements with a carry in :code:`input` tensors along the provided :code:`axis` and update the carry

    :param input: the input tensor, or tuple of tensors
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done
    :type axis: int
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)
    :type combine_fn: Callable
    :param reverse: whether to apply the associative scan in the reverse direction along axis
    :type reverse: bool

    """
    if isinstance(input, tensor):
        return associative_scan((input, ), axis, combine_fn, reverse, _builder=_builder, _generator=_generator)[0]

    def make_combine_region(scan_op):
        in_scalar_tys = [t.type.scalar for t in input]
        prototype = function_type(in_scalar_tys, in_scalar_tys * 2)

        region = scan_op.get_region(0)
        with _insertion_guard(_builder):
            param_types = [ty.to_ir(_builder) for ty in prototype.param_types]
            block = _builder.create_block_with_parent(region, param_types)
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(prototype.param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            _builder.create_scan_ret(*handles)

    axis = _constexpr_to_value(axis)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    return semantic.associative_scan(input, axis, make_combine_region, reverse, _builder)


@_tensor_member_fn
@builtin
def histogram(input, num_bins, _builder=None, _generator=None):
    """computes an histogram based on input tensor with num_bins bins, the bins have a width of 1 and start at 0.

    :param input: the input tensor
    :type input: Tensor
    :param num_bins: number of histogram bins
    :type num_bins: int

    """
    num_bins = _constexpr_to_value(num_bins)
    return semantic.histogram(input, num_bins, _builder)


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
    Let the compiler know that the values in :code:`input` are all multiples of :code:`value`.
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
    Let the compiler know that the `value` first values in :code:`input` are contiguous.
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
    Let the compiler know that the `value` first values in :code:`input` are constant.

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

        tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
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
def device_print(prefix, *args, hex=False, _builder=None):
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

    On CUDA, printfs are streamed through a buffer of limited size (on one host,
    we measured the default as 6912 KiB, but this may not be consistent across
    GPUs and CUDA versions).  If you notice some printfs are being dropped, you
    can increase the buffer size by calling

    .. highlight:: python
    .. code-block:: python

        triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)

    CUDA may raise an error if you try to change this value after running a
    kernel that uses printfs.  The value set here may only affect the current
    device (so if you have multiple GPUs, you'd need to call it multiple times).

    :param prefix: a prefix to print before the values. This is required to be a string literal.
    :param args: the values to print. They can be any tensor or scalar.
    :param hex: print all values as hex instead of decimal
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
    return semantic.device_print(prefix, new_args, hex, _builder)


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
    if frame is not None and frame.f_back is not None:
        func_name = frame.f_code.co_name
        file_name = frame.f_back.f_code.co_filename
        # TODO: The line number currently indicates the line
        # where the triton function is called but not where the
        # device_assert is called. Need to enhance this.
        lineno = frame.f_back.f_lineno
    return semantic.device_assert(_to_tensor(cond, _builder), msg, file_name, func_name, lineno, _builder)


@builtin
def inline_asm_elementwise(asm: str, constraints: str, args: Sequence, dtype: Union[dtype, Sequence[dtype]],
                           is_pure: bool, pack: int, _builder=None):
    '''
        Execute inline assembly over a tensor.  Essentially, this is :code:`map`
        where the function is inline assembly.

        The input tensors :code:`args` are implicitly broadcasted to the same shape.

        :code:`dtype` can be a tuple of types, in which case the output is a
        tuple of tensors.

        Each invocation of the inline asm processes :code:`pack` elements at a
        time.  Exactly which set of inputs a block receives is unspecified.
        Input elements of size less than 4 bytes are packed into 4-byte
        registers.

        This op does not support empty :code:`dtype` -- the inline asm must
        return at least one tensor, even if you don't need it.  You can work
        around this by returning a dummy tensor of arbitrary type; it shouldn't
        cost you anything if you don't use it.

        Example using
        `PTX <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`_
        assembly:

        .. highlight:: python
        .. code-block:: python

            @triton.jit
            def kernel(A, B, C, D, BLOCK: tl.constexpr):
                a = tl.load(A + tl.arange(0, BLOCK)) # uint8 tensor
                b = tl.load(B + tl.arange(0, BLOCK)) # float32 tensor

                # For each (a,b) in zip(a,b), perform the following:
                # - Let ai be `a` converted to int32.
                # - Let af be `a` converted to float.
                # - Let m be the max of ai and b.
                # - Return ai and mi.
                # Do the above 4 elements at a time.
                (c, d) = tl.inline_asm_elementwise(
                    asm="""
                    {
                        // Unpack `a` into `ai`.
                        .reg .b8 tmp<4>;
                        mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
                        cvt.u32.u8 $0, tmp0;
                        cvt.u32.u8 $1, tmp1;
                        cvt.u32.u8 $2, tmp2;
                        cvt.u32.u8 $3, tmp3;
                    }
                    // Convert `ai` to float.
                    cvt.rn.f32.s32 $4, $0;
                    cvt.rn.f32.s32 $5, $1;
                    cvt.rn.f32.s32 $6, $2;
                    cvt.rn.f32.s32 $7, $3;
                    // Take max of `ai` and `b`.
                    max.f32 $4, $4, $9;
                    max.f32 $5, $5, $10;
                    max.f32 $6, $6, $11;
                    max.f32 $7, $7, $12;
                    """,
                    constraints=(
                        # 8 output registers, namely
                        #   $0=ai0, $1=ai1, $2=ai2, $3=ai3,
                        #   $4=m0,  $5=m1,  $6=m2,  $7=m3.
                        "=r,=r,=r,=r,=r,=r,=r,=r,"
                        # 5 input registers, namely
                        #   $8=ai,
                        #   $9=b0, $10=b1, $11=b2, $12=b3.
                        # The four elements from `a` are all packed into one register.
                        "r,r,r,r,r"),
                    args=[a, b],
                    dtype=(tl.int32, tl.float32),
                    is_pure=True,
                    pack=4,
                )
                tl.store(C + tl.arange(0, BLOCK), c)
                tl.store(D + tl.arange(0, BLOCK), d)

        :param asm: assembly to run.  Must match target's assembly format.
        :param constraints: asm constraints in
            `LLVM format <https://llvm.org/docs/LangRef.html#inline-asm-constraint-string>`_
        :param args: the input tensors, whose values are passed to the asm block
        :param dtype: the element type(s) of the returned tensor(s)
        :param is_pure: if true, the compiler assumes the asm block has no side-effects
        :param pack: the number of elements to be processed by one instance of inline assembly
        :param _builder: the builder
        :return: one tensor or a tuple of tensors of the given dtypes
    '''
    asm = _constexpr_to_value(asm)
    constraints = _constexpr_to_value(constraints)
    pack = _constexpr_to_value(pack)
    is_pure = _constexpr_to_value(is_pure)

    # Wrap `dtype` in a tuple if it's not already.
    try:
        iter(dtype)  # type: ignore
        has_multiple_outputs = True
    except TypeError:
        has_multiple_outputs = False
        dtype = (dtype, )  # type: ignore

    dtype = typing.cast(Sequence[_DtypeClass], dtype)

    res_tys = dtype
    if dispatch_args := [_to_tensor(arg, _builder) for arg in args]:
        bin_op_type_checking = partial(
            semantic.binary_op_type_checking_impl,
            builder=_builder,
            arithmetic_check=False,
            allow_lhs_ptr=True,
            allow_rhs_ptr=True,
        )
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for item in dispatch_args:
            _, broadcast_arg = bin_op_type_checking(item, broadcast_arg)
        if broadcast_arg.shape:
            # Change the shape of each argument based on the broadcast shape
            for i, item in enumerate(dispatch_args):
                dispatch_args[i], _ = bin_op_type_checking(item, broadcast_arg)
            res_tys = [block_type(dt, broadcast_arg.shape) for dt in dtype]
    handles = [t.handle for t in dispatch_args]
    call = _builder.create_inline_asm(asm, constraints, handles, [ty.to_ir(_builder) for ty in res_tys], is_pure, pack)

    if not has_multiple_outputs:
        return tensor(call.get_result(0), res_tys[0])
    return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(res_tys))


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


class range:
    """
    Iterator that counts upward forever.

    .. highlight:: python
    .. code-block:: python

        @triton.jit
        def kernel(...):
            for i in tl.range(10, num_stages=3):
                ...
    :note: This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
        :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param arg1: the start value.
    :param arg2: the end value.
    :param step: the step value.
    :param num_stages: pipeline the loop into this many stages (so there are
        :code:`num_stages` iterations of the loop in flight at once).

        Note this is subtly different than passing :code:`num_stages` as a
        kernel argument.  The kernel argument only pipelines loads that feed
        into :code:`dot` operations, while this attribute tries to pipeline most
        (though not all) loads in this loop.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None):
        if step is None:
            self.step = constexpr(1)
        else:
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2
        self.num_stages = num_stages

    def __iter__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")


# -----------------------
# Extern functions
# -----------------------


def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, ret_shape: tuple,
             is_pure: bool, _builder=None):
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


@builtin
def extern_elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool,
                       _builder=None):
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
    for i in builtins.range(len(dispatch_args)):
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
        for item in dispatch_args:
            _, broadcast_arg = semantic.binary_op_type_checking_impl(item, broadcast_arg, _builder,
                                                                     arithmetic_check=arithmetic_check)
        # Change the shape of each argument based on the broadcast shape
        for i in builtins.range(len(dispatch_args)):
            dispatch_args[i], _ = semantic.binary_op_type_checking_impl(dispatch_args[i], broadcast_arg, _builder,
                                                                        arithmetic_check=arithmetic_check)
        if not all_scalar:
            ret_shape = broadcast_arg.shape
    func = _builder.create_extern_elementwise
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
