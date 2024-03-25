from ..runtime.jit import jit
from . import core as tl
from . import math

N_ROUNDS_DEFAULT = 10  # Default number of rounds for philox

# -------------------
# randint
# -------------------


@jit
def philox_impl(c0, c1, c2, c3, k0, k1, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Run `n_rounds` rounds of Philox for state (c0, c1, c2, c3) and key (k0, k1).
    """
    if c0.dtype == tl.uint32:
        PHILOX_KEY_A: tl.constexpr = 0x9E3779B9
        PHILOX_KEY_B: tl.constexpr = 0xBB67AE85
        PHILOX_ROUND_A: tl.constexpr = 0xD2511F53
        PHILOX_ROUND_B: tl.constexpr = 0xCD9E8D57
    else:
        tl.static_assert(c0.dtype == tl.uint64, "dtype not supported in philox_impl")
        PHILOX_KEY_A: tl.constexpr = 0x9E3779B97F4A7C15
        PHILOX_KEY_B: tl.constexpr = 0xBB67AE8584CAA73B
        PHILOX_ROUND_A: tl.constexpr = 0xD2E7470EE14C6C93
        PHILOX_ROUND_B: tl.constexpr = 0xCA5A826395121157

    for _ in tl.static_range(n_rounds):
        # for _ in range(n_rounds):
        # update random state
        A = PHILOX_ROUND_A
        B = PHILOX_ROUND_B
        _c0, _c2 = c0, c2
        c0 = math.umulhi(B, _c2) ^ c1 ^ k0
        c2 = math.umulhi(A, _c0) ^ c3 ^ k1
        c1 = B * _c2
        c3 = A * _c0
        # raise key
        k0 = k0 + PHILOX_KEY_A
        k1 = k1 + PHILOX_KEY_B
    return c0, c1, c2, c3


@jit
def philox(seed, c0, c1, c2, c3, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    seed = tl.to_tensor(seed)
    c0 = tl.to_tensor(c0)
    c1 = tl.to_tensor(c1)
    c2 = tl.to_tensor(c2)
    c3 = tl.to_tensor(c3)
    seed = seed.to(tl.uint64)
    if tl.constexpr(c0.dtype.primitive_bitwidth) == 32:
        int_dtype = tl.uint32
        seed_hi = ((seed >> 32) & 0xffffffff).to(tl.uint32)
        seed_lo = (seed & 0xffffffff).to(tl.uint32)
    else:
        tl.static_assert(tl.constexpr(c0.dtype.primitive_bitwidth) == 64, "bitwidth not supported in philox")
        int_dtype = tl.uint64
        seed_hi = tl.full((1, ), 0, dtype=int_dtype)
        seed_lo = seed
    c0 = c0.to(int_dtype, bitcast=True)
    c1 = c1.to(int_dtype, bitcast=True)
    c2 = c2.to(int_dtype, bitcast=True)
    c3 = c3.to(int_dtype, bitcast=True)
    return philox_impl(c0, c1, c2, c3, seed_lo, seed_hi, n_rounds)


@jit
def randint(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block, returns a single
    block of random :code:`int32`.

    If you need multiple streams of random numbers,
    using `randint4x` is likely to be faster than calling `randint` 4 times.

    :param seed: The seed for generating random numbers.
    :param offset: The offsets to generate random numbers for.
    """
    ret, _, _, _ = randint4x(seed, offset, n_rounds)
    return ret


@jit
def randint4x(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block, returns four
    blocks of random :code:`int32`.

    This is the maximally efficient entry point
    to Triton's Philox pseudo-random number generator.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    # _0 = tl.zeros(offset.shape, offset.dtype)

    offset_lo = offset.to(tl.uint32)
    _0 = offset_lo * 0

    if tl.constexpr(offset.dtype.primitive_bitwidth) > 32:
        offset_hi = (offset >> 32).to(tl.uint32)
    else:
        offset_hi = _0

    return philox(seed, offset_lo, offset_hi, _0, _0, n_rounds)


# -------------------
# rand
# -------------------

# @jit
# def uint32_to_uniform_float(x):
#     """
#     Numerically stable function to convert a random uint32 into a random float uniformly sampled in [0, 1).
#     """
#     two_to_the_minus_32: tl.constexpr = 2.328306e-10
#     return x * two_to_the_minus_32


@jit
def uint_to_uniform_float(x):
    """
    Numerically stable function to convert a random uint into a random float uniformly sampled in [0, 1).
    """
    # TODO: fix frontend issues and cleanup
    # conditions can be simplified
    # scale is ((2**23 - 1) / 2**23) * 2**(N_BITS - 1)
    if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
        # maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
        x = x.to(tl.int32, bitcast=True)
        scale = 4.6566127342e-10
    else:
        tl.static_assert(tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64))
        x = x.to(tl.int64, bitcast=True)
        scale = 1.0842020432385337e-19
    x = tl.where(x < 0, -x - 1, x)
    return x * scale


@jit
def rand(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block,
    returns a block of random :code:`float32` in :math:`U(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    source = randint(seed, offset, n_rounds)
    return uint_to_uniform_float(source)


@jit
def rand4x(seed, offsets, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offsets` block,
    returns 4 blocks of random :code:`float32` in :math:`U(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    i1, i2, i3, i4 = randint4x(seed, offsets, n_rounds)
    u1 = uint_to_uniform_float(i1)
    u2 = uint_to_uniform_float(i2)
    u3 = uint_to_uniform_float(i3)
    u4 = uint_to_uniform_float(i4)
    return u1, u2, u3, u4


# -------------------
# randn
# -------------------


@jit
def pair_uniform_to_normal(u1, u2):
    """Box-Muller transform"""
    u1 = tl.maximum(1.0e-7, u1)
    th = 6.283185307179586 * u2
    r = math.sqrt(-2.0 * math.log(u1))
    return r * math.cos(th), r * math.sin(th)


@jit
def randn(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block,
    returns a block of random :code:`float32` in :math:`\\mathcal{N}(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    i1, i2, _, _ = randint4x(seed, offset, n_rounds)
    u1 = uint_to_uniform_float(i1)
    u2 = uint_to_uniform_float(i2)
    n1, _ = pair_uniform_to_normal(u1, u2)
    return n1


@jit
def randn4x(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    """
    Given a :code:`seed` scalar and an :code:`offset` block,
    returns 4 blocks of random :code:`float32` in :math:`\\mathcal{N}(0, 1)`.

    :param seed: The seed for generating random numbers.
    :param offsets: The offsets to generate random numbers for.
    """
    u1, u2, u3, u4 = rand4x(seed, offset, n_rounds)
    n1, n2 = pair_uniform_to_normal(u1, u2)
    n3, n4 = pair_uniform_to_normal(u3, u4)
    return n1, n2, n3, n4
