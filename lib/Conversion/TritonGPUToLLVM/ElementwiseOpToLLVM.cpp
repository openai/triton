#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

namespace {
/* ----- FP8E5M2 ------ */
// This data-type is the standard FP8E5M2 format

struct Fp8ConversionDesc {
  std::string ptx;
  int inVecWidthBits;
  int outVecWidthBits;
  size_t numElements;
};

static const Fp8ConversionDesc Fp16_to_Fp8E5M2_RTNE(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {"{                            \n"
           ".reg .b32 a<2>;              \n"
           "and.b32 a0, $1, 0xfffefffe;  \n"   // a0 &= 0xfffefffe
           "and.b32 a1, $2, 0xfffefffe;  \n"   // (strip lowest bit)
           "add.u32 a0, a0, 0x00800080;  \n"   // a0 += 0x00800080
           "add.u32 a1, a1, 0x00800080;  \n"   // (round to nearest)
           "prmt.b32 $0, a0, a1, 0x7531; \n\t" // output = a1a0
           "}",
           32, 32, 4};
  } else {
    ret = {"cvt.rn.satfinite.e5m2x2.f16x2 $0, $1; \n\t", 32, 16, 2};
  }
  return ret;
}

const Fp8ConversionDesc Fp16_to_Fp8E5M2_RTZ = {
    "{                            \n"
    ".reg .b32 a<2>;              \n"
    "and.b32 a0, $1, 0xfffefffe;  \n"   // a0 &= 0xfffefffe
    "and.b32 a1, $2, 0xfffefffe;  \n"   // (strip lowest bit)
    "prmt.b32 $0, a0, a1, 0x7531; \n\t" // output = a1a0
    "}",
    32, 32, 4};

static const Fp8ConversionDesc Fp8E5M2_to_Fp16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {"{                           \n"
           "prmt.b32 $0, 0, $2, 0x5140; \n\t"
           "prmt.b32 $1, 0, $2, 0x7362; \n\t"
           "}",
           32, 32, 4};
  } else {
    ret = {"cvt.rn.f16x2.e5m2x2 $0, $1; \n\t", 16, 32, 2};
  }
  return ret;
}

static const Fp8ConversionDesc Fp8E5M2_to_Bf16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {
        "{                                        \n"
        ".reg .b32 a<2>, b<2>, c<4>, d<4>, e112;  \n" // if input = 0xf1f2f3f4
        "mov.u32 e112, 0x77800000;                \n"
        "prmt.b32 a0, 0, $2, 0x5140;              \n" // a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7362;              \n" // a1 = 0xf100f200
        "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;    \n" // b0 = a0 & 0x7fff7fff
        "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;    \n" // (strip sign)
        "shr.b32  b0, b0, 3;                      \n" // b0 >>= 3
        "shr.b32  b1, b1, 3;                      \n" // shift into bf16
                                                      // position
        "and.b32 c0, b0, 0xFFFF0000;              \n" // c0 = f3
        "shl.b32 c1, b0, 16;                      \n" // c1 = f4
        "and.b32 c2, b1, 0xFFFF0000;              \n" // c2 = f1
        "shl.b32 c3, b1, 16;                      \n" // c3 = f2
        "mul.f32 d0, c0, e112;                    \n" // d0 = c0 * 0x77800000
        "mul.f32 d1, c1, e112;                    \n" // d1 = c1 * 0x77800000
        "mul.f32 d2, c2, e112;                    \n" // d2 = c2 * 0x77800000
        "mul.f32 d3, c3, e112;                    \n" // d3 = c3 * 0x77800000
        "prmt.b32 b0, d0, d1, 0x3276;             \n" // b0 = 0xd3d4
        "prmt.b32 b1, d2, d3, 0x3276;             \n" // b1 = 0xd1d2
        "lop3.b32 $0, b0, 0x80008000, a0, 0xf8;   \n" // out0 =
                                                      // b0|(0x80008000&a0)
        "lop3.b32 $1, b1, 0x80008000, a1, 0xf8;   \n" // (restore sign)
        "}",
        32, 32, 4};
  } else {
    ret = {
        "{                                       \n"
        ".reg .b32 a<2>, b<2>;                  \n" // if input = 0xf1f2f3f4
        ".reg .b32 e112;                        \n"
        "mov.u32 e112, 0x77807780;              \n" // 2**112 represented as
                                                    // bf16x2
        "prmt.b32 a0, 0, $2, 0x5140;            \n" // a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7362;            \n" // a1 = 0xf100f200
        "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n" // b0 = a0 & 0x7fff7fff
        "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n" // (strip sign)
        "shr.b32  b0, b0, 3;                    \n" // b0 >>= 3
        "shr.b32  b1, b1, 3;                    \n" // shift into bf16 position
        "lop3.b32 b0, b0, 0x80008000, a0, 0xf8; \n" // out0 = b0|(0x80008000&a0)
        "lop3.b32 b1, b1, 0x80008000, a1, 0xf8; \n" // (restore sign)
        "mul.rn.bf16x2 $0, b0, e112;            \n" // b0.exp += 2**7-2**4
        "mul.rn.bf16x2 $1, b1, e112;            \n" // exponent compensate = 112
        "}",
        32, 32, 4};
  }
  return ret;
}

static const Fp8ConversionDesc Bf16_to_Fp8E5M2(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {
        "{                                           \n" // bf16=fp8>>3 + 112<<7
        ".reg .u32 sign, sign<2>, nosign, nosign<2>; \n" // fp8_min = 0b00000000
        ".reg .u32 fp8_min, fp8_max, rn_;            \n" // fp8_max = 0b11111111
        "mov.u32 fp8_min, 0x38003800;                \n" // so bf16_min = 0x3800
        "mov.u32 fp8_max, 0x57e057e0;                \n" // so bf16_max = 0x57e0
        "mov.u32 rn_, 0x00100010;                    \n" // round to nearest
        "and.b32 sign0, $1, 0x80008000;              \n" // sign0=in0&0x80008000
        "and.b32 sign1, $2, 0x80008000;              \n" // (store sign)
        "prmt.b32 sign, sign0, sign1, 0x7531;        \n"
        "and.b32 nosign0, $1, 0x7fff7fff;            \n" // nosign0=in0&0x7fff7fff
        "and.b32 nosign1, $2, 0x7fff7fff;            \n" // (strip sign)

        // nosign = clamp(nosign, min, max)
        ".reg .u32 nosign_0_<2>, nosign_1_<2>;       \n"
        "and.b32 nosign_0_0, nosign0, 0xffff0000;    \n"
        "max.u32 nosign_0_0, nosign_0_0, 0x38000000; \n"
        "min.u32 nosign_0_0, nosign_0_0, 0x57e00000; \n"
        "and.b32 nosign_0_1, nosign0, 0x0000ffff;    \n"
        "max.u32 nosign_0_1, nosign_0_1, 0x3800;     \n"
        "min.u32 nosign_0_1, nosign_0_1, 0x57e0;     \n"
        "or.b32 nosign0, nosign_0_0, nosign_0_1;     \n"
        "and.b32 nosign_1_0, nosign1, 0xffff0000;    \n"
        "max.u32 nosign_1_0, nosign_1_0, 0x38000000; \n"
        "min.u32 nosign_1_0, nosign_1_0, 0x57e00000; \n"
        "and.b32 nosign_1_1, nosign1, 0x0000ffff;    \n"
        "max.u32 nosign_1_1, nosign_1_1, 0x3800;     \n"
        "min.u32 nosign_1_1, nosign_1_1, 0x57e0;     \n"
        "or.b32 nosign1, nosign_1_0, nosign_1_1;     \n"

        "add.u32 nosign0, nosign0, rn_;              \n" // nosign0 += rn_
        "add.u32 nosign1, nosign1, rn_;              \n" // (round to nearest)
        "sub.u32 nosign0, nosign0, 0x38003800;       \n" // nosign0-=0x38003800
        "sub.u32 nosign1, nosign1, 0x38003800;       \n" // (compensate offset)
        "shl.b32 nosign0, nosign0, 3;                \n" // nosign0 <<= 3
        "shl.b32 nosign1, nosign1, 3;                \n" // shift into to fp8e4
        "prmt.b32 nosign, nosign0, nosign1, 0x7531;  \n" // nosign0 = 0xf100f200
                                                         // nosign1 = 0xf300f400
                                                         // nosign = 0xf3f4f1f2
        "or.b32 $0, nosign, sign;                    \n" // restore sign
        "}",
        32, 32, 4};
  } else {
    ret = {"{                                       \n"
           ".reg .b16 a<2>;                         \n"
           ".reg .f32 b<2>;                         \n"
           "mov.b32 {a0, a1}, $1;                   \n"
           "cvt.f32.bf16 b0, a0;                    \n"
           "cvt.f32.bf16 b1, a1;                    \n"
           "cvt.rn.satfinite.e5m2x2.f32 $0, b1, b0; \n"
           "}",
           32, 16, 2};
  }
  return ret;
}
/* ----- FP8E4M3B15 ------ */
// This data-type is a variant of the standard FP8E4M3 format.
// It was designed for fast software conversion to FP16 on
// nvidia GPUs that do not support it natively.
// This is the same format as FP8E4M3Nv, but:
//   - the exponent bias is 15 instead of 7
//   - 0xff and 0x7f are mapped to +-1.750 instead of +-nan
const Fp8ConversionDesc Fp8E4M3B15_to_Fp16 = {
    "{                                      \n"
    ".reg .b32 a<2>, b<2>;                  \n"
    "prmt.b32 a0, 0, $2, 0x5746;            \n"
    "and.b32 b0, a0, 0x7f007f00;            \n"
    "and.b32 b1, a0, 0x00ff00ff;            \n"
    "and.b32 a1, a0, 0x00800080;            \n"
    "shr.b32  b0, b0, 1;                    \n"
    "add.u32 b1, b1, a1;                    \n"
    "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n"
    "shl.b32 $1, b1, 7;                     \n"
    "}                                      \n",
    32, 32, 4};

static const Fp8ConversionDesc Fp16_to_Fp8E4M3B15(bool has_minx2) {
  std::string ret;
  ret += "{                                      \n"
         ".reg .pred p<4>;                       \n"
         ".reg .b32 a<2>, b<2>;                  \n"
         ".reg .b16 c<4>;                        \n"
         ".reg .b16 max_val_f16;                 \n"
         ".reg .b32 max_val_f16x2;               \n"
         "mov.b16 max_val_f16,   0x3F00;         \n"
         "mov.b32 max_val_f16x2, 0x3F003F00;     \n"
         "and.b32 a0, $1, 0x7fff7fff;            \n"
         "and.b32 a1, $2, 0x7fff7fff;            \n";
  if (has_minx2)
    ret += "min.f16x2 a0, a0, max_val_f16x2;      \n"
           "min.f16x2 a1, a1, max_val_f16x2;      \n";
  else
    ret += "setp.lt.f16x2  p0|p1, a0, max_val_f16x2;   \n"
           "setp.lt.f16x2  p2|p3, a1, max_val_f16x2;   \n"
           "mov.b32 {c0, c1}, a0;                \n"
           "mov.b32 {c2, c3}, a1;                \n"
           "selp.b16  c0, c0, max_val_f16, p0;   \n"
           "selp.b16  c1, c1, max_val_f16, p1;   \n"
           "selp.b16  c2, c2, max_val_f16, p2;   \n"
           "selp.b16  c3, c3, max_val_f16, p3;   \n"
           "mov.b32 a0, {c0, c1};                \n"
           "mov.b32 a1, {c2, c3};                \n";
  ret += "mad.lo.u32 a0, a0, 2, 0x00800080;      \n"
         "mad.lo.u32 a1, a1, 2, 0x00800080;      \n"
         "lop3.b32 b0, $1, 0x80008000, a0, 0xea; \n"
         "lop3.b32 b1, $2, 0x80008000, a1, 0xea; \n"
         "prmt.b32 $0, b0, b1, 0x7531;           \n"
         "}";
  return {ret, 32, 32, 4};
}

/* ----- FP8E4M3B15X4 ------ */
// NOTE: NOT USED RIGHT NOW
// Packed variant of FP8E4M3B15
// A little bit more efficient but elements need are not
// serialized as you expect when 4 are packed into int32.

// fast conversion code provided by Scott Gray @ OpenAI
// $0 = (($2 << 1) & 0x80008000u) | (($2 << 7) & 0x3f803f80u);
// $1 = (($2 << 0) & 0x80008000u) | (($2 << 0) & 0x3f803f80u);
// WARN: subnormal (0bs0000xxx) are not handled
static const Fp8ConversionDesc Fp8E4M3B15x4_to_Fp16 = {
    "{                                      \n"
    ".reg .b32 a<2>;                        \n"
    "add.u32 a0, $2, $2;                    \n"
    "shl.b32 a1, $2, 7;                     \n"
    "and.b32  $0, a0, 0x80008000;           \n"
    "lop3.b32 $0, $0, a1, 0x3f803f80, 0xf8; \n"
    "and.b32  $1, $2, 0xbf80bf80;           \n"
    "}",
    32, 32, 4};

// Fp16 -> Fp8E4M3B15 (packed)
// fast conversion code provided by Scott Gray @ OpenAI
// ret = ((e4.x >> 1) & (0x80008000u >> 1)) |
//       ((e4.x >> 7) & (0x3f803f80u >> 7)) |
//       ((e4.y >> 0) & (0x80008000u >> 0)) |
//       ((e4.y >> 0) & (0x3f803f80u >> 0)) ;
// WARN: subnormal (0bs0000xxx) are not handled
static const Fp8ConversionDesc Fp16_to_Fp8E4M3B15x4 = {
    "{                                       \n"
    ".reg .b32 a<2>;                         \n"
    "shr.b32  a0, $1, 1;                     \n"
    "shr.b32  a1, $1, 7;                     \n"
    "and.b32  $0,     a0, 0x40004000;        \n"
    "lop3.b32 $0, $0, a1, 0x007f007f, 0xf8;  \n"
    "lop3.b32 $0, $0, $2, 0xbf80bf80, 0xf8;  \n"
    "}",
    32, 32, 4};

// Fp8E4M3 (x2) -> Fp16 (x2) (packed)
static const Fp8ConversionDesc Fp8E4M3Nv_to_Fp16 = {
    "{ \n"
    "cvt.rn.f16x2.e4m3x2 $0, $1; \n"
    "}",
    16, 32, 2};

// Fp16 (x2) -> Fp8E4M3 (x2) (packed)
static const Fp8ConversionDesc Fp16_to_Fp8E4M3Nv = {
    "{ \n"
    "cvt.rn.satfinite.e4m3x2.f16x2 $0, $1; \n"
    "}",
    32, 16, 2};

// Fp8E4M3 (x2) -> Fp16 (x2) (packed)
static const Fp8ConversionDesc Fp8E4M3Nv_to_Bf16 = {
    "{                                       \n"
    ".reg .b32 a;                            \n"
    ".reg .f16 a<2>;                         \n"
    ".reg .b16 b<2>;                         \n"
    "cvt.rn.f16x2.e4m3x2 a, $1;              \n"
    "mov.b32 {a0, a1}, a;                    \n"
    "cvt.bf16.f16 b0, a0;                    \n"
    "cvt.bf16.f16 b1, a1;                    \n"
    "mov.b32 $0, {b0, b1};                   \n"
    "}",
    16, 32, 2};

// Bf16 (x2) -> Fp8E4M3 (x2) (packed)
static const Fp8ConversionDesc Bf16_to_Fp8E4M3Nv = {
    "{                                       \n"
    ".reg .b16 a<2>;                         \n"
    ".reg .f32 b<2>;                         \n"
    "mov.b32 {a0, a1}, $1;                   \n"
    "cvt.f32.bf16 b0, a0;                    \n"
    "cvt.f32.bf16 b1, a1;                    \n"
    "cvt.rn.satfinite.e4m3x2.f32 $0, b1, b0; \n"
    "}",
    32, 16, 2};

// Fp32 (x2) -> Fp8 (x2) (packed)
static const Fp8ConversionDesc Fp32_to_Fp8E4M3Nv = {
    "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \n", 32, 16, 2};
static const Fp8ConversionDesc Fp32_to_Fp8E5M2 = {
    "cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1; \n", 32, 16, 2};

/* ----- Packed integer to BF16 ------ */
static const std::string S8_to_Bf16 =
    "{                                      \n"
    ".reg .f32 f<8>;                        \n"
    ".reg .b32 r<8>;                        \n"
    "prmt.b32 r0, $2, $2, 34944;            \n"
    "prmt.b32 r1, $2, $2, 39313;            \n"
    "prmt.b32 r2, $2 ,$2, 43682;            \n"
    "prmt.b32 r3, $2 ,$2, 48051;            \n"
    "add.s32 r4, r0, 1262485504;            \n"
    "mov.b32 f0, r4;                        \n"
    "add.f32 f4, f0, 0fCB400000;            \n"
    "add.s32 r5, r1, 1262485504;            \n"
    "mov.b32 f1, r5;                        \n"
    "add.f32 f5, f1, 0fCB400000;            \n"
    "add.s32 r6, r2, 1262485504;            \n"
    "mov.b32 f2, r6;                        \n"
    "add.f32 f6, f2, 0fCB400000;            \n"
    "add.s32 r7, r3, 1262485504;            \n"
    "mov.b32 f3, r7;                        \n"
    "add.f32 f7, f3, 0fCB400000;            \n"
    "cvt.rn.bf16x2.f32 $0, f5, f4;          \n"
    "cvt.rn.bf16x2.f32 $1, f7, f6;          \n"
    "}";

// MMA encoding has a different order depending on the element's bit width;
// reorder if we're in this case.
static SmallVector<Value> reorderValues(const SmallVector<Value> &values,
                                        Type inType, Type ouType) {
  auto inTensorTy = inType.dyn_cast<RankedTensorType>();
  auto ouTensorTy = ouType.dyn_cast<RankedTensorType>();
  if (!inTensorTy || !ouTensorTy)
    return values;
  auto inEncoding =
      dyn_cast<triton::gpu::DotOperandEncodingAttr>(inTensorTy.getEncoding());
  auto ouEncoding =
      dyn_cast<triton::gpu::DotOperandEncodingAttr>(ouTensorTy.getEncoding());
  assert(inEncoding == ouEncoding);
  if (!inEncoding)
    return values;
  // If the parent of the dot operand is in block encoding, we don't need to
  // reorder elements
  auto parentEncoding =
      dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>(ouEncoding.getParent());
  if (!parentEncoding)
    return values;
  size_t inBitWidth = inTensorTy.getElementType().getIntOrFloatBitWidth();
  size_t ouBitWidth = ouTensorTy.getElementType().getIntOrFloatBitWidth();
  auto ouEltTy = ouTensorTy.getElementType();
  if (inBitWidth == ouBitWidth)
    return values;
  if (inBitWidth == 16 && ouBitWidth == 32) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 8) {
      ret.push_back(values[i]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
    }
    return ret;
  }
  if (inBitWidth == 8 && ouBitWidth == 16) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 16) {
      ret.push_back(values[i + 0]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 8]);
      ret.push_back(values[i + 9]);
      ret.push_back(values[i + 10]);
      ret.push_back(values[i + 11]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
      ret.push_back(values[i + 12]);
      ret.push_back(values[i + 13]);
      ret.push_back(values[i + 14]);
      ret.push_back(values[i + 15]);
    }
    return ret;
    // for (unsigned i = 0; i < values.size(); i += 16) {
    //   ret.push_back(values[i]);
    //   ret.push_back(values[i + 1]);
    //   ret.push_back(values[i + 4]);
    //   ret.push_back(values[i + 5]);
    //   ret.push_back(values[i + 8]);
    //   ret.push_back(values[i + 9]);
    //   ret.push_back(values[i + 12]);
    //   ret.push_back(values[i + 13]);

    //   ret.push_back(values[i + 2]);
    //   ret.push_back(values[i + 3]);
    //   ret.push_back(values[i + 6]);
    //   ret.push_back(values[i + 7]);
    //   ret.push_back(values[i + 10]);
    //   ret.push_back(values[i + 11]);
    //   ret.push_back(values[i + 14]);
    //   ret.push_back(values[i + 15]);
    // }
  }
  llvm_unreachable("unimplemented code path");
}

inline Type getElementType(Value value) {
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return tensorType.getElementType();
  return type;
}

inline SmallVector<Value> unpackI32(const SmallVector<Value> &inValues,
                                    Type srcTy,
                                    ConversionPatternRewriter &rewriter,
                                    Location loc,
                                    const LLVMTypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<NvidiaMmaEncodingAttr>()))
    return inValues;
  SmallVector<Value> outValues;
  for (auto v : inValues) {
    // cast i32 to appropriate eltType vector and extract elements
    auto eltType = typeConverter->convertType(tensorTy.getElementType());
    auto vecType = vec_ty(eltType, 32 / eltType.getIntOrFloatBitWidth());
    auto vec = bitcast(v, vecType);
    for (int i = 0; i < 32 / eltType.getIntOrFloatBitWidth(); i++) {
      outValues.push_back(extract_element(vec, i32_val(i)));
    }
  }
  return outValues;
}

inline SmallVector<Value> packI32(const SmallVector<Value> &inValues,
                                  Type srcTy,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc,
                                  const LLVMTypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<NvidiaMmaEncodingAttr>()))
    return inValues;
  SmallVector<Value> outValues;
  auto eltType = typeConverter->convertType(tensorTy.getElementType());
  int vecWidth = 32 / eltType.getIntOrFloatBitWidth();
  auto vecType = vec_ty(eltType, vecWidth);
  for (int i = 0; i < inValues.size(); i += vecWidth) {
    Value vec = undef(vecType);
    for (int j = 0; j < vecWidth; j++) {
      vec = insert_element(vec, inValues[i + j], i32_val(j));
    }
    outValues.push_back(bitcast(vec, i32_ty));
  }
  return outValues;
}

typedef std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                         const SmallVector<Value> &)>
    ConverterT;

static ConverterT makeConverterFromPtx(const std::string &ptxAsm, Type inType,
                                       Type outType,
                                       const int inVecWidthBits = 32,
                                       const int outVecWidthBits = 32) {

  ConverterT converter =
      [ptxAsm, inType, outType, inVecWidthBits,
       outVecWidthBits](Location loc, ConversionPatternRewriter &rewriter,
                        const SmallVector<Value> &v) -> SmallVector<Value> {
    int numElements = v.size();
    assert(numElements == 4 || numElements == 2 && "invalid vector size");

    auto ctx = rewriter.getContext();
    int inBitwidth = inType.getIntOrFloatBitWidth();
    int outBitwidth = outType.getIntOrFloatBitWidth();
    // first, we pack `v` into 32-bit ints
    int inVecWidth = inVecWidthBits / inBitwidth;
    auto inVecTy = vec_ty(inType, inVecWidth);
    SmallVector<Value> inPacked(numElements / inVecWidth, undef(inVecTy));
    for (size_t i = 0; i < numElements; i++)
      inPacked[i / inVecWidth] = insert_element(
          inVecTy, inPacked[i / inVecWidth], v[i], i32_val(i % inVecWidth));
    for (size_t i = 0; i < inPacked.size(); i++)
      inPacked[i] = bitcast(inPacked[i], int_ty(inVecWidthBits));

    // then, we run the provided inline PTX
    int outVecWidth = outVecWidthBits / outBitwidth;
    int outNums = numElements / outVecWidth;
    PTXBuilder builder;
    SmallVector<PTXBuilder::Operand *> operands;
    auto outConstriant = outVecWidthBits == 16 ? "=h" : "=r";
    auto inConstraint = inVecWidthBits == 16 ? "h" : "r";
    for (int i = 0; i < outNums; i++) {
      operands.push_back(builder.newOperand(outConstriant));
    }

    for (Value inVal : inPacked) {
      operands.push_back(builder.newOperand(inVal, inConstraint));
    }

    auto &ptxOp = *builder.create(ptxAsm);
    ptxOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto outVecTy = vec_ty(outType, outVecWidth);
    SmallVector<Value> outPacked;
    if (outNums == 1)
      outPacked.push_back(builder.launch(rewriter, loc, outVecTy, false));
    else {
      auto outStructTy = struct_ty(SmallVector<Type>(outNums, outVecTy));
      auto outStruct = builder.launch(rewriter, loc, outStructTy, false);
      for (int i = 0; i < outNums; i++)
        outPacked.push_back(extract_val(outVecTy, outStruct, i));
    }
    // unpack the output
    SmallVector<Value> ret;
    for (size_t i = 0; i < numElements; i++)
      ret.push_back(extract_element(outType, outPacked[i / outVecWidth],
                                    i32_val(i % outVecWidth)));
    return ret;
  };
  return converter;
}

class MultipleOperandsRange
    : public iterator_range<SmallVector<SmallVector<Value>>::iterator> {
  using ContainerT = SmallVector<SmallVector<Value>>;

public:
  using iterator_range<ContainerT::iterator>::iterator_range;
  ContainerT::reference operator[](ContainerT::size_type idx) {
    return begin()[idx];
  }
  ContainerT::const_reference operator[](ContainerT::size_type idx) const {
    return begin()[idx];
  }
  ContainerT::size_type size() const { return end() - begin(); }
};

// Base pattern for elementwise conversion using ConcreteT. Unpacks individual
// elements from a `!llvm.struct` via `llvm.extactvalue`, calls
// ConcreteT::createDestOps on each element, and packs them back into an
// `!llvm.struct` using `llvm.insertvalue`.
//
// Also supports processing the inputs in a vectorized form by consuming and
// producing multiple operand sets in ConcreteT::createDestOps.
template <typename SourceOp, typename ConcreteT>
class ElementwiseOpConversionBase : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ElementwiseOpConversionBase(LLVMTypeConverter &typeConverter,
                                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                                       PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        axisAnalysisPass(axisAnalysisPass) {}

  // Try to deduplicate the resultVals based on the
  // constancy properties of the result discovered by
  // the axis analysis pass. If possible, redundant
  // computation is eliminated.
  SmallVector<Value> maybeDeduplicate(SourceOp op,
                                      SmallVector<Value> resultVals) const {
    if (!isMemoryEffectFree(op))
      // the op has side effects: can't dedup
      return resultVals;
    SmallVector<Value> results = op->getResults();
    if (results.size() == 0 || results.size() > 1)
      // there must be exactly 1 result
      return resultVals;
    Value result = results[0];
    Type type = result.getType();
    if (!type)
      return resultVals;
    RankedTensorType rtType = type.dyn_cast<RankedTensorType>();
    if (!rtType)
      // the result must be a tensor
      return resultVals;
    Attribute encoding = rtType.getEncoding();
    if (!encoding)
      // encoding not available
      return resultVals;
    if (!encoding.dyn_cast<triton::gpu::BlockedEncodingAttr>() &&
        !encoding.dyn_cast<triton::gpu::SliceEncodingAttr>()) {
      // TODO: constraining the ecndoing type here is necessary
      // for avoiding crashes in the triton::gpu::getElemsPerThread
      // call below happening in the test_core::test_fp8_dot_acc
      return resultVals;
    }

    SmallVector<unsigned> elemsPerThread =
        triton::gpu::getElemsPerThread(rtType);
    int rank = elemsPerThread.size();
    if (product<unsigned>(elemsPerThread) != resultVals.size())
      return resultVals;
    AxisInfo *axisInfo = axisAnalysisPass.getAxisInfo(result);
    if (!axisInfo)
      // axis info (e.g., constancy) not available
      return resultVals;
    SmallVector<unsigned> sizePerThread =
        triton::gpu::getSizePerThread(encoding);
    if (rank != sizePerThread.size())
      return resultVals;

    SmallVector<int64_t> constancy = axisInfo->getConstancy();
    if (rank != constancy.size())
      return resultVals;
    bool hasConstancy = false;
    for (int i = 0; i < rank; ++i) {
      if (constancy[i] > sizePerThread[i]) {
        if (constancy[i] % sizePerThread[i] != 0)
          // constancy is not evenly covered by sizePerThread
          return resultVals;
        // can't move the values across different
        // "sizePerThread"-sized blocks
        constancy[i] = sizePerThread[i];
      }
      if (elemsPerThread[i] < 1 || constancy[i] < 1)
        return resultVals;
      if (!(elemsPerThread[i] % constancy[i] == 0 ||
            constancy[i] % elemsPerThread[i] == 0))
        // either the constancy along each dimension must fit
        // into the elemsPerThread or the other way around
        return resultVals;
      if (constancy[i] > 1)
        hasConstancy = true;
    }
    if (!hasConstancy)
      // nothing to deduplicate
      return resultVals;

    if (rank > 1) {
      // reorder the shape and constancy vectors by the axis order:
      // from the fastest-changing to the smallest-changing axis
      SmallVector<unsigned> order = triton::gpu::getOrder(encoding);
      if (rank != order.size())
        return resultVals;
      elemsPerThread = applyPermutation(elemsPerThread, order);
      constancy = applyPermutation(constancy, order);
    }

    SmallVector<unsigned> strides(rank, 1);
    for (int i = 1; i < rank; ++i) {
      strides[i] = strides[i - 1] * elemsPerThread[i - 1];
    }
    SmallVector<Value> dedupResultVals;
    dedupResultVals.reserve(resultVals.size());
    for (int i = 0; i < resultVals.size(); ++i) {
      // each coordinate of the orig_idx is "coarsened" using the
      // constancy along this dimension: the resulting dedup_idx
      // points to the reused value in the original resultsVal
      int orig_idx = i;
      int dedup_idx = 0;
      for (int j = 0; j < rank; ++j) {
        int coord_j = orig_idx % elemsPerThread[j];
        dedup_idx += (coord_j / constancy[j] * constancy[j]) * strides[j];
        orig_idx /= elemsPerThread[j];
      }
      dedupResultVals.push_back(resultVals[dedup_idx]);
    }

    return dedupResultVals;
  }

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    Location loc = op->getLoc();
    // element type
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    SmallVector<SmallVector<Value>> allOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto subOperands = unpackLLElements(loc, operand, rewriter);
      subOperands = unpackI32(subOperands, argTy, rewriter, loc,
                              this->getTypeConverter());
      allOperands.resize(subOperands.size());
      for (auto v : llvm::enumerate(subOperands))
        allOperands[v.index()].push_back(v.value());
    }
    if (allOperands.size() == 0)
      allOperands.push_back({});

    SmallVector<Value> resultVals;
    for (auto it = allOperands.begin(), end = allOperands.end(); it != end;) {
      auto curr = static_cast<const ConcreteT *>(this)->createDestOps(
          op, adaptor, rewriter, elemTy, MultipleOperandsRange(it, end), loc);
      if (curr.size() == 0)
        return failure();
      for (auto v : curr) {
        if (!static_cast<bool>(v))
          return failure();
        resultVals.push_back(v);
      }
      it += curr.size();
    }
    if (op->getNumOperands() > 0) {
      auto argTy = op->getOperand(0).getType();
      resultVals = reorderValues(resultVals, argTy, resultTy);
    }
    resultVals = maybeDeduplicate(op, resultVals);
    resultVals =
        packI32(resultVals, resultTy, rewriter, loc, this->getTypeConverter());
    Value view = packLLElements(loc, this->getTypeConverter(), resultVals,
                                rewriter, resultTy);
    rewriter.replaceOp(op, view);

    return success();
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;

private:
  int computeCapability;
};

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<DestOp> createDestOps(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    Type elemTy, MultipleOperandsRange operands,
                                    Location loc) const {
    return {rewriter.create<DestOp>(loc, elemTy, operands[0],
                                    adaptor.getAttributes().getValue())};
  }
};

// Attempts to use vectorized conversions via inline PTX when possible.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion> {
  using ElementwiseOpConversionBase<
      triton::FpToFpOp, FpToFpOpConversion>::ElementwiseOpConversionBase;

  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability, PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  static Value convertBf16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    PTXBuilder builder;
    auto &cvt = *builder.create("cvt.f32.bf16");
    auto res = builder.newOperand("=r");
    auto operand = builder.newOperand(v, "h");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f32_ty, false);
  }

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    PTXBuilder builder;
    auto &cvt = *builder.create("cvt.f32.f16");
    auto res = builder.newOperand("=r");
    auto operand = builder.newOperand(v, "h");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f32_ty, false);
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    PTXBuilder builder;
    StringRef ptx;
    switch (rounding) {
    case RoundingMode::RTNE:
      ptx = "cvt.rn.bf16.f32";
      break;
    case RoundingMode::RTZ:
      ptx = "cvt.rz.bf16.f32";
      break;
    }
    auto &cvt = *builder.create(ptx.str());
    auto res = builder.newOperand("=h");
    auto operand = builder.newOperand(v, "r");
    cvt(res, operand);
    // TODO: This is a hack to get the right type. We should be able to invoke
    // the type converter
    return builder.launch(rewriter, loc, i16_ty, false);
  }

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    PTXBuilder builder;
    StringRef ptx;
    switch (rounding) {
    case RoundingMode::RTNE:
      ptx = "cvt.rn.f16.f32";
      break;
    case RoundingMode::RTZ:
      ptx = "cvt.rz.f16.f32";
      break;
    default:
      llvm::errs() << "WARNING: unsupported rounding mode for f32->f16 "
                      "conversion: "
                   << stringifyRoundingMode(rounding) << "\n";
      llvm_unreachable("");
    }
    auto &cvt = *builder.create(ptx.str());
    auto res = builder.newOperand("=h");
    auto operand = builder.newOperand(v, "r");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f16_ty, false);
  }

  std::pair<ConverterT, size_t>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3B15TyID = TypeID::get<mlir::Float8E4M3B11FNUZType>();
    auto F8E4M3TyID = TypeID::get<mlir::Float8E4M3FNUZType>();
    auto F8E5M2TyID = TypeID::get<mlir::Float8E5M2Type>();
    auto F8E4M3FNTyID = TypeID::get<mlir::Float8E4M3FNType>();
    auto F16TyID = TypeID::get<mlir::Float16Type>();
    auto BF16TyID = TypeID::get<mlir::BFloat16Type>();
    auto F32TyID = TypeID::get<mlir::Float32Type>();
    auto F64TyID = TypeID::get<mlir::Float64Type>();

    auto undefRounding = static_cast<RoundingMode>(-1);

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>, Fp8ConversionDesc>
        srcMap = {
            // F8 -> F16
            {{F8E4M3B15TyID, F16TyID, undefRounding}, Fp8E4M3B15_to_Fp16},
            {{F8E4M3FNTyID, F16TyID, undefRounding}, Fp8E4M3B15x4_to_Fp16},
            {{F8E4M3TyID, F16TyID, undefRounding}, Fp8E4M3Nv_to_Fp16},
            {{F8E5M2TyID, F16TyID, undefRounding},
             Fp8E5M2_to_Fp16(computeCapability >= 90)},
            // F16 -> F8
            {{F16TyID, F8E4M3B15TyID, RoundingMode::RTNE},
             Fp16_to_Fp8E4M3B15(computeCapability >= 80)},
            {{F16TyID, F8E4M3FNTyID, RoundingMode::RTNE}, Fp16_to_Fp8E4M3B15x4},
            {{F16TyID, F8E4M3TyID, RoundingMode::RTNE}, Fp16_to_Fp8E4M3Nv},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp16_to_Fp8E5M2_RTNE(computeCapability >= 90)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTZ}, Fp16_to_Fp8E5M2_RTZ},
            // F8 -> BF16
            {{F8E5M2TyID, BF16TyID, undefRounding},
             Fp8E5M2_to_Bf16(computeCapability >= 90)},
            {{F8E4M3TyID, BF16TyID, undefRounding}, Fp8E4M3Nv_to_Bf16},
            // BF16 -> F8
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Bf16_to_Fp8E5M2(computeCapability >= 90)},
            {{BF16TyID, F8E4M3TyID, RoundingMode::RTNE}, Bf16_to_Fp8E4M3Nv},
            // F32 -> F8
            {{F32TyID, F8E4M3TyID, RoundingMode::RTNE}, Fp32_to_Fp8E4M3Nv},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE}, Fp32_to_Fp8E5M2},
        };
    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (srcMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to "
                   << dstTy;
      if (roundingMode.has_value())
        llvm::errs() << " with rounding mode "
                     << stringifyRoundingMode(roundingMode.value());
      llvm::errs() << "\n";
      llvm::report_fatal_error("Unsupported rounding mode for conversion.");
    }
    if (computeCapability < 90 &&
        (srcTy.isFloat8E4M3FNUZ() || dstTy.isFloat8E4M3FNUZ())) {
      llvm::errs() << "Conversion from/to f8e4m3nv is only supported on "
                      "compute capability >= 90"
                   << "\n";
      llvm_unreachable("");
    }
    auto convDesc = srcMap.lookup(key);
    return {makeConverterFromPtx(
                convDesc.ptx, getTypeConverter()->convertType(srcTy),
                getTypeConverter()->convertType(dstTy), convDesc.inVecWidthBits,
                convDesc.outVecWidthBits),
            convDesc.numElements};
  }

  SmallVector<Value> createDestOps(triton::FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto srcElementType = getElementType(op.getFrom());
    auto dstElementType = getElementType(op.getResult());
    auto roundingMode = op.getRounding();

    if (dstElementType.isFloat8E5M2() || dstElementType.isFloat8E4M3FNUZ()) {
      assert(roundingMode.has_value() &&
             "Rounding mode must be specified for convertsions to fp8");

      // For now only RTNE is supported for conversions from fp16 to fp8
      if (!srcElementType.isF32() &&
          roundingMode.value() != RoundingMode::RTNE) {
        llvm::errs() << "Unsupported rounding mode for conversion to fp8: "
                     << stringifyRoundingMode(roundingMode.value()) << "\n";
        llvm_unreachable("");
      }
    }

    if (srcElementType.isF32() && dstElementType.isF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->fp16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToFp16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    if (srcElementType.isF32() && dstElementType.isBF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->bf16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToBf16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    bool useFP16IntermediateSrc =
        srcElementType.isF32() &&
        (!(computeCapability >= 90 && (dstElementType.isFloat8E4M3FNUZ() ||
                                       dstElementType.isFloat8E5M2())) ||
         roundingMode.value() == RoundingMode::RTZ);
    bool isDstFP32 = dstElementType.isF32();
    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    Type dstType = isDstFP32 ? f16_ty : dstElementType;
    auto [cvtFunc, numElements] =
        getConversionFunc(srcType, dstType, roundingMode);
    SmallVector<Value> inVals;
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    if (useFP16IntermediateSrc)
      for (Value &v : inVals)
        v = convertFp32ToFp16(loc, rewriter, v, RoundingMode::RTZ);
    inVals.resize(numElements, undef(typeConverter->convertType(srcType)));
    SmallVector<Value> outVals = cvtFunc(loc, rewriter, inVals);
    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    if (isDstFP32)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    return outVals;
  }

private:
  int computeCapability;
};

struct CmpIOpConversion
    : public ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<LLVM::ICmpOp> createDestOps(arith::CmpIOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          Type elemTy,
                                          MultipleOperandsRange operands,
                                          Location loc) const {
    return {rewriter.create<LLVM::ICmpOp>(
        loc, elemTy, ArithCmpIPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
  }

  static LLVM::ICmpPredicate
  ArithCmpIPredicateToLLVM(arith::CmpIPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__)                                                    \
  case arith::CmpIPredicate::item__:                                           \
    return LLVM::ICmpPredicate::item__

      __PRED_ENUM(eq);
      __PRED_ENUM(ne);
      __PRED_ENUM(sgt);
      __PRED_ENUM(sge);
      __PRED_ENUM(slt);
      __PRED_ENUM(sle);
      __PRED_ENUM(ugt);
      __PRED_ENUM(uge);
      __PRED_ENUM(ult);
      __PRED_ENUM(ule);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpIPredicate");
  }
};

struct CmpFOpConversion
    : public ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  static SmallVector<LLVM::FCmpOp>
  createDestOps(arith::CmpFOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                MultipleOperandsRange operands, Location loc) {
    return {rewriter.create<LLVM::FCmpOp>(
        loc, elemTy, ArithCmpFPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
  }

  static LLVM::FCmpPredicate
  ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

      __PRED_ENUM(OEQ, oeq);
      __PRED_ENUM(ONE, one);
      __PRED_ENUM(OGT, ogt);
      __PRED_ENUM(OGE, oge);
      __PRED_ENUM(OLT, olt);
      __PRED_ENUM(OLE, ole);
      __PRED_ENUM(ORD, ord);
      __PRED_ENUM(UEQ, ueq);
      __PRED_ENUM(UGT, ugt);
      __PRED_ENUM(UGE, uge);
      __PRED_ENUM(ULT, ult);
      __PRED_ENUM(ULE, ule);
      __PRED_ENUM(UNE, une);
      __PRED_ENUM(UNO, uno);
      __PRED_ENUM(AlwaysTrue, _true);
      __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }
};

struct ExternElementwiseOpConversion
    : public ElementwiseOpConversionBase<ExternElementwiseOp,
                                         ExternElementwiseOpConversion> {
  using Base = ElementwiseOpConversionBase<ExternElementwiseOp,
                                           ExternElementwiseOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  SmallVector<Value> createDestOps(ExternElementwiseOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    StringRef funcName = op.getSymbol();
    if (funcName.empty())
      llvm::errs() << "ExternElementwiseOpConversion";

    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetFuncOp(rewriter, op, funcName, funcType);
    return {
        rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]).getResult()};
  }

private:
  Type getFunctionType(Type resultType, ValueRange operands) const {
    SmallVector<Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(ConversionPatternRewriter &rewriter,
                                     ExternElementwiseOp op, StringRef funcName,
                                     Type funcType) const {
    using LLVM::LLVMFuncOp;

    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);

    auto parent = ((Operation *)op)->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    mlir::OpBuilder b(parent);
    auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
    ret.getOperation()->setAttr(
        "libname", StringAttr::get(op->getContext(), op.getLibname()));
    ret.getOperation()->setAttr(
        "libpath", StringAttr::get(op->getContext(), op.getLibpath()));
    return ret;
  }
};

struct ElementwiseInlineAsmOpConversion
    : public ConvertOpToLLVMPattern<ElementwiseInlineAsmOp> {
  using Base = ConvertOpToLLVMPattern<ElementwiseInlineAsmOp>;

  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  // If operand size is smaller than 32 bits, pack in groups of 32 bits.
  SmallVector<Value> packOperands(ElementwiseInlineAsmOp op,
                                  MultipleOperandsRange operands,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) const {
    SmallVector<Value> packedOperands;
    unsigned numPackedElements = op.getPackedElement();
    for (int i = 0, e = op.getNumOperands(); i < e; i++) {
      Type elemTy = getElementType(op.getOperand(i));
      unsigned bitWidth =
          elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 64;
      unsigned numElementPerReg = bitWidth < 32 ? 32 / bitWidth : 1;
      numElementPerReg = std::min(numElementPerReg, numPackedElements);
      for (int j = 0; j < numPackedElements; j += numElementPerReg) {
        if (numElementPerReg == 1) {
          packedOperands.push_back(operands[j][i]);
          continue;
        }
        Type t =
            vec_ty(getTypeConverter()->convertType(elemTy), numElementPerReg);
        Value packed = undef(t);
        for (int k = 0; k < numElementPerReg; k++) {
          packed = insert_element(packed, operands[j + k][i], i32_val(k));
        }
        packedOperands.push_back(packed);
      }
    }
    return packedOperands;
  }

  SmallVector<SmallVector<Value>>
  createDestOps(ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter,
                MultipleOperandsRange operands, Location loc) const {
    auto ctx = op->getContext();

    if (operands.size() % op.getPackedElement() != 0)
      llvm::report_fatal_error("Inline asm op has more packed elements than "
                               "number of elements per thread.");

    // Pack elems smaller than 32 bits into 32-bit registers.
    SmallVector<Value> packedOperands =
        packOperands(op, operands, rewriter, loc);

    // Types returned by the LLVM asm op.  If there's more than one, they'll be
    // wrapped in a struct.
    SmallVector<Type> asmRetTypes;
    for (auto result : op.getResult()) {
      auto ty = getTypeConverter()->convertType(getElementType(result));

      // Pack return elements into 32-bits.
      unsigned bitWidth = ty.isIntOrFloat() ? ty.getIntOrFloatBitWidth() : 64;
      unsigned numElemsPerReg =
          std::min(bitWidth < 32 ? 32 / bitWidth : 1, op.getPackedElement());
      assert(op.getPackedElement() % numElemsPerReg == 0);
      if (numElemsPerReg > 1) {
        ty = vec_ty(ty, numElemsPerReg);
      }
      for (unsigned i = 0; i < op.getPackedElement() / numElemsPerReg; i++) {
        asmRetTypes.push_back(ty);
      }
    }
    Type asmRetType =
        asmRetTypes.size() > 1 ? struct_ty(asmRetTypes) : asmRetTypes[0];

    Value asmResults =
        rewriter
            .create<LLVM::InlineAsmOp>(
                loc, asmRetType,
                /*operands=*/packedOperands,
                /*asm_string=*/op.getAsmString(),
                /*constraints=*/op.getConstraints(),
                /*has_side_effects=*/!op.getPure(),
                /*is_align_stack=*/false,
                /*asm_dialect=*/
                LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                          LLVM::AsmDialect::AD_ATT),
                /*operand_attrs=*/ArrayAttr())
            ->getResult(0);

    // asmResults is a flat struct; pack its values into
    // [return_value][op.getPackedElement()].
    SmallVector<SmallVector<Value>> ret(op->getNumResults());
    for (int i = 0; i < op->getNumResults(); i++) {
      for (int j = 0; j < op.getPackedElement(); j++) {
        auto val = asmRetTypes.size() > 1
                       ? extract_val(asmResults, i * op.getPackedElement() + j)
                       : asmResults;
        if (auto vectorTy = val.getType().dyn_cast<VectorType>()) {
          for (int k = 0; k < vectorTy.getNumElements(); k++) {
            ret[i].push_back(extract_element(val, i32_val(k)));
          }
          j += vectorTy.getNumElements() - 1;
        } else {
          ret[i].push_back(val);
        }
      }
    }
    return ret;
  }

  LogicalResult
  matchAndRewrite(ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Layout is unpackedOperands[operand][elem].
    SmallVector<SmallVector<Value>> unpackedOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto subOperands = unpackLLElements(loc, operand, rewriter);
      unpackedOperands.push_back(
          unpackI32(subOperands, argTy, rewriter, loc, getTypeConverter()));
    }
    if (unpackedOperands.empty())
      unpackedOperands.push_back({});

    // Although we ensure that all operands and results to this op have the same
    // encoding, MMA layouts have a different physical ordering depending on the
    // bit width of the underlying element.
    //
    // Thus if the inputs to the inline asm op are MMA with different widths, we
    // need to reorder them so we iterate over the operands' elements in the
    // same logical order.
    for (unsigned i = 1; i < unpackedOperands.size(); ++i) {
      unpackedOperands[i] = reorderValues(
          unpackedOperands[i], /*inType=*/op->getOperand(i).getType(),
          /*ouType=*/op->getResult(0).getType());
    }

    // Number of (unpacked) elements to process per operand.  Normally this
    // equals the number of output elements per return value, except when the
    // asm has no inputs, in which case there's 1 output element.
    size_t numInputElems = unpackedOperands[0].size();

    // These are checked by the verifier, so we don't need to raise a nice
    // error.
    assert(all_of(unpackedOperands, [&](auto &operands) {
      return operands.size() == numInputElems;
    }));
    assert(numInputElems % op.getPackedElement() == 0);

    // Run the inline asm op on each block of elements.
    //
    // Layout is unpackedResults[result_idx][elem].
    //
    // This loop always runs at least once, even when the asm has no input
    // elements.
    SmallVector<SmallVector<Value>> unpackedResults(op->getNumResults());
    for (unsigned i = 0; i < std::max(numInputElems, size_t{1});
         i += op.getPackedElement()) {
      // Block of elements to process with one call to the inline asm.  This is
      // ordered opposite `unpackedResults`: The outer dim is
      // op.getPackedElement(), and the inner dim is the operand.
      SmallVector<SmallVector<Value>> block(op.getPackedElement());
      if (numInputElems > 0) {
        for (auto &os : unpackedOperands) {
          for (int j = 0; j < op.getPackedElement(); j++) {
            block[j].push_back(os[i + j]);
          }
        }
      }
      auto cur = createDestOps(op, adaptor, rewriter, block, loc);
      assert(cur.size() == unpackedResults.size());
      for (unsigned j = 0; j < cur.size(); j++) {
        unpackedResults[j].insert(unpackedResults[j].end(), cur[j].begin(),
                                  cur[j].end());
      }
    }

    // Reorder and pack the results.
    SmallVector<Value> outs;
    for (int i = 0; i < unpackedResults.size(); i++) {
      // We reordered all the inputs so they match operand 0.  Reorder the
      // outputs accordingly.
      if (op->getNumOperands() > 0) {
        unpackedResults[i] = reorderValues(
            unpackedResults[i], /*inType=*/op->getOperand(0).getType(),
            /*ouType=*/op->getResult(i).getType());
      }
      auto packed = packI32(unpackedResults[i], op->getResult(i).getType(),
                            rewriter, loc, getTypeConverter());
      outs.push_back(packLLElements(loc, getTypeConverter(), unpackedResults[i],
                                    rewriter, op->getResult(i).getType()));
    }

    rewriter.replaceOp(op, outs);
    return success();
  }
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    PTXBuilder ptxBuilder;
    auto &fdiv = *ptxBuilder.create<PTXInstr>("div");
    unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
    if (32 == bitwidth) {
      fdiv.o("full").o("f32");
    } else if (64 == bitwidth) {
      fdiv.o("rn").o("f64");
    } else {
      llvm::report_fatal_error("Unsupported bitwidth");
    }

    auto res = ptxBuilder.newOperand(bitwidth == 32 ? "=r" : "=l");
    auto lhs =
        ptxBuilder.newOperand(operands[0][0], bitwidth == 32 ? "r" : "l");
    auto rhs =
        ptxBuilder.newOperand(operands[0][1], bitwidth == 32 ? "r" : "l");
    fdiv(res, lhs, rhs);

    Value ret = ptxBuilder.launch(rewriter, loc, elemTy, false);
    return {ret};
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<mlir::arith::MulFOp, FMulOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::MulFOp, FMulOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::MulFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      PTXBuilder builder;
      auto ptxAsm = " { .reg .b16 c;        \n"
                    "    mov.b16 c, 0x8000U; \n" // 0.0
                    "    fma.rn.bf16 $0, $1, $2, c; } \n";
      auto &fMul = *builder.create<PTXInstr>(ptxAsm);
      auto res = builder.newOperand("=h");
      auto lhs = builder.newOperand(operands[0][0], "h");
      auto rhs = builder.newOperand(operands[0][1], "h");
      fMul({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
      return {builder.launch(rewriter, loc, i16_ty, false)};
    } else {
      return {rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct FAddOpConversion
    : ElementwiseOpConversionBase<mlir::arith::AddFOp, FAddOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::AddFOp, FAddOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::AddFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      PTXBuilder builder;
      auto ptxAsm = "{ .reg .b16 c;         \n"
                    "   mov.b16 c, 0x3f80U; \n" // 1.0
                    "   fma.rn.bf16 $0, $1, c, $2; } \n";
      auto &fAdd = *builder.create<PTXInstr>(ptxAsm);
      auto res = builder.newOperand("=h");
      auto lhs = builder.newOperand(operands[0][0], "h");
      auto rhs = builder.newOperand(operands[0][1], "h");
      fAdd({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
      return {builder.launch(rewriter, loc, i16_ty, false)};
    } else {
      return {rewriter.create<LLVM::FAddOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct FSubOpConversion
    : ElementwiseOpConversionBase<mlir::arith::SubFOp, FSubOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::SubFOp, FSubOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::SubFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      PTXBuilder builder;
      auto ptxAsm = " { .reg .b16 c;         \n"
                    "    mov.b16 c, 0xbf80U; \n" // -1.0
                    "    fma.rn.bf16 $0, $2, c, $1;} \n";
      auto &fSub = *builder.create<PTXInstr>(ptxAsm);
      auto res = builder.newOperand("=h");
      auto lhs = builder.newOperand(operands[0][0], "h");
      auto rhs = builder.newOperand(operands[0][1], "h");
      fSub({res, lhs, rhs}, /*onlyAttachMLIRArgs=*/true);
      return {builder.launch(rewriter, loc, i16_ty, false)};
    } else {
      return {rewriter.create<LLVM::FSubOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

// Uses inline ptx to convert s8/u8 to bf16, since the
struct SIToFPOpConversion
    : ElementwiseOpConversionBase<mlir::arith::SIToFPOp, SIToFPOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::SIToFPOp, SIToFPOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      auto cvtFunc = makeConverterFromPtx(
          S8_to_Bf16, getTypeConverter()->convertType(inElemTy),
          getTypeConverter()->convertType(outElemTy));
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      auto outVals = cvtFunc(loc, rewriter, inVals);
      assert(outVals.size() == 4);
      return outVals;
    } else if (outElemTy.isBF16()) {
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0][0]);
      return {FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, value,
                                                    RoundingMode::RTNE)};
    } else {
      return {rewriter.create<LLVM::SIToFPOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<mlir::arith::FPToSIOp, FPToSIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto value =
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, value)};
    } else {
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<mlir::arith::ExtFOp, ExtFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::ExtFOp, ExtFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return {
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0])};
    } else {
      return {rewriter.create<LLVM::FPExtOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<mlir::arith::TruncFOp, TruncFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::TruncFOp, TruncFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto inElemTy = getElementType(op.getIn());
      assert(inElemTy.isF32() && "unsupported conversion");
      return {// Trunc uses the default rounding mode: RTNE
              FpToFpOpConversion::convertFp32ToBf16(
                  loc, rewriter, operands[0][0], RoundingMode::RTNE)};
    } else {
      return {rewriter.create<LLVM::FPTruncOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // For non-FP32 input, call __nv_expf for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = fmul(f32_ty, operands[0][0], f32_val(log2e));

    PTXBuilder ptxBuilder;
    auto &exp2 = ptxBuilder.create<PTXInstr>("ex2")->o("approx").o("f32");
    auto output = ptxBuilder.newOperand("=f");
    auto input = ptxBuilder.newOperand(prod, "f");
    exp2(output, input);
    return {ptxBuilder.launch(rewriter, loc, f32_ty, false)};
  }
};

struct AbsIOpConversion
    : ElementwiseOpConversionBase<mlir::math::AbsIOp, AbsIOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::AbsIOp, AbsIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::math::AbsIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto boolFalse = rewriter.getBoolAttr(false);
    auto constFalse = rewriter.create<LLVM::ConstantOp>(loc, boolFalse);
    return {rewriter.create<LLVM::AbsOp>(loc, elemTy, operands[0][0],
                                         /*is_int_min_poison=*/constFalse)};
  }
};

struct AbsFOpConversion
    : ElementwiseOpConversionBase<mlir::math::AbsFOp, AbsFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::math::AbsFOp, AbsFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::math::AbsFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (llvm::isa<IntegerType>(elemTy)) {
      // Mask out the sign bit
      auto num_bits =
          getElementTypeOrSelf(op.getType()).getIntOrFloatBitWidth();
      assert(num_bits <= 16);
      auto mask = (1u << (num_bits - 1u)) - 1u;
      auto maskAttr = rewriter.getIntegerAttr(elemTy, mask);
      auto maskConst = rewriter.create<LLVM::ConstantOp>(loc, maskAttr);
      return {and_(operands[0][0], maskConst)};
    }

    return {rewriter.create<LLVM::FAbsOp>(loc, elemTy, operands[0][0])};
  }
};

template <typename OpTy>
struct MinMaxFOpConversion
    : ElementwiseOpConversionBase<OpTy, MinMaxFOpConversion<OpTy>> {
  using Base = ElementwiseOpConversionBase<OpTy, MinMaxFOpConversion<OpTy>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  static_assert(std::is_same<OpTy, arith::MinimumFOp>::value ||
                    std::is_same<OpTy, arith::MaximumFOp>::value,
                "OpTy must be arith::MinimumFOp or arith::MaximumFOp");

  // Choose the destination op based on the OpTy.
  using DestOpNanProp =
      typename std::conditional<std::is_same<OpTy, arith::MinimumFOp>::value,
                                LLVM::MinimumOp, LLVM::MaximumOp>::type;
  using DestOpNoNanProp =
      typename std::conditional<std::is_same<OpTy, arith::MinimumFOp>::value,
                                LLVM::MinNumOp, LLVM::MaxNumOp>::type;

  explicit MinMaxFOpConversion(LLVMTypeConverter &typeConverter,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               int computeCapability,
                               PatternBenefit benefit = 1)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        computeCapability(computeCapability) {}

  SmallVector<Value> createDestOps(OpTy op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (computeCapability >= 80) {
      return {rewriter.create<DestOpNanProp>(loc, elemTy, operands[0][0],
                                             operands[0][1])};
    }
    // Handle pre-80 compute capability.
    // If any of the operands is NaN, return NaN.
    auto lhs = operands[0][0];
    auto rhs = operands[0][1];
    auto lhsIsNan =
        rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une, lhs, lhs);
    auto rhsIsNan =
        rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une, rhs, rhs);
    auto isNan = rewriter.create<LLVM::OrOp>(loc, lhsIsNan, rhsIsNan);
    auto nonNanRes = rewriter.create<DestOpNoNanProp>(loc, elemTy, lhs, rhs);

    auto nan = LLVM::createNaNConstant(loc, rewriter, elemTy);

    // Select the result based on the isNan flag.
    return {rewriter.create<LLVM::SelectOp>(loc, isNan, nan, nonNanRes)};
  }

private:
  int computeCapability;
};

struct ClampFOpConversion
    : ElementwiseOpConversionBase<mlir::triton::ClampFOp, ClampFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::triton::ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit ClampFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability, PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  SmallVector<Value> createDestOps(mlir::triton::ClampFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    bool xorsignAbsAvailable = (computeCapability >= 90);
    // Pattern matching the sequence of clamp(x, -limit, limit) to generate more
    // efficient PTX code.
    // NOTE: This pattern matching is not general enough, but it is sufficient.
    // We detect only two cases here:
    // 1. where the "-limit" is computed as 0 - limit:
    //   %cst = arith.constant dense<0.000000e+00>
    //   %8 = tt.load %7, %2
    //   %11 = arith.subf %cst, %8
    //   %12 = tt.clamp %5, %11, %8
    // 2. where "-limit" and "limit" are constants.
    //   %cst_6 = arith.constant dense<-6.0000e+00>
    //   %cst_7 = arith.constant dense<6.0000e+00>
    //   %160 = tt.clamp %158, %cst_6, %cst_7
    bool clipPatternFound = false;

    auto getSplatInitializer = [](Value v) -> std::optional<double> {
      if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto attr = constOp.getValueAttr()
                            .dyn_cast<mlir::DenseIntOrFPElementsAttr>()) {
          if (attr.isSplat()) {
            return attr.getSplatValue<APFloat>().convertToDouble();
          }
        }
      }
      return std::nullopt;
    };

    if (xorsignAbsAvailable) {
      if (auto subOp = op.getOperand(1).getDefiningOp<arith::SubFOp>()) {
        if (subOp.getOperand(1) == op.getOperand(2)) {
          auto initializer = getSplatInitializer(subOp.getOperand(0));
          if (initializer.has_value() && initializer.value() == 0.0) {
            clipPatternFound = true;
          }
        }
      } else {
        auto initializer1 = getSplatInitializer(op.getOperand(1));
        auto initializer2 = getSplatInitializer(op.getOperand(2));
        if (initializer1.has_value() && initializer2.has_value() &&
            initializer1.value() == -initializer2.value()) {
          clipPatternFound = true;
        }
      }
    }

    assert(elemTy.isF32() || elemTy.isF16());

    if (clipPatternFound) {
      // min.xorsign.abs
      PTXBuilder ptxBuilder;
      bool propNan = (op.getPropagateNan() == mlir::triton::PropagateNan::ALL);
      auto &minXorsign = ptxBuilder.create<PTXInstr>("min")
                             ->o("NaN", propNan)
                             .o("xorsign")
                             .o("abs");
      const char *outType = nullptr;
      const char *inType = nullptr;
      if (elemTy.isF32()) {
        minXorsign.o("f32");
        outType = "=f";
        inType = "f";
      } else if (elemTy.isF16()) {
        minXorsign.o("f16");
        outType = "=h";
        inType = "h";
      }
      auto output = ptxBuilder.newOperand(outType);
      auto inputA = ptxBuilder.newOperand(operands[0][0], inType);
      auto inputB = ptxBuilder.newOperand(operands[0][2], inType);
      minXorsign(output, inputA, inputB);

      return {ptxBuilder.launch(rewriter, loc, elemTy, false)};
    }

    // Clip pattern not found, use min/max.
    if (op.getPropagateNan() == triton::PropagateNan::ALL) {
      if (computeCapability >= 80) {
        auto v = rewriter.create<LLVM::MaximumOp>(loc, elemTy, operands[0][0],
                                                  operands[0][1]);
        return {rewriter.create<LLVM::MinimumOp>(loc, v, operands[0][2])};
      }
      // On pre-80 compute capability, we need to handle NaN propagation
      // manually. We need to check only the first operand for clamp.
      auto lhs = operands[0][0];
      auto isNan = rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une,
                                                 lhs, lhs);
      auto v = rewriter.create<LLVM::MaxNumOp>(loc, elemTy, operands[0][0],
                                               operands[0][1]);
      auto nonNanRes = rewriter.create<LLVM::MinNumOp>(loc, v, operands[0][2]);
      auto nan = LLVM::createNaNConstant(loc, rewriter, elemTy);
      // Select the result based on the isNan flag.
      return {rewriter.create<LLVM::SelectOp>(loc, isNan, nan, nonNanRes)};
    }

    // No NaN propagation.
    assert(op.getPropagateNan() == triton::PropagateNan::NONE);
    auto v = rewriter.create<LLVM::MaxNumOp>(loc, elemTy, operands[0][0],
                                             operands[0][1]);
    return {rewriter.create<LLVM::MinNumOp>(loc, v, operands[0][2])};
  }

private:
  int computeCapability;
};

/// The lowering of index_cast becomes an integer conversion since index
/// becomes an integer.  If the bit width of the source and target integer
/// types is the same, just erase the cast.  If the target type is wider,
/// sign-extend the value, otherwise truncate it.
struct IndexCastOpLowering
    : public ElementwiseOpConversionBase<arith::IndexCastOp,
                                         IndexCastOpLowering> {
  using Base =
      ElementwiseOpConversionBase<arith::IndexCastOp, IndexCastOpLowering>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::IndexCastOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy =
        this->getTypeConverter()->convertType(getElementType(op.getIn()));
    unsigned targetBits = elemTy.getIntOrFloatBitWidth();
    unsigned sourceBits = inElemTy.getIntOrFloatBitWidth();

    if (targetBits == sourceBits)
      return {operands[0][0]};
    if (targetBits < sourceBits)
      return {rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, elemTy,
                                                         operands[0][0])};
    return {
        rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, elemTy, operands[0][0])};
  }
};

struct SelectOpConversion
    : ElementwiseOpConversionBase<mlir::arith::SelectOp, SelectOpConversion> {
  using Base =
      ElementwiseOpConversionBase<mlir::arith::SelectOp, SelectOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(mlir::arith::SelectOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    std::array<Value, 3> llvmOperands;
    if (operands[0].size() == 2) {
      // Case of scalar condition with tensor operands.
      assert(op.getCondition().getType().isInteger(1));
      llvmOperands = {adaptor.getCondition(), operands[0][0], operands[0][1]};
    } else {
      llvmOperands = {operands[0][0], operands[0][1], operands[0][2]};
    }
    return {rewriter.create<LLVM::SelectOp>(
        loc, llvmOperands[1].getType(), llvmOperands,
        adaptor.getAttributes().getValue())};
  }
};

struct AddPtrOpConversion : public ConvertOpToLLVMPattern<triton::AddPtrOp> {
  using ConvertOpToLLVMPattern<triton::AddPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto offsetTy = op.getOffset().getType();
    auto typeConverter = getTypeConverter();
    auto resultTensorTy = resultTy.dyn_cast<RankedTensorType>();
    if (resultTensorTy) {
      unsigned elems = getTotalElemsPerThread(resultTy);
      Type elemTy = typeConverter->convertType(resultTensorTy.getElementType()
                                                   .cast<triton::PointerType>()
                                                   .getPointeeType());
      Type ptrTy = typeConverter->convertType(resultTensorTy.getElementType());
      auto ptrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
      auto offsets = unpackLLElements(loc, adaptor.getOffset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(ptrTy, elemTy, ptrs[i], offsets[i]);
      }
      Value view =
          packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(resultTy.isa<triton::PointerType>());
      auto resultPtrTy = typeConverter->convertType(resultTy);
      auto resultElemTy = typeConverter->convertType(
          resultTy.cast<triton::PointerType>().getPointeeType());
      Value result =
          gep(resultPtrTy, resultElemTy, adaptor.getPtr(), adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

} // namespace

void mlir::triton::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    PatternBenefit benefit) {
#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);
  POPULATE_BINARY_OP(arith::SubIOp, LLVM::SubOp) // -
  POPULATE_BINARY_OP(arith::AddIOp, LLVM::AddOp) // +
  POPULATE_BINARY_OP(arith::MulIOp, LLVM::MulOp) // *
  POPULATE_BINARY_OP(arith::DivSIOp, LLVM::SDivOp)
  POPULATE_BINARY_OP(arith::DivUIOp, LLVM::UDivOp)
  POPULATE_BINARY_OP(arith::RemFOp, LLVM::FRemOp) // %
  POPULATE_BINARY_OP(arith::RemSIOp, LLVM::SRemOp)
  POPULATE_BINARY_OP(arith::RemUIOp, LLVM::URemOp)
  POPULATE_BINARY_OP(arith::AndIOp, LLVM::AndOp)   // &
  POPULATE_BINARY_OP(arith::OrIOp, LLVM::OrOp)     // |
  POPULATE_BINARY_OP(arith::XOrIOp, LLVM::XOrOp)   // ^
  POPULATE_BINARY_OP(arith::ShLIOp, LLVM::ShlOp)   // <<
  POPULATE_BINARY_OP(arith::ShRSIOp, LLVM::AShrOp) // >>
  POPULATE_BINARY_OP(arith::ShRUIOp, LLVM::LShrOp) // >>
  POPULATE_BINARY_OP(
      arith::MinNumFOp,
      LLVM::MinNumOp) // fmin (return non-NaN if either op is non-NaN)
  POPULATE_BINARY_OP(
      arith::MaxNumFOp,
      LLVM::MaxNumOp) // fmax (return non-NaN if either op is non-NaN)
  POPULATE_BINARY_OP(arith::MinSIOp, LLVM::SMinOp) // smin
  POPULATE_BINARY_OP(arith::MaxSIOp, LLVM::SMaxOp) // smax
  POPULATE_BINARY_OP(arith::MinUIOp, LLVM::UMinOp) // umin
  POPULATE_BINARY_OP(arith::MaxUIOp, LLVM::UMaxOp) // umax
#undef POPULATE_BINARY_OP

#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);
  POPULATE_UNARY_OP(arith::TruncIOp, LLVM::TruncOp)
  POPULATE_UNARY_OP(arith::ExtSIOp, LLVM::SExtOp)
  POPULATE_UNARY_OP(arith::ExtUIOp, LLVM::ZExtOp)
  POPULATE_UNARY_OP(arith::FPToUIOp, LLVM::FPToUIOp)
  POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)
  POPULATE_UNARY_OP(math::LogOp, math::LogOp)
  POPULATE_UNARY_OP(math::CosOp, math::CosOp)
  POPULATE_UNARY_OP(math::SinOp, math::SinOp)
  POPULATE_UNARY_OP(math::SqrtOp, math::SqrtOp)
  POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)
  POPULATE_UNARY_OP(triton::BitcastOp, LLVM::BitcastOp)
  POPULATE_UNARY_OP(triton::IntToPtrOp, LLVM::IntToPtrOp)
  POPULATE_UNARY_OP(triton::PtrToIntOp, LLVM::PtrToIntOp)
#undef POPULATE_UNARY_OP

  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<AbsIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<AbsFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<CmpIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<CmpFOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<SelectOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<IndexCastOpLowering>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);

  patterns.add<ExternElementwiseOpConversion>(typeConverter, axisInfoAnalysis,
                                              benefit);
  patterns.add<ElementwiseInlineAsmOpConversion>(typeConverter, benefit);
  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);
  patterns.add<MinMaxFOpConversion<arith::MinimumFOp>>(
      typeConverter, axisInfoAnalysis, computeCapability, benefit);
  patterns.add<MinMaxFOpConversion<arith::MaximumFOp>>(
      typeConverter, axisInfoAnalysis, computeCapability, benefit);
}
