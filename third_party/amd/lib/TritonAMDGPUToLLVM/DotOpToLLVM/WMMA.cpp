/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "../PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace AMD {
namespace {

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

enum class WMMAInstrType : uint8_t {
  // D = AB + C;
  // typeof(D) == typeof(C)
  // typeof(A) == typeof(B)
  // typeof(D), typeof(A):
  FP32_FP16,
  FP32_BF16,
  FP16_FP16,
  BF16_BF16,
  INT32_IU8,
  INT32_IU4,
  NOT_APPLICABLE,
};

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

ValueTable getValuesFromDotOperandLayoutStruct(
    ConversionPatternRewriter &rewriter, const LLVMTypeConverter *typeConverter,
    Value value, int n0, int n1, int kWidth, Type type, Location loc) {
  auto elems = unpackLLElements(loc, value, rewriter);
  ValueTable vals;
  Type castedTy = type.isBF16() || type.isF16() ? type : i32_ty;
  int numElemsCasted =
      kWidth * type.getIntOrFloatBitWidth() / castedTy.getIntOrFloatBitWidth();
  for (int i = 0; i < n0; i++) {
    for (int j = 0; j < n1; j++) {
      Type elemTy = typeConverter->convertType(type);
      Type ty = vec_ty(castedTy, numElemsCasted);
      Value castedElems = undef(ty);
      for (int k = 0; k < numElemsCasted; ++k) {
        if (int groupSize = kWidth / numElemsCasted; groupSize > 1) {
          Type groupTy = vec_ty(elemTy, groupSize);
          Value groupedElems = undef(groupTy);
          for (int g = 0; g < groupSize; ++g) {
            groupedElems = insert_element(
                groupTy, groupedElems,
                elems[kWidth * (n1 * i + j) + k * groupSize + g], i32_val(g));
          }
          castedElems = insert_element(
              ty, castedElems, bitcast(groupedElems, castedTy), i32_val(k));
        } else {
          castedElems = insert_element(
              ty, castedElems, elems[kWidth * (n1 * i + j) + k], i32_val(k));
        }
      }
      vals[{i, j}] = castedElems;
    }
  }
  return vals;
}

static WMMAInstrType getWMMAInstrTypeFromDot(DotOp op) {
  auto aOperandTy = op.getA().getType();
  auto aTensorTy = aOperandTy.cast<RankedTensorType>();
  auto aElemTy = aTensorTy.getElementType();
  auto bOperandTy = op.getB().getType();
  auto bTensorTy = bOperandTy.cast<RankedTensorType>();
  auto bElemTy = bTensorTy.getElementType();
  assert(aElemTy == bElemTy);
  auto cOperandTy = op.getC().getType();
  auto cTensorTy = cOperandTy.cast<RankedTensorType>();
  auto cElemTy = cTensorTy.getElementType();
  auto dOperandTy = op.getD().getType();
  auto dTensorTy = dOperandTy.cast<RankedTensorType>();
  auto dElemTy = dTensorTy.getElementType();
  assert(cElemTy == dElemTy);

  if (dElemTy.isF32() && aElemTy.isF16())
    return WMMAInstrType::FP32_FP16;
  if (dElemTy.isF32() && aElemTy.isBF16())
    return WMMAInstrType::FP32_BF16;
  if (dElemTy.isF16() && aElemTy.isF16())
    return WMMAInstrType::FP16_FP16;
  if (dElemTy.isBF16() && aElemTy.isBF16())
    return WMMAInstrType::BF16_BF16;
  if (dElemTy.isInteger(32) && aElemTy.isInteger(8))
    return WMMAInstrType::INT32_IU8;
  if (dElemTy.isInteger(32) && aElemTy.isInteger(4))
    return WMMAInstrType::INT32_IU4;

  return WMMAInstrType::NOT_APPLICABLE;
}

Value generateWMMAOp(ConversionPatternRewriter &rewriter, Location loc,
                     WMMAInstrType wmmaType, Value valA, Value valB,
                     Value valC) {
  auto resType = valC.getType();
  Value falseFlag = int_val(1, false);
  switch (wmmaType) {
  case WMMAInstrType::FP32_FP16:
    return rewriter.create<ROCDL::wmma_f32_16x16x16_f16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC});
  case WMMAInstrType::FP32_BF16:
    return rewriter.create<ROCDL::wmma_f32_16x16x16_bf16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC});
  case WMMAInstrType::FP16_FP16:
    return rewriter.create<ROCDL::wmma_f16_16x16x16_f16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
  case WMMAInstrType::BF16_BF16:
    return rewriter.create<ROCDL::wmma_bf16_16x16x16_bf16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
  case WMMAInstrType::INT32_IU8:
    return rewriter.create<ROCDL::wmma_i32_16x16x16_iu8>(
        loc, TypeRange{resType},
        ValueRange{falseFlag, valA, falseFlag, valB, valC, falseFlag});
  case WMMAInstrType::INT32_IU4:
    return rewriter.create<ROCDL::wmma_i32_16x16x16_iu4>(
        loc, TypeRange{resType},
        ValueRange{falseFlag, valA, falseFlag, valB, valC, falseFlag});
  default:
    llvm::report_fatal_error("WMMA data type not supported");
  }
  return Value();
}

// Conduct the Dot conversion.
LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter,
                         const LLVMTypeConverter *typeConverter) {
  auto wmmaLayout = op.getResult()
                        .getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .cast<AMDWmmaEncodingAttr>();
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  auto mnkDim = AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr();
  auto wmmaInstrType = getWMMAInstrTypeFromDot(op);

  auto loc = op.getLoc();
  Value a = op.getA();
  Value b = op.getB();
  Value d = op.getD();
  auto aTensorTy = a.getType().cast<RankedTensorType>();
  auto bTensorTy = b.getType().cast<RankedTensorType>();
  auto dTensorTy = d.getType().cast<RankedTensorType>();
  auto elemTy = aTensorTy.getElementType();

  auto aEncoding = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  auto bEncoding = bTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  int kWidth = aEncoding.getKWidth();

  auto repA =
      wmmaLayout.getWMMARepForOperands(aTensorTy.getShape(), elemTy, kWidth, 0);
  auto repB =
      wmmaLayout.getWMMARepForOperands(bTensorTy.getShape(), elemTy, kWidth, 1);

  assert(repA[1] == repB[0]);

  Value loadedA = adaptor.getA();
  Value loadedB = adaptor.getB();
  Value loadedC = adaptor.getC();
  auto numRepM = repA[0];
  auto numRepN = repB[1];
  auto numRepK = repA[1];

  ValueTable ha = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedA, numRepM, numRepK, kWidth,
      aTensorTy.getElementType(), loc);
  ValueTable hb = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedB, numRepN, numRepK, kWidth,
      aTensorTy.getElementType(), loc);
  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackLLElements(loc, loadedC, rewriter);

  unsigned warpSize = triton::gpu::getWarpSize(wmmaLayout);
  // TODO get rid of magic numbers
  unsigned vgprElemWidth = 32;
  unsigned paddedOutputElemSize =
      vgprElemWidth / dstElemTy.getIntOrFloatBitWidth();
  // compute number of output elements that each thread holds for one WMMA
  // instruction.
  auto elemsPerVec = mnkDim[0] * mnkDim[1] * paddedOutputElemSize / warpSize;
  auto dElemsToStorePerThread = mnkDim[0] * mnkDim[1] / warpSize;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      Value acc = undef(vecTy);
      for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
        acc = insert_element(vecTy, acc,
                             fc[m * numRepN * dElemsToStorePerThread +
                                n * dElemsToStorePerThread + v],
                             i32_val(v * paddedOutputElemSize));
      }
      for (size_t k = 0; k < numRepK; k++) {
        acc = generateWMMAOp(rewriter, loc, wmmaInstrType, ha[{m, k}],
                             hb[{n, k}], acc);
      }
      for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
        fc[m * numRepN * dElemsToStorePerThread + n * dElemsToStorePerThread +
           v] =
            extract_element(dstElemTy, acc, i32_val(v * paddedOutputElemSize));
      }
    }
  }

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      wmmaLayout.getContext(), SmallVector<Type>(fc.size(), dstElemTy));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

} // namespace

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return tensor.getType().cast<RankedTensorType>();
  };

  assert(rankedTType(op.getA()).getEncoding().isa<DotOperandEncodingAttr>() &&
         rankedTType(op.getB()).getEncoding().isa<DotOperandEncodingAttr>() &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(cTensorTy.getEncoding().isa<AMDWmmaEncodingAttr>() &&
         "Currently, we only support $c with a wmma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  return convertDot(op, adaptor, rewriter, typeConverter);
}
} // namespace AMD
