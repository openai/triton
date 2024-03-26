/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

triton::nvgpu::WGMMAEltType getMmaRetType(Value d) {
  auto dTy = d.getType().cast<RankedTensorType>().getElementType();
  if (dTy.isF32()) {
    return triton::nvgpu::WGMMAEltType::f32;
  } else if (dTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (dTy.isInteger(32)) {
    return triton::nvgpu::WGMMAEltType::s32;
  } else {
    llvm::report_fatal_error("Unsupported mma result type found");
  }
}

triton::nvgpu::WGMMAEltType getMmaOperandType(Value a, bool allowTF32) {
  auto aTy = a.getType().cast<TensorOrMemDesc>().getElementType();
  if (aTy.isF16()) {
    return triton::nvgpu::WGMMAEltType::f16;
  } else if (aTy.isBF16()) {
    return triton::nvgpu::WGMMAEltType::bf16;
  } else if (aTy.isF32() && allowTF32) {
    return triton::nvgpu::WGMMAEltType::tf32;
  } else if (aTy.isInteger(8)) {
    return triton::nvgpu::WGMMAEltType::s8;
  } else if (aTy.isFloat8E5M2()) {
    return triton::nvgpu::WGMMAEltType::e5m2;
  } else if (aTy.isFloat8E4M3FNUZ()) {
    return triton::nvgpu::WGMMAEltType::e4m3;
  } else {
    llvm::report_fatal_error("Unsupported mma operand type found");
  }
}

int64_t getSwizzlingFromLayout(const SharedEncodingAttr &layout,
                               uint32_t widthInByte) {
  int perPhase = layout.getPerPhase();
  int maxPhase = layout.getMaxPhase();
  uint32_t swizzlingByteWidth = 0;
  if (perPhase == 4 && maxPhase == 2) {
    swizzlingByteWidth = 32;
  } else if (perPhase == 2 && maxPhase == 4) {
    swizzlingByteWidth = 64;
  } else if (perPhase == 1 && maxPhase == 8) {
    swizzlingByteWidth = 128;
  } else {
    llvm::report_fatal_error("Unsupported shared layout.");
  }

  // TODO[biaow]: remove it once we support swizzling size larger than matrix
  // width, which requires padding the matrix width to the swizzling size when
  // allocating shared memory.
  assert(swizzlingByteWidth <= widthInByte &&
         "swizzling size larger than matrix width is not supported.");
  return swizzlingByteWidth;
}

static Value createDescriptor(ConversionPatternRewriter &rewriter, Location loc,
                              int64_t swizzling, uint32_t stride) {
  // Create descriptor based on the format described in the spec:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
  union WGMMADescriptor {
    uint64_t descriptor;
    struct {
      uint64_t baseAddress : 14;
      uint64_t : 2;
      uint64_t leadDimensionBaseOffset : 14;
      uint64_t : 2;
      uint64_t strideDimensionBaseOffset : 14;
      uint64_t : 3;
      uint64_t matrixBaseOffset : 3;
      uint64_t : 10;
      uint64_t swizzlingMode : 2;
    };
  };
  static_assert(sizeof(WGMMADescriptor) == 8,
                "Descriptor size should be 64 bits.");
  WGMMADescriptor desc;
  desc.descriptor = 0;
  switch (swizzling) {
  case 0:
    desc.swizzlingMode = 0;
    break;
  case 32:
    desc.swizzlingMode = 3;
    break;
  case 64:
    desc.swizzlingMode = 2;
    break;
  case 128:
    desc.swizzlingMode = 1;
    break;
  default:
    llvm::report_fatal_error("Unsupported swizzling size.");
  }
  desc.strideDimensionBaseOffset = swizzling >> 1;
  desc.leadDimensionBaseOffset = (swizzling * stride) >> 4;
  return int_val(64, desc.descriptor);
}

class DotOpMmaV3SmemLoader {
public:
  DotOpMmaV3SmemLoader() {}
  DotOpMmaV3SmemLoader(Value tensor, Value base, SmallVector<int64_t> shape,
                       Value warpId, unsigned int dimWpt, bool trans,
                       SmallVector<unsigned int> instrShape,
                       ConversionPatternRewriter &rewriter, Location loc)
      : base(base), shape(shape), warpId(warpId), dimWpt(dimWpt), trans(trans),
        instrShape(instrShape) {
    auto ty = tensor.getType().cast<MemDescType>();
    auto sharedLayout = ty.getEncoding().cast<SharedEncodingAttr>();
    ord = sharedLayout.getOrder();
    const int perPhase = sharedLayout.getPerPhase();
    const int maxPhase = sharedLayout.getMaxPhase();
    elemBytes = ty.getElementTypeBitWidth() / 8;
    elemsPerSwizzlingRow = 128 / perPhase / elemBytes;
    elemsPerSwizzlingRowVal = i32_val(elemsPerSwizzlingRow);

    uint32_t widthInByte = shape[ord[0]] * elemBytes;
    int64_t swizzling = getSwizzlingFromLayout(sharedLayout, widthInByte);

    descriptor = createDescriptor(rewriter, loc, swizzling, shape[ord[1]]);
  }

  Value smemLoad(int a, int b, ConversionPatternRewriter &rewriter,
                 Location loc) {
    Value k = i32_val(b * instrShape[1]);
    Value m = add(i32_val(a * dimWpt * instrShape[0]),
                  mul(warpId, i32_val(instrShape[0])));
    if (trans) {
      std::swap(k, m);
    }
    Value leading_offset = mul(udiv(k, elemsPerSwizzlingRowVal),
                               i32_val(shape[ord[1]] * elemsPerSwizzlingRow));
    Value stride_offset = mul(m, elemsPerSwizzlingRowVal);
    Value offset = add(add(leading_offset, stride_offset),
                       urem(k, elemsPerSwizzlingRowVal));
    Value off1 = mul(i32_val(elemBytes), offset);
    Value off_ = zext(i64_ty, udiv(off1, i32_val(16)));

    Value loadDesc = add(descriptor, off_);
    // Add the base at the end to make it easier to do loop invariant code
    // motion.
    loadDesc = add(loadDesc, lshr(shl(ptrtoint(i64_ty, base), int_val(64, 46)),
                                  int_val(64, 50)));
    return loadDesc;
  }

private:
  Value base;
  SmallVector<int64_t> shape;
  Value warpId;
  int dimWpt;
  bool trans;
  Value elemsPerSwizzlingRowVal;
  SmallVector<unsigned int> instrShape;
  ArrayRef<unsigned> ord;
  int elemsPerSwizzlingRow;
  int elemBytes;
  Value descriptor;
};

DotOpMmaV3SmemLoader loadA(const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const NvidiaMmaEncodingAttr &mmaEncoding,
                           Value tensor, Value smemObjBase, Value thread) {
  auto aTy = tensor.getType().cast<TensorOrMemDesc>();
  auto aSharedLayout = aTy.getEncoding().dyn_cast<SharedEncodingAttr>();
  assert(aSharedLayout && "only support load dot operand from shared.");
  auto instrShape = mmaEncoding.getInstrShape();
  auto wpt = mmaEncoding.getWarpsPerCTA();
  auto aOrd = aSharedLayout.getOrder();
  bool transA = aOrd[0] == 0;
  auto shapePerCTA = getShapePerCTA(aTy);

  int numRepM = ceil<unsigned>(shapePerCTA[0], instrShape[0] * wpt[0]);
  int numRepK = ceil<unsigned>(shapePerCTA[1], instrShape[2]);

  // The descriptor should be calculated based on the first warp of the
  // warpgroup.
  Value warp = and_(udiv(thread, i32_val(32)), i32_val(0xFFFFFFFC));
  // Workaround for a bug in ptxas 12.3 that cause a failure in
  // test_core.py::test_dot. The shuffle will force the compiler to treat the
  // value as uniform and prevent wrong optimizations.
  warp = mlir::LLVM::NVIDIA::shuffleIdx(loc, rewriter, warp, 0);
  Value warpM = urem(warp, i32_val(wpt[0]));
  Value warpId = urem(warpM, i32_val(shapePerCTA[0] / instrShape[0]));

  return {tensor,
          smemObjBase,
          shapePerCTA,
          warpId,
          wpt[0],
          transA,
          {instrShape[0], instrShape[2]},
          rewriter,
          loc};
}

DotOpMmaV3SmemLoader loadB(const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Location loc,
                           NvidiaMmaEncodingAttr &mmaEncoding, Value tensor,
                           Value base, Value thread) {
  auto bTy = tensor.getType().cast<MemDescType>();
  auto bSharedLayout = bTy.getEncoding().cast<SharedEncodingAttr>();
  assert(bSharedLayout && "only support load B from shared.");
  auto instrShape = mmaEncoding.getInstrShape();
  auto wpt = mmaEncoding.getWarpsPerCTA();
  auto bOrd = bSharedLayout.getOrder();
  bool transB = bOrd[0] == 1;
  auto shapePerCTA = triton::gpu::getShapePerCTA(bTy);

  int numRepK = ceil<unsigned>(shapePerCTA[0], instrShape[2]);
  int numRepN = ceil<unsigned>(shapePerCTA[1], instrShape[1] * wpt[1]);

  Value warp = and_(udiv(thread, i32_val(32)), i32_val(0xFFFFFFFC));
  Value warpMN = udiv(warp, i32_val(wpt[0]));
  Value warpN = urem(warpMN, i32_val(wpt[1]));
  Value warpId = urem(warpN, i32_val(shapePerCTA[1] / instrShape[1]));

  return {tensor,
          base,
          shapePerCTA,
          warpId,
          wpt[1],
          transB,
          {instrShape[1], instrShape[2]},
          rewriter,
          loc};
}

// Return a vector of Value of the accumulator start at startIndex and pack the
// values into 32bits in case the accumulator is fp16.
llvm::SmallVector<Value> loadReg(ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 const SmallVector<Value> &elements,
                                 int startIndex, int numElements,
                                 Operation *insertBefore) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(insertBefore);

  if (!elements[0].getType().isIntOrFloat() ||
      elements[0].getType().getIntOrFloatBitWidth() >= 32) {
    llvm::SmallVector<Value> mmaOut(numElements);
    for (int i = 0; i < numElements; ++i)
      mmaOut[i] = elements[startIndex + i];
    return mmaOut;
  }
  Type elementType = elements[0].getType();
  int numElemsPer32Bits = 32 / elementType.getIntOrFloatBitWidth();

  // For FP16 and BF16 we need to pack accumulator into 32-bit integers.
  int num32BitValues = numElements / numElemsPer32Bits;
  llvm::SmallVector<Value> mmaOut(num32BitValues);
  Type packTy = vec_ty(elementType, numElemsPer32Bits);
  for (int i = 0; i < num32BitValues; ++i) {
    Value pack = rewriter.create<LLVM::UndefOp>(loc, packTy);
    for (int j = 0; j < numElemsPer32Bits; ++j) {
      Value element = elements[startIndex + i * numElemsPer32Bits + j];
      pack = insert_element(packTy, pack, element, i32_val(j));
    }
    pack = bitcast(pack, rewriter.getIntegerType(32));
    mmaOut[i] = pack;
  }
  return mmaOut;
}

// If the accumulator is fp16 unpack it from 32-bit integers.
SmallVector<Value> unpackAccumulator(ConversionPatternRewriter &rewriter,
                                     Location loc,
                                     const SmallVector<Value> &packed,
                                     RankedTensorType tensorTy) {
  if (!tensorTy.getElementType().isF16())
    return packed;
  // For fp16 the accumulator is pack into 32-bit integers so we need to unpack
  // it.
  SmallVector<Value> results;
  for (Value elem : packed) {
    elem = bitcast(elem, vec_ty(rewriter.getF16Type(), 2));
    results.push_back(extract_element(rewriter.getF16Type(), elem, i32_val(0)));
    results.push_back(extract_element(rewriter.getF16Type(), elem, i32_val(1)));
  }
  return results;
}

static bool isFP8(triton::nvgpu::WGMMAEltType eltType) {
  return eltType == triton::nvgpu::WGMMAEltType::e5m2 ||
         eltType == triton::nvgpu::WGMMAEltType::e4m3;
}

static Value faddAccumulate(ConversionPatternRewriter &rewriter, Location loc,
                            Value a, Value b) {
  int numEl = a.getType().cast<LLVM::LLVMStructType>().getBody().size();
  Value newStruct = rewriter.create<LLVM::UndefOp>(loc, a.getType());
  for (int i = 0; i < numEl; ++i) {
    Value lhs = rewriter.create<LLVM::ExtractValueOp>(loc, a, i);
    Value rhs = rewriter.create<LLVM::ExtractValueOp>(loc, b, i);
    Value add = rewriter.create<LLVM::FAddOp>(loc, lhs, rhs);
    newStruct = rewriter.create<LLVM::InsertValueOp>(loc, newStruct, add, i);
  }
  return newStruct;
}

static bool isZero(Value v) {
  auto constantOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return false;
  if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APFloat>().isZero();
  if (auto denseAttr =
          dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APInt>().isZero();
  return false;
}

static SmallVector<Value> emitWait(ConversionPatternRewriter &rewriter,
                                   Location loc, SmallVector<Value> acc,
                                   int pendings) {
  SmallVector<Type> types(acc.size(), acc[0].getType());
  auto structTy =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
  int i = 0;
  for (Value v : acc) {
    llvmStruct = insert_val(structTy, llvmStruct, v, i++);
  }
  Value res = rewriter.create<triton::nvgpu::WGMMAWaitGroupOp>(loc, llvmStruct,
                                                               pendings);
  SmallVector<Value> results;
  for (int i = 0; i < acc.size(); ++i) {
    results.push_back(extract_val(types[0], res, i));
  }
  return results;
}

LogicalResult convertDot(const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Operation *op, Value a, Value b, Value c, Value d,
                         Value loadedA, Value loadedB, Value loadedC,
                         bool allowTF32, uint32_t maxNumImpreciseAcc, bool sync,
                         Value thread) {
  auto aTensorTy = a.getType().cast<TensorOrMemDesc>();
  auto bTensorTy = b.getType().cast<TensorOrMemDesc>();
  auto dTensorTy = d.getType().cast<RankedTensorType>();
  auto aSharedLayout = aTensorTy.getEncoding().dyn_cast<SharedEncodingAttr>();
  auto bSharedLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto mmaEncoding = dTensorTy.getEncoding().cast<NvidiaMmaEncodingAttr>();
  auto bOrd = bSharedLayout.getOrder();
  bool transA = false;
  Value baseA;
  Value baseB;
  if (aSharedLayout)
    baseA =
        getSharedMemoryObjectFromStruct(
            loc, loadedA,
            typeConverter->convertType(aTensorTy.getElementType()), rewriter)
            .base;
  baseB = getSharedMemoryObjectFromStruct(
              loc, loadedB,
              typeConverter->convertType(bTensorTy.getElementType()), rewriter)
              .base;
  if (aSharedLayout) {
    auto aOrd = aSharedLayout.getOrder();
    transA = aOrd[0] == 0;
  }
  bool transB = bOrd[0] == 1;
  auto dShapePerCTA = getShapePerCTA(dTensorTy);
  auto instrShape = mmaEncoding.getInstrShape();
  auto accSize = 2 * (instrShape[1] / 4);
  int M = 4 * instrShape[0];
  int N = instrShape[1];
  int K = instrShape[2];
  bool zeroAcc = isZero(c);
  auto shapePerCTATile = getShapePerCTATile(mmaEncoding);
  int numRepM = ceil<unsigned>(dShapePerCTA[0], shapePerCTATile[0]);
  int numRepN = ceil<unsigned>(dShapePerCTA[1], shapePerCTATile[1]);
  int numRepK = ceil<unsigned>(aTensorTy.getShape()[1], instrShape[2]);
  DotOpMmaV3SmemLoader aLoader;
  SmallVector<Value> structA;
  if (aSharedLayout) {
    aLoader =
        loadA(typeConverter, rewriter, loc, mmaEncoding, a, baseA, thread);
  } else {
    structA = unpackLLElements(loc, loadedA, rewriter);
  }
  DotOpMmaV3SmemLoader bLoader =
      loadB(typeConverter, rewriter, loc, mmaEncoding, b, baseB, thread);

  auto fc = unpackLLElements(loc, loadedC, rewriter);

  triton::nvgpu::WGMMAEltType eltTypeC = getMmaRetType(d);
  triton::nvgpu::WGMMAEltType eltTypeA = getMmaOperandType(a, allowTF32);
  triton::nvgpu::WGMMAEltType eltTypeB = getMmaOperandType(b, allowTF32);

  triton::nvgpu::WGMMALayout layoutA = transA ? triton::nvgpu::WGMMALayout::col
                                              : triton::nvgpu::WGMMALayout::row;
  triton::nvgpu::WGMMALayout layoutB = transB ? triton::nvgpu::WGMMALayout::row
                                              : triton::nvgpu::WGMMALayout::col;

  auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
  Operation *startSequence = rewriter.create<triton::nvgpu::WGMMAFenceOp>(loc);
  // WGMMA fp8 -> fp32 accumulates in lower precision than fp32.
  bool needsPartialAccumulator = isFP8(eltTypeA) &&
                                 eltTypeC == triton::nvgpu::WGMMAEltType::f32 &&
                                 maxNumImpreciseAcc <= aTensorTy.getShape()[1];
  SmallVector<Value> mmaResults;
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      llvm::SmallVector<Value> mmaOut =
          loadReg(rewriter, loc, fc, (m * numRepN + n) * accSize, accSize,
                  startSequence);
      llvm::SmallVector<Type> elemTypes;
      for (Value accEl : mmaOut)
        elemTypes.push_back(accEl.getType());
      auto accTy =
          LLVM::LLVMStructType::getLiteral(rewriter.getContext(), elemTypes);
      Value d;
      if (!zeroAcc)
        d = packLLElements(loc, typeConverter, mmaOut, rewriter, accTy);
      uint32_t numLowPrecisionAcc = 0;
      Value partialAcc;
      for (int k = 0; k < numRepK; ++k) {
        Value a;
        if (aSharedLayout) {
          a = aLoader.smemLoad(m, k, rewriter, loc);
        } else {
          unsigned regASize = (instrShape[0] * instrShape[2]) / 32;
          llvm::SmallVector<Value> regA =
              loadReg(rewriter, loc, structA, (m * numRepK + k) * regASize,
                      regASize, startSequence);
          auto regATy = LLVM::LLVMStructType::getLiteral(
              rewriter.getContext(),
              SmallVector<Type>(regA.size(), regA[0].getType()));
          a = packLLElements(loc, typeConverter, regA, rewriter, regATy);
        }
        auto b = bLoader.smemLoad(n, k, rewriter, loc);
        numLowPrecisionAcc += K;
        // If using native accumulation would cause use to do more low precion
        // accumulation than allowed do a separate allocation.
        bool requireAddAccumulator =
            needsPartialAccumulator &&
            (numLowPrecisionAcc >= maxNumImpreciseAcc || k == numRepK - 1);
        Value mmaAcc = needsPartialAccumulator ? partialAcc : d;
        mmaAcc = rewriter.create<triton::nvgpu::WGMMAOp>(
            loc, accTy, a, b, mmaAcc, M, N, K, eltTypeC, eltTypeA, eltTypeB,
            layoutA, layoutB);
        if (needsPartialAccumulator)
          partialAcc = mmaAcc;
        else
          d = mmaAcc;
        // If we need accumulate separately to have higher precision, insert
        // adds.
        if (requireAddAccumulator) {
          d = d ? faddAccumulate(rewriter, loc, d, partialAcc) : partialAcc;
          numLowPrecisionAcc = 0;
          partialAcc = Value();
        }
      }
      auto acc = unpackLLElements(loc, d, rewriter);
      for (int i = 0; i < acc.size(); ++i) {
        mmaResults.push_back(acc[i]);
      }
    }
  }
  rewriter.create<triton::nvgpu::WGMMACommitGroupOp>(loc);

  if (sync)
    mmaResults = emitWait(rewriter, loc, mmaResults, 0);

  SmallVector<Value> results =
      unpackAccumulator(rewriter, loc, mmaResults, dTensorTy);

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      mmaEncoding.getContext(),
      SmallVector<Type>(results.size(), dTensorTy.getElementType()));
  auto res = packLLElements(loc, typeConverter, results, rewriter, structTy);
  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult convertWGMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread) {
  auto AEnc = op.getA().getType().getEncoding();
  auto BEnc = op.getB().getType().getEncoding();
  assert(AEnc.isa<SharedEncodingAttr>() || AEnc.isa<DotOperandEncodingAttr>());
  assert(BEnc.isa<SharedEncodingAttr>() &&
         "Operand B should use Shared layout.");
  return convertDot(typeConverter, rewriter, op.getLoc(), op.getOperation(), //
                    op.getA(), op.getB(), op.getC(), op.getD(),              //
                    adaptor.getA(), adaptor.getB(), adaptor.getC(),          //
                    op.getInputPrecision() == InputPrecision::TF32,
                    op.getMaxNumImpreciseAcc(), true, thread);
}

LogicalResult convertAsyncWGMMA(triton::nvidia_gpu::DotAsyncOp op,
                                triton::nvidia_gpu::DotAsyncOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter,
                                Value thread) {
  auto AEnc = op.getA().getType().getEncoding();
  auto BEnc = op.getB().getType().getEncoding();
  assert(AEnc.isa<SharedEncodingAttr>() || AEnc.isa<DotOperandEncodingAttr>());
  assert(BEnc.isa<SharedEncodingAttr>() &&
         "Operand B should use Shared layout.");
  return convertDot(typeConverter, rewriter, op.getLoc(), op.getOperation(), //
                    op.getA(), op.getB(), op.getC(), op.getD(),              //
                    adaptor.getA(), adaptor.getB(), adaptor.getC(),
                    op.getInputPrecision() == InputPrecision::TF32,
                    op.getMaxNumImpreciseAcc(), false, thread);
}
