#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "ConvertLayoutOpToLLVM.h"
#include "LoadStoreOpToLLVM.h"
#include "Utility.h"

#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#include <numeric>

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getCTALayout;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

static CUtensorMapDataType getCUtensorMapDataType(Type ty) {
  if (ty.isF16()) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if (ty.isBF16()) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (ty.isF32()) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else {
    llvm::report_fatal_error("Unsupported elemTy for InsertSliceAsyncV2Op");
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  }
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(ModuleAxisInfoAnalysis &axisAnalysisPass)
      : axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    // The maximum vector size is 128 bits on NVIDIA GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::LoadOp>::ConvertTritonGPUOpToLLVMPattern;

  LoadOpConversion(TritonGPUToLLVMTypeConverter &converter,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getResult().getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    // Get the LLVM values for pointers
    auto ptrElems = getTypeConverter()->unpackLLElements(loc, llPtr, rewriter,
                                                         ptr.getType());
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = getTypeConverter()->unpackLLElements(loc, llMask, rewriter,
                                                       mask.getType());
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && valueElemTy.isa<IntegerType>() &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        constAttr.getElementType().isa<IntegerType>()) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = getTypeConverter()->unpackLLElements(loc, llOther, rewriter,
                                                        other.getType());
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is GEP with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.
      const bool hasL2EvictPolicy = false;

      PTXBuilder ptxBuilder;

      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);

      const std::string readConstraint =
          (width == 64) ? "l" : ((width == 32) ? "r" : "c");
      const std::string writeConstraint =
          (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");

      // prepare asm operands
      auto *dstsOpr = ptxBuilder.newListOperand();
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        auto *opr = ptxBuilder.newOperand(writeConstraint,
                                          /*init=*/true); // =r operations
        dstsOpr->listAppend(opr);
      }

      auto *addrOpr =
          ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

      // Define the instruction opcode
      auto &ld = ptxBuilder.create<>("ld")
                     ->o("volatile", op.getIsVolatile())
                     .global()
                     .o("ca", op.getCache() == triton::CacheModifier::CA)
                     .o("cg", op.getCache() == triton::CacheModifier::CG)
                     .o("L1::evict_first",
                        op.getEvict() == triton::EvictionPolicy::EVICT_FIRST)
                     .o("L1::evict_last",
                        op.getEvict() == triton::EvictionPolicy::EVICT_LAST)
                     .o("L1::cache_hint", hasL2EvictPolicy)
                     .v(nWords)
                     .b(width);

      PTXBuilder::Operand *evictOpr{};

      // Here lack a mlir::Value to bind to this operation, so disabled.
      // if (has_l2_evict_policy)
      //   evictOpr = ptxBuilder.newOperand(l2Evict, "l");

      if (!evictOpr)
        ld(dstsOpr, addrOpr).predicate(pred, "b");
      else
        ld(dstsOpr, addrOpr, evictOpr).predicate(pred, "b");

      if (other) {
        for (size_t ii = 0; ii < nWords; ++ii) {
          // PTX doesn't support mov.u8, so we need to use mov.u16
          PTXInstr &mov =
              ptxBuilder.create<>("mov")->o("u" + std::to_string(movWidth));

          size_t size = width / valueElemNBits;

          auto vecTy = LLVM::getFixedVectorType(valueElemTy, size);
          Value v = undef(vecTy);
          for (size_t s = 0; s < size; ++s) {
            Value falseVal = otherElems[vecStart + ii * size + s];
            Value sVal = createIndexAttrConstant(
                rewriter, loc, this->getTypeConverter()->getIndexType(), s);
            v = insert_element(vecTy, v, falseVal, sVal);
          }
          v = bitcast(v, IntegerType::get(getContext(), width));

          PTXInstr::Operand *opr{};

          if (otherIsSplatConstInt) {
            for (size_t s = 0; s < 32; s += valueElemNBits)
              splatVal |= splatVal << valueElemNBits;
            opr = ptxBuilder.newConstantOperand(splatVal);
          } else
            opr = ptxBuilder.newOperand(v, readConstraint);

          mov(dstsOpr->listGet(ii), opr).predicateNot(pred, "b");
        }
      }

      // Create inline ASM signature
      SmallVector<Type> retTys(nWords, IntegerType::get(getContext(), width));
      Type retTy = retTys.size() > 1
                       ? LLVM::LLVMStructType::getLiteral(getContext(), retTys)
                       : retTys[0];

      // TODO: if (has_l2_evict_policy)
      // auto asmDialectAttr =
      // LLVM::AsmDialectAttr::get(rewriter.getContext(),
      //                                                 LLVM::AsmDialect::AD_ATT);
      Value ret = ptxBuilder.launch(rewriter, loc, retTy);

      // Extract and store return values
      SmallVector<Value> rets;
      for (unsigned int ii = 0; ii < nWords; ++ii) {
        Value curr;
        if (retTy.isa<LLVM::LLVMStructType>()) {
          curr = extract_val(IntegerType::get(getContext(), width), ret, ii);
        } else {
          curr = ret;
        }
        curr = bitcast(curr, LLVM::getFixedVectorType(valueElemTy,
                                                      width / valueElemNBits));
        rets.push_back(curr);
      }
      int tmp = width / valueElemNBits;
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, this->getTypeConverter()->getIndexType(), ii % tmp);
        Value loaded = extract_element(valueElemTy, rets[ii / tmp], vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = getTypeConverter()->packLLElements(
        loc, loadedVals, rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::StoreOp>::ConvertTritonGPUOpToLLVMPattern;

  StoreOpConversion(TritonGPUToLLVMTypeConverter &converter,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = getTypeConverter()->unpackLLElements(loc, llPtr, rewriter,
                                                         ptr.getType());
    auto valueElems = getTypeConverter()->unpackLLElements(
        loc, llValue, rewriter, value.getType());
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    SmallVector<Value> maskElems;
    if (llMask) {
      Value mask = op.getMask();
      maskElems = getTypeConverter()->unpackLLElements(loc, llMask, rewriter,
                                                       mask.getType());
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    Value mask = getMask(valueTy, rewriter, loc);
    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    const int numVecs = elemsPerThread / vec;
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      // TODO: optimization when ptr is AddPtr with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        // llWord is a width-len composition
        Value llWord = undef(wordTy);
        // Insert each value element to the composition
        for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
          const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
          assert(elemOffset < valueElems.size());
          Value elem = valueElems[elemOffset];
          if (elem.getType().isInteger(1))
            elem = sext(i8_ty, elem);
          elem = bitcast(elem, valueElemTy);

          llWord = insert_element(wordTy, llWord, elem, i32_val(elemIdx));
        }
        llWord = bitcast(llWord, valArgTy);
        std::string constraint =
            (width == 64) ? "l" : ((width == 32) ? "r" : "c");
        asmArgs.emplace_back(llWord, constraint);
      }

      // Prepare the PTX inline asm.
      PTXBuilder ptxBuilder;
      auto *asmArgList = ptxBuilder.newListOperand(asmArgs);

      Value maskVal = llMask ? and_(mask, maskElems[vecStart]) : mask;

      auto *asmAddr =
          ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

      auto &ptxStoreInstr =
          ptxBuilder.create<>("st")
              ->global()
              .o("wb", op.getCache() == triton::CacheModifier::WB)
              .o("cg", op.getCache() == triton::CacheModifier::CG)
              .o("cs", op.getCache() == triton::CacheModifier::CS)
              .o("wt", op.getCache() == triton::CacheModifier::WT)
              .o("L1::evict_first",
                 op.getEvict() == triton::EvictionPolicy::EVICT_FIRST)
              .o("L1::evict_last",
                 op.getEvict() == triton::EvictionPolicy::EVICT_LAST)
              .v(nWords)
              .b(width);
      ptxStoreInstr(asmAddr, asmArgList).predicate(maskVal, "b");

      Type boolTy = getTypeConverter()->convertType(rewriter.getIntegerType(1));
      llvm::SmallVector<Type> argTys({boolTy, ptr.getType()});
      argTys.insert(argTys.end(), nWords, valArgTy);

      auto asmReturnTy = void_ty(ctx);

      ptxBuilder.launch(rewriter, loc, asmReturnTy);
    }
    rewriter.eraseOp(op);
    return success();
  }
};
// TODO: refactor to save common logic with insertsliceasyncv2
struct StoreAsyncOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::nvidia_gpu::StoreAsyncOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::StoreAsyncOp>::ConvertTritonGPUOpToLLVMPattern;

  StoreAsyncOpConversion(TritonGPUToLLVMTypeConverter &converter,
                         ModuleAllocation &allocation,
                         mlir::triton::gpu::TMAMetadataTy *tmaMetadata,
                         const TensorPtrMapT *tensorPtrMap,
                         PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::nvidia_gpu::StoreAsyncOp>(
            converter, allocation, tmaMetadata, benefit),
        tensorPtrMap(tensorPtrMap) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::StoreAsyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto srcEncoding = srcTy.getEncoding();
    if (srcEncoding.isa<MmaEncodingAttr>()) {
      return lowerStoreAsyncWithSlice(op, adaptor, rewriter);
    } else {
      return lowerStoreAsync(op, adaptor, rewriter);
    }
  }

  LogicalResult lowerStoreAsync(triton::nvidia_gpu::StoreAsyncOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto dst = op.getDst();
    auto src = op.getSrc();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto elemTy = srcTy.getElementType();

    auto rank = srcTy.getRank();
    // The sotre async op only supports tensor with ranke <= 5.
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-dimension-size-and-format
    assert(rank > 0 && rank <= 5);

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for StoreAsyncOp");

    auto llFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(llFuncOp && "LLVMFuncOp not found for StoreAsyncOp");

    int numTMADescs = getNumTMADescs(llFuncOp);
    assert(numTMADescs > 0);

    auto sharedLayout = srcTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(sharedLayout && "expected shared encoding");

    mlir::triton::gpu::TMAInfo tmaInfo;

    tmaInfo.tensorDataType = getCUtensorMapDataType(elemTy);
    tmaInfo.tensorRank = rank;
    assert(tmaMetadata);

    auto inOrder = sharedLayout.getOrder();
    unsigned TMADescIdx = tmaMetadata->size();
    unsigned numFuncArgs = llFuncOp.getBody().front().getNumArguments();
    auto makeTensorPtr = tensorPtrMap->lookup(op.getOperation());
    auto dstOrder = makeTensorPtr.getOrder();

    unsigned globalAddressArgIdx = getArgIdx(makeTensorPtr.getBase());
    tmaInfo.globalAddressArgIdx = globalAddressArgIdx;
    tmaInfo.TMADescArgIdx = numFuncArgs - numTMADescs + TMADescIdx;

    auto getDimOfOrder = [](ArrayRef<int32_t> order, int32_t i) {
      auto it = std::find(order.begin(), order.end(), i);
      assert(it != order.end());
      return std::distance(order.begin(), it);
    };

    std::vector<int32_t> globalDimsArgIdx;
    std::vector<int32_t> globalStridesArgIdx;
    // constant values are mapped to (-1 - value)
    for (int i = 0; i < rank; ++i) {
      int32_t argIdx = -1;
      auto dim = getDimOfOrder(dstOrder, i);
      argIdx = getArgIdx(makeTensorPtr.getShape()[dim]);
      globalDimsArgIdx.emplace_back(argIdx);
      // handle constant stride
      argIdx = getArgIdx(makeTensorPtr.getStrides()[dim]);
      globalStridesArgIdx.emplace_back(argIdx);
    }

    tmaInfo.globalDimsArgIdx = globalDimsArgIdx;
    tmaInfo.globalStridesArgIdx = globalStridesArgIdx;
    std::vector<uint32_t> boxDims;
    auto CTAsPerCGA = sharedLayout.getCTALayout().getCTAsPerCGA();
    auto CTAOrder = sharedLayout.getCTALayout().getCTAOrder();
    auto CTASplitNum = sharedLayout.getCTALayout().getCTASplitNum();
    auto tensorShape = makeTensorPtr.getResult()
                           .getType()
                           .cast<triton::PointerType>()
                           .getPointeeType()
                           .cast<RankedTensorType>()
                           .getShape();
    auto shapePerCTA = getShapePerCTA(CTASplitNum, tensorShape);
    const uint32_t bytesPerCacheline = 128;
    uint32_t bytesPerElem = elemTy.getIntOrFloatBitWidth() / 8;
    uint32_t numBox{1};
    for (int i = 0; i < rank; ++i) {
      auto dim = getDimOfOrder(dstOrder, i);
      auto tNumElems = shapePerCTA[dim];
      if (i == 0 && tNumElems * bytesPerElem > bytesPerCacheline) {
        tNumElems = bytesPerCacheline / bytesPerElem;
        numBox = (shapePerCTA[dim] + tNumElems - 1) / tNumElems;
      }
      boxDims.emplace_back(tNumElems);
    }
    std::vector<uint32_t> elementStrides(rank, 1);
    tmaInfo.boxDims = boxDims;
    tmaInfo.elementStrides = elementStrides;

    CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    assert(
        ((elemTy.getIntOrFloatBitWidth() == 16 && sharedLayout.getVec() == 8) or
         (elemTy.getIntOrFloatBitWidth() == 32 &&
          sharedLayout.getVec() == 4)) &&
        "Unexpected shared layout for StoreAsyncOp");
    if (sharedLayout.getPerPhase() == 4 && sharedLayout.getMaxPhase() == 2)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
    else if (sharedLayout.getPerPhase() == 2 && sharedLayout.getMaxPhase() == 4)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
    else if (sharedLayout.getPerPhase() == 1 && sharedLayout.getMaxPhase() == 8)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    else
      llvm::report_fatal_error("Unsupported shared layout for StoreAsyncOp");
    tmaInfo.swizzle = swizzle;
    tmaInfo.interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    tmaInfo.l2Promotion =
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    tmaInfo.oobFill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    tmaMetadata->emplace_back(tmaInfo);

    Value llDst = adaptor.getDst();
    Value llSrc = adaptor.getSrc();
    auto srcShape = srcTy.getShape();
    auto smemObj = getSharedMemoryObjectFromStruct(loc, llSrc, rewriter);

    SmallVector<Value> offsetVals;
    for (auto i = 0; i < srcShape.size(); ++i) {
      offsetVals.emplace_back(i32_val(0));
    }

    Value tmaDesc =
        llFuncOp.getBody().front().getArgument(tmaInfo.TMADescArgIdx);
    auto ptrI8SharedTy = LLVM::LLVMPointerType::get(
        typeConverter->convertType(rewriter.getI8Type()), 3);

    auto threadId = getThreadId(rewriter, loc);
    Value pred = icmp_eq(threadId, i32_val(0));

    auto llCoord = getTypeConverter()->unpackLLElements(loc, llDst, rewriter,
                                                        dst.getType());
    uint32_t boxStride = std::accumulate(boxDims.begin(), boxDims.end(), 1,
                                         std::multiplies<uint32_t>());

    Value clusterCTAId = getClusterCTAId(rewriter, loc);
    SmallVector<Value> multiDimClusterCTAId =
        delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

    rewriter.create<triton::nvgpu::FenceAsyncSharedOp>(loc, 0);

    for (uint32_t b = 0; b < numBox; ++b) {
      SmallVector<Value> coord;
      // raw coord
      for (int i = 0; i < rank; ++i) {
        auto dim = getDimOfOrder(dstOrder, i);
        coord.push_back(llCoord[dim]);
      }
      // coord with box and cta offset
      for (int i = 0; i < rank; ++i) {
        auto dim = getDimOfOrder(dstOrder, i);
        if (i == 0) {
          coord[i] = add(coord[i], i32_val(b * boxDims[i]));
          auto CTAOffset =
              mul(multiDimClusterCTAId[dim], i32_val(numBox * boxDims[i]));
          coord[i] = add(coord[i], CTAOffset);
        } else {
          coord[i] = add(coord[i],
                         mul(multiDimClusterCTAId[dim], i32_val(boxDims[i])));
        }
      }
      Value srcOffset = i32_val(b * boxStride);
      auto srcPtrTy = ptr_ty(getTypeConverter()->convertType(elemTy), 3);
      Value srcPtrBase = gep(srcPtrTy, smemObj.base, srcOffset);
      auto addr = bitcast(srcPtrBase, ptrI8SharedTy);
      rewriter.create<triton::nvgpu::TMAStoreTiledOp>(loc, tmaDesc, addr, pred,
                                                      coord);
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  lowerStoreAsyncWithSlice(triton::nvidia_gpu::StoreAsyncOp op,
                           OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto dst = op.getDst();
    auto src = op.getSrc();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto makeTensorPtr = tensorPtrMap->lookup(op.getOperation());
    auto dstTensorTy = makeTensorPtr.getResult()
                           .getType()
                           .cast<triton::PointerType>()
                           .getPointeeType()
                           .cast<RankedTensorType>();
    auto tensorShape = dstTensorTy.getShape();
    auto dstOrder = makeTensorPtr.getOrder();
    auto dstElemTy = dstTensorTy.getElementType();

    auto rank = srcTy.getRank();
    // The sotre async op only supports tensor with ranke <= 5.
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-dimension-size-and-format
    assert(rank > 0 && rank <= 5);

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for StoreAsyncOp");

    auto llFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(llFuncOp && "LLVMFuncOp not found for StoreAsyncOp");

    int numTMADescs = getNumTMADescs(llFuncOp);
    assert(numTMADescs > 0);

    auto ctaLayout = getCTALayout(dstTensorTy.getEncoding());
    // The order of smem should be consistent with gmem.
    SmallVector<unsigned> sharedOrder;
    for (auto o : makeTensorPtr.getOrder()) {
      sharedOrder.emplace_back(o);
    }
    auto sharedLayout = SharedEncodingAttr::get(ctx, tensorShape, sharedOrder,
                                                ctaLayout, dstElemTy);

    mlir::triton::gpu::TMAInfo tmaInfo;

    tmaInfo.tensorDataType = getCUtensorMapDataType(dstElemTy);
    tmaInfo.tensorRank = rank;
    assert(tmaMetadata);

    unsigned TMADescIdx = tmaMetadata->size();
    unsigned numFuncArgs = llFuncOp.getBody().front().getNumArguments();

    unsigned globalAddressArgIdx = getArgIdx(makeTensorPtr.getBase());
    tmaInfo.globalAddressArgIdx = globalAddressArgIdx;
    tmaInfo.TMADescArgIdx = numFuncArgs - numTMADescs + TMADescIdx;

    auto getDimOfOrder = [](ArrayRef<int32_t> order, int32_t i) {
      auto it = std::find(order.begin(), order.end(), i);
      assert(it != order.end());
      return std::distance(order.begin(), it);
    };

    std::vector<int32_t> globalDimsArgIdx;
    std::vector<int32_t> globalStridesArgIdx;
    // constant values are mapped to (-1 - value)
    for (int i = 0; i < rank; ++i) {
      int32_t argIdx = -1;
      auto dim = getDimOfOrder(dstOrder, i);
      argIdx = getArgIdx(makeTensorPtr.getShape()[dim]);
      globalDimsArgIdx.emplace_back(argIdx);
      // handle constant stride
      argIdx = getArgIdx(makeTensorPtr.getStrides()[dim]);
      globalStridesArgIdx.emplace_back(argIdx);
    }

    tmaInfo.globalDimsArgIdx = globalDimsArgIdx;
    tmaInfo.globalStridesArgIdx = globalStridesArgIdx;
    std::vector<uint32_t> boxDims;
    auto CTAsPerCGA = sharedLayout.getCTALayout().getCTAsPerCGA();
    auto CTAOrder = sharedLayout.getCTALayout().getCTAOrder();
    auto CTASplitNum = sharedLayout.getCTALayout().getCTASplitNum();
    auto shapePerCTA = getShapePerCTA(CTASplitNum, tensorShape);

    auto srcLayout = srcTy.getEncoding();
    auto mmaLayout = srcLayout.dyn_cast<MmaEncodingAttr>();

    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);

    auto instrShape = mmaLayout.getInstrShape();
    auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
    uint32_t repM =
        ceil<unsigned>(shapePerCTA[0], instrShape[0] * warpsPerCTA[0]);
    uint32_t numElemsPerRep = numElems / repM;

    const uint32_t bytesPerCacheline = 128;
    uint32_t bytesPerElem = dstElemTy.getIntOrFloatBitWidth() / 8;
    uint32_t numBox{1};
    for (int i = 0; i < rank; ++i) {
      auto dim = getDimOfOrder(dstOrder, i);
      auto tNumElems = shapePerCTA[dim];
      if (i == 0 && tNumElems * bytesPerElem > bytesPerCacheline) {
        tNumElems = bytesPerCacheline / bytesPerElem;
        numBox = (shapePerCTA[dim] + tNumElems - 1) / tNumElems;
      }
      if (i == 1) {
        tNumElems = tNumElems / repM / warpsPerCTA[0];
      }
      boxDims.emplace_back(tNumElems);
    }
    std::vector<uint32_t> elementStrides(rank, 1);
    tmaInfo.boxDims = boxDims;
    tmaInfo.elementStrides = elementStrides;

    CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    assert(((dstElemTy.getIntOrFloatBitWidth() == 16 &&
             sharedLayout.getVec() == 8) or
            (dstElemTy.getIntOrFloatBitWidth() == 32 &&
             sharedLayout.getVec() == 4)) &&
           "Unexpected shared layout for StoreAsyncOp");
    if (sharedLayout.getPerPhase() == 4 && sharedLayout.getMaxPhase() == 2)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
    else if (sharedLayout.getPerPhase() == 2 && sharedLayout.getMaxPhase() == 4)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
    else if (sharedLayout.getPerPhase() == 1 && sharedLayout.getMaxPhase() == 8)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    else
      llvm::report_fatal_error("Unsupported shared layout for StoreAsyncOp");
    tmaInfo.swizzle = swizzle;
    tmaInfo.interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    tmaInfo.l2Promotion =
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    tmaInfo.oobFill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    tmaMetadata->emplace_back(tmaInfo);

    Value llDst = adaptor.getDst();
    Value llSrc = adaptor.getSrc();
    auto srcShape = srcTy.getShape();
    auto dstElemPtrTy = ptr_ty(getTypeConverter()->convertType(dstElemTy), 3);
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    smemBase = bitcast(smemBase, dstElemPtrTy);

    SmallVector<Value> offsetVals;
    for (auto i = 0; i < srcShape.size(); ++i) {
      offsetVals.emplace_back(i32_val(0));
    }

    Value tmaDesc =
        llFuncOp.getBody().front().getArgument(tmaInfo.TMADescArgIdx);
    auto ptrI8SharedTy = LLVM::LLVMPointerType::get(
        typeConverter->convertType(rewriter.getI8Type()), 3);

    auto threadId = getThreadId(rewriter, loc);
    Value pred = int_val(1, 1);

    auto llCoord = getTypeConverter()->unpackLLElements(loc, llDst, rewriter,
                                                        dst.getType());
    uint32_t boxStride = std::accumulate(boxDims.begin(), boxDims.end(), 1,
                                         std::multiplies<uint32_t>());
    boxStride = boxStride * repM * warpsPerCTA[0];

    Value clusterCTAId = getClusterCTAId(rewriter, loc);
    SmallVector<Value> multiDimClusterCTAId =
        delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

    // rowStride in bytes
    uint32_t rowStrideInBytes = shapePerCTA[dstOrder[0]] * bytesPerElem;
    uint32_t swizzlingByteWidth =
        std::min<uint32_t>(rowStrideInBytes, bytesPerCacheline);

    unsigned numElemsPerSwizzlingRow = swizzlingByteWidth / bytesPerElem;
    unsigned leadingDimOffset =
        numElemsPerSwizzlingRow * shapePerCTA[dstOrder[1]];

    uint32_t rowsPerRep = getShapePerCTATile(mmaLayout)[0];

    Value warpId = udiv(threadId, i32_val(32));
    Value warpId0 = urem(urem(warpId, i32_val(warpsPerCTA[0])),
                         i32_val(srcShape[0] / instrShape[0]));
    auto srcOrder = triton::gpu::getOrder(srcLayout);
    unsigned inVec =
        srcOrder == sharedLayout.getOrder()
            ? triton::gpu::getContigPerThread(srcLayout)[srcOrder[0]]
            : 1;
    unsigned outVec = sharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    assert(minVec == 2);

    auto wordTy = vec_ty(dstElemTy, minVec);

    auto inVals = getTypeConverter()->unpackLLElements(loc, adaptor.getSrc(),
                                                       rewriter, srcTy);
    for (uint32_t b = 0; b < numBox; ++b) {
      for (int rep = 0; rep < repM; ++rep) {
        Value rowOfWarp = add(mul(warpId0, i32_val(instrShape[0])),
                              i32_val(rep * rowsPerRep));
        uint32_t elemIdxOffset = rep * numElemsPerRep;

        for (unsigned idx = 0; idx < numElemsPerRep / numBox; idx += 8) {
          uint32_t elemIdx = elemIdxOffset + b * numElemsPerRep / numBox + idx;

          Value offset = rewriter.create<triton::nvgpu::OffsetOfStmatrixV4Op>(
              loc, i32_ty, threadId, rowOfWarp,
              i32_val(b * numElemsPerRep / numBox + idx), leadingDimOffset,
              numElemsPerSwizzlingRow, true);

          Value addr = gep(dstElemPtrTy, smemBase, offset);
          Value words[4];
          for (unsigned i = 0; i < 8; ++i) {
            if (i % minVec == 0)
              words[i / 2] = undef(wordTy);
            words[i / 2] = insert_element(
                wordTy, words[i / 2], inVals[elemIdx + i], i32_val(i % minVec));
          }

          rewriter.create<triton::nvgpu::StoreMatrixOp>(
              loc, bitcast(addr, ptrI8SharedTy),
              ValueRange{bitcast(words[0], i32_ty), bitcast(words[1], i32_ty),
                         bitcast(words[2], i32_ty), bitcast(words[3], i32_ty)});
        }
        rewriter.create<triton::nvgpu::FenceAsyncSharedOp>(loc, 0);

        SmallVector<Value> coord;
        // raw coord
        for (int i = 0; i < rank; ++i) {
          auto dim = getDimOfOrder(dstOrder, i);
          coord.push_back(llCoord[dim]);
        }
        // coord with box and cta offset
        for (int i = 0; i < rank; ++i) {
          auto dim = getDimOfOrder(dstOrder, i);
          if (i == 0) {
            coord[i] = add(coord[i], i32_val(b * boxDims[i]));
            auto CTAOffset =
                mul(multiDimClusterCTAId[dim], i32_val(numBox * boxDims[i]));
            coord[i] = add(coord[i], CTAOffset);
          } else {
            Value blockOffset = i32_val(rep * instrShape[0] * warpsPerCTA[0]);
            Value warpOffset = mul(warpId0, i32_val(instrShape[0]));
            coord[i] = add(add(coord[i], add(blockOffset, warpOffset)),
                           mul(multiDimClusterCTAId[dim],
                               i32_val(boxDims[i] * repM * warpsPerCTA[0])));
          }
        }
        Value srcOffset =
            add(i32_val(b * boxStride + rep * instrShape[0] * warpsPerCTA[0] *
                                            instrShape[1] * warpsPerCTA[1] /
                                            numBox),
                mul(warpId0, i32_val(instrShape[0] * numElemsPerSwizzlingRow)));
        auto srcPtrTy = ptr_ty(getTypeConverter()->convertType(dstElemTy), 3);
        Value srcPtrBase = gep(srcPtrTy, smemBase, srcOffset);
        auto addr = bitcast(srcPtrBase, ptrI8SharedTy);
        rewriter.create<triton::nvgpu::TMAStoreTiledOp>(loc, tmaDesc, addr,
                                                        pred, coord);
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned getArgIdx(Value v) const {
    if (auto op = v.getDefiningOp<mlir::arith::ConstantOp>()) {
      return -1 -
             op.getValue().dyn_cast<IntegerAttr>().getValue().getZExtValue();
    }
    if (v.getDefiningOp() &&
        isa<mlir::UnrealizedConversionCastOp>(v.getDefiningOp())) {
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else if (v.getParentBlock()->isEntryBlock() && v.isa<BlockArgument>()) {
      // in entryblock and is BlockArgument; Because argument of func are
      // arugments of entryblock bb0 in MLIR
      return v.cast<BlockArgument>().getArgNumber();
    } else if (v.getParentBlock()->isEntryBlock() &&
               (!v.isa<BlockArgument>())) {
      // in entryblock but not BlockArgument
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else if (!v.getParentBlock()->isEntryBlock()) {
      // in non-entryblock
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else {
      llvm::report_fatal_error(
          "Operand of `MakeTensorPtrOp` is not the function's argument");
      return 0;
    }
  }

  int getNumTMADescs(LLVM::LLVMFuncOp func) const {
    if (!func->hasAttr(kAttrNumTMALoadDescsName)) {
      llvm::report_fatal_error("TritonGPU module should contain a "
                               "triton_gpu.num-tma-load attribute");
      return -1;
    }
    if (!func->hasAttr(kAttrNumTMAStoreDescsName)) {
      llvm::report_fatal_error("TritonGPU module should contain a "
                               "triton_gpu.num-tma-store attribute");
      return -1;
    }
    return func->getAttr(kAttrNumTMAStoreDescsName)
               .cast<IntegerAttr>()
               .getInt() +
           func->getAttr(kAttrNumTMALoadDescsName).cast<IntegerAttr>().getInt();
  }

  const TensorPtrMapT *tensorPtrMap;
};

namespace {
void createBarrier(ConversionPatternRewriter &rewriter, Location loc,
                   int numCTAs) {
  if (numCTAs == 1) {
    barrier();
  } else {
    rewriter.create<triton::nvidia_gpu::ClusterArriveOp>(loc, false);
    rewriter.create<triton::nvidia_gpu::ClusterWaitOp>(loc);
  }
}

Value broadcastScalar(ConversionPatternRewriter &rewriter, Location loc,
                      int numCTAs, Value smemPtr, Value value, Value mask) {
  smemPtr = bitcast(smemPtr, ptr_ty(value.getType(), 3));
  LLVM::storeShared(rewriter, loc, smemPtr, value, mask);
  createBarrier(rewriter, loc, numCTAs);
  Value ret = LLVM::loadShared(rewriter, loc, smemPtr, /*mask=*/i32_val(true));
  createBarrier(rewriter, loc, numCTAs);
  return ret;
}

} // namespace

struct AtomicCASOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicCASOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicCASOpConversion(TritonGPUToLLVMTypeConverter &converter,
                        ModuleAllocation &allocation,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>(
            converter, allocation, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicCASOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto ptrElements = getTypeConverter()->unpackLLElements(
        loc, llPtr, rewriter, op.getPtr().getType());
    auto cmpElements = getTypeConverter()->unpackLLElements(
        loc, llCmp, rewriter, op.getCmp().getType());
    auto valElements = getTypeConverter()->unpackLLElements(
        loc, llVal, rewriter, op.getVal().getType());

    auto valueTy = op.getResult().getType();
    auto TensorTy = valueTy.dyn_cast<RankedTensorType>();
    Type valueElemTy =
        TensorTy ? getTypeConverter()->convertType(TensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    Value mask = getMask(valueTy, rewriter, loc);

    Value casPtr = ptrElements[0];
    Value casCmp = cmpElements[0];
    Value casVal = valElements[0];

    PTXBuilder ptxBuilderAtomicCAS;
    auto *dstOpr = ptxBuilderAtomicCAS.newOperand("=r", /*init=*/true);
    auto *ptrOpr = ptxBuilderAtomicCAS.newAddrOperand(casPtr, "l");
    auto *cmpOpr = ptxBuilderAtomicCAS.newOperand(casCmp, "r");
    auto *valOpr = ptxBuilderAtomicCAS.newOperand(casVal, "r");
    auto &atom = *ptxBuilderAtomicCAS.create<PTXInstr>("atom");
    std::string semStr;
    llvm::raw_string_ostream os(semStr);
    os << op.getSem();
    atom.global().o(semStr).o("cas").o("b32");
    atom(dstOpr, ptrOpr, cmpOpr, valOpr).predicate(mask);
    auto old = ptxBuilderAtomicCAS.launch(rewriter, loc, valueElemTy);
    createBarrier(rewriter, loc, numCTAs);

    Value smemPtr = getSharedMemoryBase(loc, rewriter, op.getOperation());
    Value ret = broadcastScalar(rewriter, loc, numCTAs, smemPtr, old, mask);
    rewriter.replaceOp(op, {ret});
    return success();
  }
};

struct AtomicLoadOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicLoadOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicLoadOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicLoadOpConversion(TritonGPUToLLVMTypeConverter &converter,
                         ModuleAllocation &allocation,
                         ModuleAxisInfoAnalysis &axisAnalysisPass,
                         PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicLoadOp>(
            converter, allocation, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicCASOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Value llPtr = adaptor.getPtr();

    auto ptrElements = getTypeConverter()->unpackLLElements(
        loc, llPtr, rewriter, op.getPtr().getType());

    auto valueTy = op.getResult().getType();
    Value mask = getMask(valueTy, rewriter, loc);
    Value ptr = ptrElements[0];

    auto sem = op.getSem();
    std::string semStr;
    llvm::raw_string_ostream os(semStr);
    os << sem;

    const unsigned width = std::max(8u, valueTy.getIntOrFloatBitWidth());
    const char *writeConstraint =
        width == 64 ? "=l" : (width == 32 ? "=r" : "=h");
    PTXBuilder ptxBuilder;
    auto *dstOpr = ptxBuilder.newOperand(writeConstraint);
    auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "l");
    auto &ld = ptxBuilder.create<>("ld")->global().o(semStr).o("gpu").b(width);

    ld(dstOpr, ptrOpr).predicate(mask, "b");
    Value load = ptxBuilder.launch(rewriter, loc, valueTy);

    Value smemPtr = getSharedMemoryBase(loc, rewriter, op.getOperation());
    Value ret = broadcastScalar(rewriter, loc, numCTAs, smemPtr, load, mask);

    rewriter.replaceOp(op, {ret});
    return success();
  }
};

struct AtomicStoreOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicStoreOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicStoreOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicStoreOpConversion(TritonGPUToLLVMTypeConverter &converter,
                          ModuleAllocation &allocation,
                          ModuleAxisInfoAnalysis &axisAnalysisPass,
                          PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicStoreOp>(
            converter, allocation, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicCASOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Value ptr = adaptor.getPtr();
    Value val = adaptor.getValue();

    auto valueTy = op.getValue().getType();
    Value mask = getMask(valueTy, rewriter, loc);

    auto sem = op.getSem();
    std::string semStr;
    llvm::raw_string_ostream os(semStr);
    os << sem;

    if (sem != MemSemantic::RELAXED) {
      // Establish causality order between all threads
      createBarrier(rewriter, loc, numCTAs);
    }

    const unsigned width = std::max(8u, valueTy.getIntOrFloatBitWidth());
    const char *constraint = width == 64 ? "l" : (width == 32 ? "r" : "h");
    PTXBuilder ptxBuilder;
    auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "l");
    auto *valOpr = ptxBuilder.newOperand(val, constraint);
    auto &st = ptxBuilder.create<>("st")->global().o(semStr).o("gpu").b(width);
    st(ptrOpr, valOpr).predicate(mask);
    ptxBuilder.launch(rewriter, loc, void_ty(ctx));

    rewriter.eraseOp(op);
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicRMWOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicRMWOpConversion(TritonGPUToLLVMTypeConverter &converter,
                        ModuleAllocation &allocation,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>(
            converter, allocation, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicRMWOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value val = op.getVal();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = getTypeConverter()->unpackLLElements(
        loc, llVal, rewriter, val.getType());
    auto ptrElements = getTypeConverter()->unpackLLElements(
        loc, llPtr, rewriter, ptr.getType());
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = getTypeConverter()->unpackLLElements(
          loc, llMask, rewriter, op.getMask().getType());

    auto valueTy = op.getResult().getType();
    auto tensorTy = valueTy.dyn_cast<RankedTensorType>();
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    const size_t valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // vec = 1, numElements = 1 for scalar
    auto vec = getVectorSize(ptr);
    int numElems = 1;
    // tensor
    if (tensorTy) {
      auto valTy = val.getType().cast<RankedTensorType>();
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
      // mask
      numElems = tensorTy.getNumElements();
    }
    Value mask = getMask(valueTy, rewriter, loc);

    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwVal = undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        rmwVal = insert_element(vecTy, rmwVal, valElements[i + ii], iiVal);
      }

      Value rmwPtr = ptrElements[i];
      Value rmwMask = llMask ? and_(mask, maskElements[i]) : mask;
      std::string sTy;
      PTXBuilder ptxBuilderAtomicRMW;
      std::string tyId = valueElemNBits * vec == 64
                             ? "l"
                             : (valueElemNBits * vec == 32 ? "r" : "h");
      auto *dstOpr = ptxBuilderAtomicRMW.newOperand("=" + tyId, /*init=*/true);
      auto *ptrOpr = ptxBuilderAtomicRMW.newAddrOperand(rmwPtr, "l");
      auto *valOpr = ptxBuilderAtomicRMW.newOperand(rmwVal, tyId);

      auto &atom = ptxBuilderAtomicRMW.create<>("atom")->global().o("gpu");
      auto rmwOp = stringifyRMWOp(atomicRmwAttr).str();
      auto sBits = std::to_string(valueElemNBits);
      switch (atomicRmwAttr) {
      case RMWOp::AND:
        sTy = "b" + sBits;
        break;
      case RMWOp::OR:
        sTy = "b" + sBits;
        break;
      case RMWOp::XOR:
        sTy = "b" + sBits;
        break;
      case RMWOp::ADD:
        sTy = "u" + sBits;
        break;
      case RMWOp::FADD:
        rmwOp = "add";
        rmwOp += (valueElemNBits == 16 ? ".noftz" : "");
        sTy = "f" + sBits;
        sTy += (vec == 2 && valueElemNBits == 16) ? "x2" : "";
        break;
      case RMWOp::MAX:
        sTy = "s" + sBits;
        break;
      case RMWOp::MIN:
        sTy = "s" + sBits;
        break;
      case RMWOp::UMAX:
        rmwOp = "max";
        sTy = "u" + sBits;
        break;
      case RMWOp::UMIN:
        rmwOp = "min";
        sTy = "u" + sBits;
        break;
      case RMWOp::XCHG:
        sTy = "b" + sBits;
        break;
      default:
        return failure();
      }
      std::string semStr;
      llvm::raw_string_ostream os(semStr);
      os << op.getSem();
      atom.o(semStr).o(rmwOp).o(sTy);
      if (tensorTy) {
        atom(dstOpr, ptrOpr, valOpr).predicate(rmwMask);
        auto retType = vec == 1 ? valueElemTy : vecTy;
        auto ret = ptxBuilderAtomicRMW.launch(rewriter, loc, retType);
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
        }
      } else {
        auto ASMReturnTy = void_ty(ctx);
        atom(dstOpr, ptrOpr, valOpr).predicate(rmwMask);
        auto old = ptxBuilderAtomicRMW.launch(rewriter, loc, valueElemTy);
        if (op->user_begin() == op->user_end()) {
          rewriter.replaceOp(op, {old});
          return success();
        }
        Value smemPtr = getSharedMemoryBase(loc, rewriter, op.getOperation());
        // Only threads with rmwMask = True store the result
        Value ret =
            broadcastScalar(rewriter, loc, numCTAs, smemPtr, old, rmwMask);
        rewriter.replaceOp(op, {ret});
      }
    }
    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = getTypeConverter()->packLLElements(
          loc, resultVals, rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct InsertSliceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<tensor::InsertSliceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      tensor::InsertSliceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = insert_slice %src into %dst[%offsets]
    Location loc = op->getLoc();
    Value dst = op.getDest();
    Value src = op.getSource();
    Value res = op.getResult();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    auto *funcAllocation = allocation->getFuncData(funcOp);
    assert(funcAllocation->getBufferId(res) == Allocation::InvalidBufferId &&
           "Only support in-place insert_slice for now");

    auto srcTy = src.getType().dyn_cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<BlockedEncodingAttr>();
    auto srcShape = srcTy.getShape();
    assert(srcLayout && "Unexpected srcLayout in InsertSliceOpConversion");

    auto dstTy = dst.getType().dyn_cast<RankedTensorType>();
    auto dstLayout = dstTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    auto llDst = adaptor.getDest();
    assert(dstLayout && "Unexpected dstLayout in InsertSliceOpConversion");
    assert(op.hasUnitStride() &&
           "Only unit stride supported by InsertSliceOpConversion");

    // newBase = base + offset
    // Triton support either static and dynamic offsets
    auto smemObj = getSharedMemoryObjectFromStruct(loc, llDst, rewriter);
    SmallVector<Value, 4> offsets;
    SmallVector<Value, 4> srcStrides;
    auto mixedOffsets = op.getMixedOffsets();
    for (auto i = 0; i < mixedOffsets.size(); ++i) {
      if (op.isDynamicOffset(i)) {
        offsets.emplace_back(adaptor.getOffsets()[i]);
      } else {
        offsets.emplace_back(i32_val(op.getStaticOffset(i)));
      }
      // Like insert_slice_async, we only support slice from one dimension,
      // which has a slice size of 1
      if (op.getStaticSize(i) != 1) {
        srcStrides.emplace_back(smemObj.strides[i]);
      }
    }

    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, offsets, smemObj.strides);
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto elemPtrTy = ptr_ty(elemTy, 3);
    auto smemBase = gep(elemPtrTy, smemObj.base, offset);

    auto llSrc = adaptor.getSource();
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy);
    storeDistributedToShared(src, llSrc, srcStrides, srcIndices, dst, smemBase,
                             elemTy, loc, rewriter);
    // Barrier is not necessary.
    // The membar pass knows that it writes to shared memory and will handle it
    // properly.
    rewriter.replaceOp(op, llDst);
    return success();
  }
};

struct InsertSliceAsyncOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::InsertSliceAsyncOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::InsertSliceAsyncOp>::ConvertTritonGPUOpToLLVMPattern;

  InsertSliceAsyncOpConversion(
      TritonGPUToLLVMTypeConverter &converter, ModuleAllocation &allocation,
      ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::InsertSliceAsyncOp>(
            converter, allocation, indexCacheInfo, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::InsertSliceAsyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // insert_slice_async %src, %dst, %index, %mask, %other
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getDst();
    Value res = op.getResult();
    Value mask = op.getMask();
    Value other = op.getOther();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    auto *funcAllocation = allocation->getFuncData(funcOp);
    assert(funcAllocation->getBufferId(res) == Allocation::InvalidBufferId &&
           "Only support in-place insert_slice_async for now");

    auto srcTy = src.getType().cast<RankedTensorType>();
    auto resTy = dst.getType().cast<RankedTensorType>();
    auto resElemTy = getTypeConverter()->convertType(resTy.getElementType());
    auto srcBlockedLayout = srcTy.getEncoding().cast<BlockedEncodingAttr>();
    auto resSharedLayout = resTy.getEncoding().cast<SharedEncodingAttr>();
    auto srcShape = srcTy.getShape();
    assert(srcShape.size() == 2 &&
           "insert_slice_async: Unexpected rank of %src");

    Value llDst = adaptor.getDst();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();
    Value llIndex = adaptor.getIndex();

    // %src
    auto srcElems = getTypeConverter()->unpackLLElements(loc, llSrc, rewriter,
                                                         src.getType());

    // %dst
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    auto smemObj = getSharedMemoryObjectFromStruct(loc, llDst, rewriter);
    auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
    SmallVector<Value, 4> offsetVals;
    SmallVector<Value, 4> srcStrides;
    for (auto i = 0; i < dstShape.size(); ++i) {
      if (i == axis) {
        offsetVals.emplace_back(llIndex);
      } else {
        offsetVals.emplace_back(i32_val(0));
        srcStrides.emplace_back(smemObj.strides[i]);
      }
    }
    // Compute the offset based on the original dimensions of the shared
    // memory object
    auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
    auto dstPtrTy = ptr_ty(resElemTy, 3);
    Value dstPtrBase = gep(dstPtrTy, smemObj.base, dstOffset);

    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = getTypeConverter()->unpackLLElements(loc, llMask, rewriter,
                                                       mask.getType());
      assert(srcElems.size() == maskElems.size());
    }

    // %other
    SmallVector<Value> otherElems;
    if (llOther) {
      // FIXME(Keren): always assume other is 0 for now
      // It's not necessary for now because the pipeline pass will skip
      // generating insert_slice_async if the load op has any "other" tensor.
      // assert(false && "insert_slice_async: Other value not supported yet");
      otherElems = getTypeConverter()->unpackLLElements(loc, llOther, rewriter,
                                                        other.getType());
      assert(srcElems.size() == otherElems.size());
    }

    // We don't use getVec() here because we are copying from memory to memory.
    // If contiguity > vector size, we can have one pointer maintaining the
    // start of the vector and the other pointer moving to the next vector.
    unsigned inVec = getContiguity(src);
    unsigned outVec = resSharedLayout.getVec();
    unsigned minVec = inVec;
    if (outVec > 1)
      minVec = std::min(outVec, inVec);
    unsigned numElems = getTotalElemsPerThread(srcTy);
    unsigned perPhase = resSharedLayout.getPerPhase();
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    auto sizePerThread = srcBlockedLayout.getSizePerThread();
    auto threadsPerCTA = getThreadsPerCTA(srcBlockedLayout);
    auto inOrder = srcBlockedLayout.getOrder();
    DenseMap<unsigned, Value> sharedPtrs =
        getSwizzledSharedPtrs(loc, inVec, srcTy, resSharedLayout, resElemTy,
                              smemObj, rewriter, offsetVals, srcStrides);

    // If perPhase * maxPhase > threadsPerCTA, we will have elements
    // that share the same tile indices. The index calculation will
    // be cached.
    auto numSwizzleRows = std::max<unsigned>(
        (perPhase * maxPhase) / threadsPerCTA[inOrder[1]], 1);
    // A sharedLayout encoding has a "vec" parameter.
    // On the column dimension, if inVec > outVec, it means we have to divide
    // single vector read into multiple ones
    auto numVecCols = std::max<unsigned>(inVec / outVec, 1);

    auto srcIndices = emitIndices(loc, rewriter, srcBlockedLayout, srcTy);

    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      // 16 * 8 = 128bits
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto numWords = vecBitWidth / bitWidth;
      auto numWordElems = bitWidth / resElemTy.getIntOrFloatBitWidth();

      // Tune CG and CA here.
      auto byteWidth = bitWidth / 8;
      CacheModifier srcCacheModifier =
          byteWidth == 16 ? CacheModifier::CG : CacheModifier::CA;
      assert(byteWidth == 16 || byteWidth == 8 || byteWidth == 4);
      auto resByteWidth = resElemTy.getIntOrFloatBitWidth() / 8;

      Value basePtr = sharedPtrs[elemIdx];
      for (size_t wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        PTXBuilder ptxBuilder;
        auto wordElemIdx = wordIdx * numWordElems;
        auto &copyAsyncOp =
            *ptxBuilder.create<PTXCpAsyncLoadInstr>(srcCacheModifier);
        auto *dstOperand =
            ptxBuilder.newAddrOperand(basePtr, "r", wordElemIdx * resByteWidth);
        auto *srcOperand =
            ptxBuilder.newAddrOperand(srcElems[elemIdx + wordElemIdx], "l");
        auto *copySize = ptxBuilder.newConstantOperand(byteWidth);
        auto *srcSize = copySize;
        if (op.getMask()) {
          // We don't use predicate in this case, setting src-size to 0
          // if there's any mask. cp.async will automatically fill the
          // remaining slots with 0 if cp-size > src-size.
          // XXX(Keren): Always assume other = 0 for now.
          auto selectOp = select(maskElems[elemIdx + wordElemIdx],
                                 i32_val(byteWidth), i32_val(0));
          srcSize = ptxBuilder.newOperand(selectOp, "r");
        }
        copyAsyncOp(dstOperand, srcOperand, copySize, srcSize);
        ptxBuilder.launch(rewriter, loc, void_ty(getContext()));
      }
    }

    rewriter.replaceOp(op, llDst);
    return success();
  }
};

struct InsertSliceAsyncV2OpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::nvidia_gpu::InsertSliceAsyncV2Op> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::InsertSliceAsyncV2Op>::
      ConvertTritonGPUOpToLLVMPattern;

  InsertSliceAsyncV2OpConversion(TritonGPUToLLVMTypeConverter &converter,

                                 ModuleAllocation &allocation,
                                 mlir::triton::gpu::TMAMetadataTy *tmaMetadata,
                                 const TensorPtrMapT *tensorPtrMap,
                                 PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<
            triton::nvidia_gpu::InsertSliceAsyncV2Op>(converter, allocation,
                                                      tmaMetadata, benefit),
        tensorPtrMap(tensorPtrMap) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InsertSliceAsyncV2Op op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    auto resultTy = op.getResult().getType().cast<RankedTensorType>();
    auto elemTy = resultTy.getElementType();
    auto rank = resultTy.getRank() - 1;

    // TODO: support any valid rank in (3, 4, 5)
    // The sotre async op only supports tensor with ranke <= 5.
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-dimension-size-and-format
    assert(rank > 0 && rank <= 5);
    SmallVector<unsigned> shape;
    auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for InsertSliceAsyncV2Op");
    auto llFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(llFuncOp && "LLVMFuncOp not found for InsertSliceAsyncV2Op");
    int numTMADescs = getNumTMADescs(llFuncOp);
    assert(numTMADescs > 0);
    auto sharedLayout = resultTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(sharedLayout && "unexpected layout of InsertSliceAsyncV2Op");
    auto CTAsPerCGA = sharedLayout.getCTALayout().getCTAsPerCGA();
    auto CTAOrder = sharedLayout.getCTALayout().getCTAOrder();
    auto CTASplitNum = sharedLayout.getCTALayout().getCTASplitNum();

    mlir::triton::gpu::TMAInfo tmaInfo;

    tmaInfo.tensorDataType = getCUtensorMapDataType(elemTy);
    tmaInfo.tensorRank = rank;

    assert(tmaMetadata);
    unsigned TMADescIdx = tmaMetadata->size();
    unsigned numFuncArgs = llFuncOp.getBody().front().getNumArguments();
    auto makeTensorPtr = tensorPtrMap->lookup(op.getOperation());
    auto inOrder = makeTensorPtr.getOrder();
    unsigned globalAddressArgIdx = getArgIdx(makeTensorPtr.getBase());
    tmaInfo.globalAddressArgIdx = globalAddressArgIdx;
    tmaInfo.TMADescArgIdx = numFuncArgs - numTMADescs + TMADescIdx;

    auto getDimOfOrder = [](ArrayRef<int32_t> order, int32_t i) {
      auto it = std::find(order.begin(), order.end(), i);
      assert(it != order.end());
      return std::distance(order.begin(), it);
    };

    std::vector<int32_t> globalDimsArgIdx;
    std::vector<int32_t> globalStridesArgIdx;
    // constant values are mapped to (-1 - value)
    for (int i = 0; i < rank; ++i) {
      int32_t argIdx = -1;
      auto dim = getDimOfOrder(inOrder, i);
      argIdx = getArgIdx(makeTensorPtr.getShape()[dim]);
      globalDimsArgIdx.emplace_back(argIdx);
      // handle constant stride
      argIdx = getArgIdx(makeTensorPtr.getStrides()[dim]);
      globalStridesArgIdx.emplace_back(argIdx);
    }

    tmaInfo.globalDimsArgIdx = globalDimsArgIdx;
    tmaInfo.globalStridesArgIdx = globalStridesArgIdx;

    std::vector<uint32_t> boxDims;
    auto tensorShape = makeTensorPtr.getResult()
                           .getType()
                           .cast<triton::PointerType>()
                           .getPointeeType()
                           .cast<RankedTensorType>()
                           .getShape();

    SmallVector<unsigned> numMcast(rank);
    unsigned accNumMcast = 1;
    for (unsigned i = 0; i < rank; ++i) {
      numMcast[i] = CTAsPerCGA[i] / CTASplitNum[i];
      accNumMcast *= numMcast[i];
    }
    auto shapePerCTA = getShapePerCTA(CTASplitNum, tensorShape);
    for (size_t i = 0; i < rank; ++i) {
      auto dim = getDimOfOrder(inOrder, i);
      // in case of TMA multicast, we should always slice along higher order
      // dimensions
      if (i == rank - 1) {
        assert(shapePerCTA[dim] >= accNumMcast &&
               "cases when the size of the highest order is smaller "
               "than numMcasts is not implemented");
        boxDims.emplace_back(shapePerCTA[dim] / accNumMcast);
      } else {
        boxDims.emplace_back(shapePerCTA[dim]);
      }
    }

    std::vector<uint32_t> elementStrides(rank, 1);
    tmaInfo.elementStrides = elementStrides;

    CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    if (sharedLayout.getPerPhase() == 4 && sharedLayout.getMaxPhase() == 2)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
    else if (sharedLayout.getPerPhase() == 2 && sharedLayout.getMaxPhase() == 4)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
    else if (sharedLayout.getPerPhase() == 1 && sharedLayout.getMaxPhase() == 8)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    else
      llvm::report_fatal_error(
          "Unsupported shared layout for InsertSliceAsyncV2Op");

    tmaInfo.swizzle = swizzle;
    tmaInfo.interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    tmaInfo.l2Promotion =
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    tmaInfo.oobFill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    uint32_t numBoxes = 1;
    uint32_t elemSizeOfBytes = elemTy.getIntOrFloatBitWidth() / 8;
    if (swizzle == CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
      while (elemSizeOfBytes * boxDims[0] > 128) {
        boxDims[0] = boxDims[0] / 2;
        numBoxes *= 2;
      }
    }
    tmaInfo.boxDims = boxDims;
    tmaMetadata->emplace_back(tmaInfo);

    uint32_t elemsPerBox =
        std::accumulate(boxDims.begin(), boxDims.end(), 1, std::multiplies{});

    Value clusterCTAId = getClusterCTAId(rewriter, loc);
    SmallVector<Value> multiDimClusterCTAId =
        delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

    Value llDst = adaptor.getDst();
    Value llIndex = adaptor.getIndex();
    Value src = op.getSrc();
    Value dst = op.getDst();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    auto smemObj = getSharedMemoryObjectFromStruct(loc, llDst, rewriter);

    // the offset of coord considering multicast slicing
    SmallVector<Value> mcastOffsetVals;
    // The index of slice is this CTAId is responsible for
    SmallVector<Value> multiDimSliceIdx(rank);
    for (auto i = 0; i < rank; ++i)
      multiDimSliceIdx[i] =
          udiv(multiDimClusterCTAId[i], i32_val(CTASplitNum[i]));
    Value sliceIdx =
        linearize(rewriter, loc, multiDimSliceIdx, numMcast, CTAOrder);

    Value sliceCoord;
    for (auto i = 0; i < rank; ++i) {
      if (inOrder[i] == rank - 1) {
        // TODO[goostavz]: Cases when the size of the highest order is smaller
        //                 than numMcasts is not implemented.
        sliceCoord = mul(sliceIdx, i32_val(shapePerCTA[i] / accNumMcast));
        mcastOffsetVals.emplace_back(
            mul(sliceIdx, i32_val(shapePerCTA[i] / accNumMcast)));
      } else {
        mcastOffsetVals.emplace_back(i32_val(0));
      }
    }

    uint32_t elemsPerSlice = std::accumulate(
        shapePerCTA.begin(), shapePerCTA.end(), 1, std::multiplies{});
    Value dstOffsetCommon = mul(llIndex, i32_val(elemsPerSlice));
    // [benzh] sliceCoord should be higher dimension's multiplier accumulate.
    // currently only support rank == 2.
    dstOffsetCommon =
        add(dstOffsetCommon, mul(sliceCoord, i32_val(boxDims[0])));
    auto dstPtrTy = ptr_ty(getTypeConverter()->convertType(elemTy), 3);

    Value tmaDesc =
        llFuncOp.getBody().front().getArgument(tmaInfo.TMADescArgIdx);
    // TODO: sink this logic into Triton::NVGPU dialect and support more
    // cache-policy modes
    Value l2Desc = int_val(64, 0x1000000000000000ll);

    auto ptrI8SharedTy = LLVM::LLVMPointerType::get(
        typeConverter->convertType(rewriter.getI8Type()), 3);

    SmallVector<Value> coordCommon;
    auto llCoord = getTypeConverter()->unpackLLElements(
        loc, adaptor.getSrc(), rewriter, src.getType());

    for (int i = 0; i < rank; ++i) {
      auto dim = getDimOfOrder(inOrder, i);
      Value coordDim = bitcast(llCoord[dim], i32_ty);
      if (CTASplitNum[dim] != 1) {
        // Add offset for each CTA
        //   boxDims[i] * (multiDimClusterCTAId[i] % CTASplitNum[i]);
        auto CTAOffset =
            mul(i32_val(shapePerCTA[dim]),
                urem(multiDimClusterCTAId[dim], i32_val(CTASplitNum[dim])));
        coordDim = add(coordDim, CTAOffset);
      }

      if (i == rank - 1)
        // Add offset in case of multicast slicing
        coordCommon.push_back(add(coordDim, mcastOffsetVals[dim]));
      else
        coordCommon.push_back(coordDim);
    }

    auto threadId = getThreadId(rewriter, loc);
    Value pred = icmp_eq(threadId, i32_val(0));

    auto mask = adaptor.getMask();
    if (mask) {
      // TODO(thomas): What is the right implementation for this case?
      assert(mask.getType().isInteger(1) &&
             "need to implement cases with tensor mask");
      pred = rewriter.create<arith::AndIOp>(loc, pred, mask);
    }

    Value mcastMask = getMCastMask(sharedLayout, rewriter, loc, clusterCTAId);

    for (size_t i = 0; i < numBoxes; ++i) {
      Value dstOffset =
          add(dstOffsetCommon, i32_val(i * elemsPerBox * accNumMcast));
      Value dstPtrBase = gep(dstPtrTy, smemObj.base, dstOffset);
      SmallVector<Value> coord = coordCommon;
      coord[0] = add(coordCommon[0], i32_val(i * boxDims[0]));
      rewriter.create<triton::nvgpu::TMALoadTiledOp>(
          loc, bitcast(dstPtrBase, ptrI8SharedTy), adaptor.getMbar(), tmaDesc,
          l2Desc, pred, coord, mcastMask);
    }

    rewriter.replaceOp(op, llDst);
    return success();
  }

private:
  Value getMCastMask(const SharedEncodingAttr &sharedLayout,
                     ConversionPatternRewriter &rewriter, Location loc,
                     Value clusterCTAId) const {
    auto CTAsPerCGA = sharedLayout.getCTALayout().getCTAsPerCGA();
    auto CTAOrder = sharedLayout.getCTALayout().getCTAOrder();
    auto CTASplitNum = sharedLayout.getCTALayout().getCTASplitNum();

    // Short path when no multicast is needed
    if (CTAsPerCGA == CTASplitNum)
      return nullptr;

    // Short path when bcastMask is a constant
    bool isConstMcastMask = true;
    for (unsigned s : CTASplitNum) {
      if (s > 1) {
        isConstMcastMask = false;
        break;
      }
    }
    if (isConstMcastMask) {
      unsigned numCTAs = std::accumulate(CTAsPerCGA.begin(), CTAsPerCGA.end(),
                                         1, std::multiplies{});
      return int_val(/*width*/ 16, (1u << numCTAs) - 1);
    }

    SmallVector<Value> multiDimCTAId =
        delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);
    auto rank = CTAOrder.size();
    SmallVector<SmallVector<Value>> multiDimMask(rank);
    unsigned accNumMcast = 1;
    SmallVector<unsigned> numMcast(rank);
    for (unsigned i = 0; i < rank; ++i) {
      // For the ith dimension, CTAsPerCGA[i]/CTASplitNum[i] vals is to be
      // broadcasted, which for this CTAId is:
      //     multiDimCTAId[i] % CTASplitNum[i] + (0 ..
      //     (CTAsPerCGA[i]/CTASplitNum[i] - 1)) * CTASplitNum[i]
      // TODO: will there be cases if CTAsPerCGA[i]/CTASplitNum[i] < 1?
      Value rem = urem(multiDimCTAId[i], i32_val(CTASplitNum[i]));
      numMcast[i] = CTAsPerCGA[i] / CTASplitNum[i];
      accNumMcast *= numMcast[i];
      for (unsigned j = 0; j < numMcast[i]; ++j) {
        if (j == 0) {
          multiDimMask[i].push_back(rem);
        } else {
          multiDimMask[i].push_back(add(rem, i32_val(j * CTASplitNum[i])));
        }
      }
    }

    Value bcastMask = int_val(/*width*/ 16, 0);
    Value _1_i16 = int_val(/*width*/ 16, 1);
    for (unsigned i = 0; i < accNumMcast; ++i) {
      SmallVector<unsigned> multiDimIdx =
          getMultiDimIndex<unsigned>(i, numMcast, CTAOrder);
      SmallVector<Value> multiDimMaskedCTAId(rank);
      for (unsigned dim = 0; dim < rank; ++dim) {
        multiDimMaskedCTAId[dim] = multiDimMask[dim][multiDimIdx[dim]];
      }
      Value bcastCTAId =
          linearize(rewriter, loc, multiDimMaskedCTAId, CTAsPerCGA, CTAOrder);
      // bcastMask |= 1u << bcastCTAId;
      bcastMask = or_(bcastMask, shl(_1_i16, trunc(i16_ty, bcastCTAId)));
    }

    return bcastMask;
  }

  unsigned getArgIdx(Value v) const {
    if (auto op = v.getDefiningOp<mlir::arith::ConstantOp>()) {
      return -1 -
             op.getValue().dyn_cast<IntegerAttr>().getValue().getZExtValue();
    }
    if (v.getDefiningOp() &&
        isa<mlir::UnrealizedConversionCastOp>(v.getDefiningOp())) {
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else if (v.getParentBlock()->isEntryBlock() && v.isa<BlockArgument>()) {
      // in entryblock and is BlockArgument; Because argument of func are
      // arugments of entryblock bb0 in MLIR
      return v.cast<BlockArgument>().getArgNumber();
    } else if (v.getParentBlock()->isEntryBlock() &&
               (!v.isa<BlockArgument>())) {
      // in entryblock but not BlockArgument
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else if (!v.getParentBlock()->isEntryBlock()) {
      // in non-entryblock
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else {
      llvm::report_fatal_error(
          "Operand of `MakeTensorPtrOp` is not the function's argument");
      return 0;
    }
  }

  int getNumTMADescs(LLVM::LLVMFuncOp func) const {
    if (!func->hasAttr(kAttrNumTMALoadDescsName)) {
      llvm::report_fatal_error("TritonGPU module should contain a "
                               "triton_gpu.num-tma-load attribute");
      return -1;
    }
    if (!func->hasAttr(kAttrNumTMAStoreDescsName)) {
      llvm::report_fatal_error("TritonGPU module should contain a "
                               "triton_gpu.num-tma-store attribute");
      return -1;
    }
    return func->getAttr(kAttrNumTMAStoreDescsName)
               .cast<IntegerAttr>()
               .getInt() +
           func->getAttr(kAttrNumTMALoadDescsName).cast<IntegerAttr>().getInt();
  }

  const TensorPtrMapT *tensorPtrMap;
};

void populateLoadStoreOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    mlir::triton::gpu::TMAMetadataTy *tmaMetadata,
    const TensorPtrMapT *tensorPtrMap, PatternBenefit benefit) {
  patterns.add<LoadOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<StoreOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<AtomicCASOpConversion>(typeConverter, allocation,
                                      axisInfoAnalysis, benefit);
  patterns.add<AtomicRMWOpConversion>(typeConverter, allocation,
                                      axisInfoAnalysis, benefit);
  patterns.add<AtomicLoadOpConversion>(typeConverter, allocation,
                                       axisInfoAnalysis, benefit);
  patterns.add<AtomicStoreOpConversion>(typeConverter, allocation,
                                        axisInfoAnalysis, benefit);
  patterns.add<InsertSliceOpConversion>(typeConverter, allocation,
                                        indexCacheInfo, benefit);
  patterns.add<InsertSliceAsyncOpConversion>(
      typeConverter, allocation, indexCacheInfo, axisInfoAnalysis, benefit);
  patterns.add<InsertSliceAsyncV2OpConversion>(
      typeConverter, allocation, tmaMetadata, tensorPtrMap, benefit);
  patterns.add<StoreAsyncOpConversion>(typeConverter, allocation, tmaMetadata,
                                       tensorPtrMap, benefit);
}
