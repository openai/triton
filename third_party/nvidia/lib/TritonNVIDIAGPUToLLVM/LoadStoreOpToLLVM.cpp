#include "TargetInfo.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getCTALayout;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

// Return the mask for the unique data accessed by given tensor type.
// Used to mask out the redundant data accessed by threads.
Value redundantDataMask(Type valueTy, ConversionPatternRewriter &rewriter,
                        Location loc, const NVIDIA::TargetInfo &targetInfo) {
  auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
  Value mask = int_val(1, 1);
  auto tid = tid_val();
  auto clusterCTAId = targetInfo.getClusterCTAId(rewriter, loc);
  if (tensorTy) {
    auto layout = tensorTy.getEncoding();
    auto shape = tensorTy.getShape();
    unsigned rank = shape.size();
    auto sizePerThread = triton::gpu::getSizePerThread(layout);
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(layout);
    auto warpsPerCTA = triton::gpu::getWarpsPerCTA(layout);
    auto order = triton::gpu::getOrder(layout);
    auto warpOrder = triton::gpu::getWarpOrder(layout);
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(layout, shape);
    Value warpSize = i32_val(32);
    Value laneId = urem(tid, warpSize);
    Value warpId = udiv(tid, warpSize);
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, warpOrder);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
    for (unsigned dim = 0; dim < rank; ++dim) {
      // if there is no data replication across threads on this dimension
      if (shape[dim] >= shapePerCTATile[dim])
        continue;
      // Otherwise, we need to mask threads that will replicate data on this
      // dimension. Calculate the thread index on this dimension for the CTA
      Value threadDim =
          add(mul(multiDimWarpId[dim], i32_val(threadsPerWarp[dim])),
              multiDimThreadId[dim]);
      mask = and_(mask, icmp_slt(mul(threadDim, i32_val(sizePerThread[dim])),
                                 i32_val(shape[dim])));
    }
    // Do not write duplicated data when multicast is enabled
    if (triton::gpu::getNumCTAs(layout) > 1) {
      auto _0 = i32_val(0);
      auto CTAsPerCGA = triton::gpu::getCTAsPerCGA(layout);
      auto CTASplitNum = triton::gpu::getCTASplitNum(layout);
      auto CTAOrder = triton::gpu::getCTAOrder(layout);

      auto multiDimClusterCTAId =
          delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

      for (unsigned dim = 0; dim < rank; ++dim) {
        // Skip when multicast is not enabled in this dimension
        if (CTAsPerCGA[dim] == CTASplitNum[dim])
          continue;
        // This wrapping rule must be consistent with emitCTAOffsetForLayout
        unsigned splitNum = std::min<unsigned>(shape[dim], CTASplitNum[dim]);
        Value repId = udiv(multiDimClusterCTAId[dim], i32_val(splitNum));
        // Consider the example where CTAsPerCGA = [4] and CTASplitNum = [2]:
        //     CTA0 and CTA2 holds data of block0,
        //     CTA1 and CTA3 holds data of block1.
        // Only CTA0 and CTA1 are expected to write while CTA2 and CTA3 should
        // be masked. We add the following mask:
        //     multiDimClusterCTAId[dim] / splitNum == 0
        // Actually in all existing cases of multicast, splitNum is always 1.
        // The mask is equivalent to:
        //     multiDimClusterCTAId[dim] == 0
        mask = and_(mask, icmp_eq(repId, _0));
      }
    }
  } else {
    // If the tensor is not ranked, then it is a scalar and only thread 0 of
    // CTA0 can write
    mask = and_(mask, icmp_eq(clusterCTAId, i32_val(0)));
    mask = and_(mask, icmp_eq(tid, i32_val(0)));
  }
  return mask;
}

Value createCachePolicy(EvictionPolicy opEvict,
                        ConversionPatternRewriter &rewriter, Location loc) {
  // Emit createpolicy.fractional.L2::policy.b64 xx 1.0
  PTXBuilder ptxBuilder;
  const bool hasL2EvictPolicy =
      opEvict == triton::EvictionPolicy::EVICT_FIRST ||
      opEvict == triton::EvictionPolicy::EVICT_LAST;
  Value policyRet;
  if (hasL2EvictPolicy) {
    auto &policy =
        ptxBuilder.create<>("createpolicy.fractional")
            ->o("L2::evict_first",
                opEvict == triton::EvictionPolicy::EVICT_FIRST)
            .o("L2::evict_last", opEvict == triton::EvictionPolicy::EVICT_LAST)
            .b(64);
    const std::string writeConstraint = "=l";
    // prepare asm operands
    auto *dstOpr = ptxBuilder.newOperand(writeConstraint,
                                         /*init=*/true);
    std::string fractionStr = "1.0";
    auto *fractionOpr = ptxBuilder.newConstantOperand(fractionStr);
    policy(dstOpr, fractionOpr);
    Type policyRetTy = rewriter.getI64Type();
    policyRet = ptxBuilder.launch(rewriter, loc, policyRetTy);
  }
  return policyRet;
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const NVIDIA::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    LDBG("getVectorSize contiguity = " << contiguity << " pointeeBitWidth = "
                                       << pointeeBitWidth);
    // The maximum vector size is 128 bits on NVIDIA GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const NVIDIA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const NVIDIA::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto typeConverter = getTypeConverter();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();
    LDBG("Lower LoadOp for " << ptr);

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));
    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    if (llMask) {
      LLVM_DEBUG(DBGS() << "vec = " << vec
                        << " mask_alignment = " << getMaskAlignment(mask));
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      LLVM_DEBUG(llvm::dbgs() << " vec = " << vec << '\n');
    }

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    LDBG("LoadOp numElems = " << numElems << " vec = " << vec
                              << " valueElemNBits = " << valueElemNBits << " "
                              << op.getType());
    const bool hasL2EvictPolicy =
        op.getEvict() == triton::EvictionPolicy::EVICT_FIRST ||
        op.getEvict() == triton::EvictionPolicy::EVICT_LAST;
    // Create cache policy here. Consider moving it to function scope.
    Value policyRet = createCachePolicy(op.getEvict(), rewriter, loc);
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
                     .o("L2::cache_hint", hasL2EvictPolicy)
                     .v(nWords)
                     .b(width);

      PTXBuilder::Operand *evictOpr{};

      if (hasL2EvictPolicy)
        evictOpr = ptxBuilder.newOperand(policyRet, "l");

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
                rewriter, loc, typeConverter->getIndexType(), s);
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
        if (isa<LLVM::LLVMStructType>(retTy)) {
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
            rewriter, loc, typeConverter->getIndexType(), ii % tmp);
        Value loaded = extract_element(valueElemTy, rets[ii / tmp], vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const NVIDIA::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

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

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    SmallVector<Value> maskElems;
    if (llMask) {
      Value mask = op.getMask();
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    Value mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    const int numVecs = elemsPerThread / vec;
    const bool hasL2EvictPolicy =
        op.getEvict() == triton::EvictionPolicy::EVICT_FIRST ||
        op.getEvict() == triton::EvictionPolicy::EVICT_LAST;
    // Create cache policy here. Consider moving it to function scope.
    Value policyRet = createCachePolicy(op.getEvict(), rewriter, loc);
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      // TODO: optimization when ptr is AddPtr with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

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
              .o("L2::cache_hint", hasL2EvictPolicy)
              .v(nWords)
              .b(width);

      PTXBuilder::Operand *evictOpr{};
      if (hasL2EvictPolicy)
        evictOpr = ptxBuilder.newOperand(policyRet, "l");
      if (!evictOpr)
        ptxStoreInstr(asmAddr, asmArgList).predicate(maskVal, "b");
      else
        ptxStoreInstr(asmAddr, asmArgList, evictOpr).predicate(maskVal, "b");

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

void createBarrier(ConversionPatternRewriter &rewriter, Location loc,
                   int numCTAs) {
  if (numCTAs == 1) {
    barrier();
  } else {
    rewriter.create<triton::nvidia_gpu::ClusterArriveOp>(loc, false);
    rewriter.create<triton::nvidia_gpu::ClusterWaitOp>(loc);
  }
}

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const NVIDIA::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

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

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(op.getPtr());
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(op.getVal().getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
    }

    Value mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);
    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);

    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value casVal = undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        casVal = insert_element(vecTy, casVal, valElements[i + ii], iiVal);
      }

      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      casVal = valElements[i];
      PTXBuilder ptxBuilderAtomicCAS;
      std::string tyId = valueElemNBits * vec == 64
                             ? "l"
                             : (valueElemNBits * vec == 32 ? "r" : "h");
      auto *dstOpr = ptxBuilderAtomicCAS.newOperand("=" + tyId, /*init=*/true);
      auto *ptrOpr = ptxBuilderAtomicCAS.newAddrOperand(casPtr, "l");
      auto *cmpOpr = ptxBuilderAtomicCAS.newOperand(casCmp, tyId);
      auto *valOpr = ptxBuilderAtomicCAS.newOperand(casVal, tyId);
      auto &atom = *ptxBuilderAtomicCAS.create<PTXInstr>("atom");
      auto sTy = "b" + std::to_string(valueElemNBits);
      std::string semStr;
      llvm::raw_string_ostream os(semStr);
      os << op.getSem();
      auto scope = stringifyMemSyncScope(op.getScope()).str();
      atom.global().o(semStr).o(scope).o("cas").o(sTy);
      atom(dstOpr, ptrOpr, cmpOpr, valOpr).predicate(mask);

      if (tensorTy) {
        auto retType = vec == 1 ? valueElemTy : vecTy;
        auto ret = ptxBuilderAtomicCAS.launch(rewriter, loc, retType);
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
        }
      } else {
        auto old = ptxBuilderAtomicCAS.launch(rewriter, loc, valueElemTy);
        createBarrier(rewriter, loc, numCTAs);
        Value atomPtr =
            LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
        atomPtr = bitcast(atomPtr, ptr_ty(ctx, 3));
        // Only threads with mask = True store the result
        PTXBuilder ptxBuilderStore;
        auto *dstOprStore = ptxBuilderStore.newAddrOperand(atomPtr, "r");
        auto *valOprStore = ptxBuilderStore.newOperand(old, "r");
        auto &st = *ptxBuilderStore.create<PTXInstr>("st");
        st.shared().o(sTy);
        st(dstOprStore, valOprStore).predicate(mask);
        auto ASMReturnTy = void_ty(ctx);
        ptxBuilderStore.launch(rewriter, loc, ASMReturnTy);
        createBarrier(rewriter, loc, numCTAs);
        Value ret = load(valueElemTy, atomPtr);
        createBarrier(rewriter, loc, numCTAs);
        rewriter.replaceOp(op, {ret});
      }
    }

    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const NVIDIA::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

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

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
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
      auto valTy = cast<RankedTensorType>(val.getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
      // mask
      numElems = tensorTy.getNumElements();
    }
    Value mask = redundantDataMask(valueTy, rewriter, loc, targetInfo);

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

      auto scope = stringifyMemSyncScope(op.getScope()).str();
      auto &atom = ptxBuilderAtomicRMW.create<>("atom")->global().o(scope);
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
        Value atomPtr =
            LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
        atomPtr = bitcast(atomPtr, ptr_ty(ctx, 3));
        // Only threads with rmwMask = True store the result
        PTXBuilder ptxBuilderStore;
        auto &storeShared =
            ptxBuilderStore.create<>("st")->shared().o("b" + sBits);
        auto *ptrOpr = ptxBuilderStore.newAddrOperand(atomPtr, "r");
        auto *valOpr = ptxBuilderStore.newOperand(old, tyId);
        storeShared(ptrOpr, valOpr).predicate(rmwMask);
        ptxBuilderStore.launch(rewriter, loc, void_ty(ctx));
        createBarrier(rewriter, loc, numCTAs);
        Value ret = load(valueElemTy, atomPtr);
        createBarrier(rewriter, loc, numCTAs);
        rewriter.replaceOp(op, {ret});
      }
    }
    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const NVIDIA::TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value res = op.getResult();
    Value mask = op.getMask();
    Value other = op.getOther();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto srcLayout = srcTy.getEncoding();
    assert((isa<BlockedEncodingAttr, SliceEncodingAttr>(srcLayout) &&
            "Unexpected srcLayout in AsyncCopyGlobalToLocalOpConversion"));
    auto resSharedLayout = cast<SharedEncodingAttr>(dstTy.getEncoding());
    auto srcShape = srcTy.getShape();
    assert((srcShape.size() <= 3) &&
           "insert_slice_async: Unexpected rank of %src");

    Value llDst = adaptor.getResult();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // %src
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);

    // %dst
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(srcElems.size() == maskElems.size());
    }

    // %other
    SmallVector<Value> otherElems;
    if (llOther) {
      // FIXME(Keren): assume other is 0 for now.
      //
      // It's not necessary for now because the pipeline pass will skip
      // generating insert_slice_async if the load op has any "other" tensor.
      otherElems = unpackLLElements(loc, llOther, rewriter);
      assert(srcElems.size() == otherElems.size());
    }

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned maxVec = getContiguity(op.getSrc());
    if (mask) {
      maxVec = std::min(maxVec, getMaskAlignment(mask));
    }

    // Addresses to store into, one per `vecTy`.
    VectorType vecTy;
    SmallVector<Value> shmemAddrs;
    bool ok = emitTransferBetweenRegistersAndShared(
        srcTy, dstTy, resElemTy, maxVec, smemObj.base, smemObj.strides, loc,
        rewriter, targetInfo, [&](VectorType vecTy_, Value shmemAddr) {
          vecTy = vecTy_;
          shmemAddrs.push_back(shmemAddr);
        });
    assert(ok);

    int vecBytes = vecTy.getNumElements() * vecTy.getElementTypeBitWidth() / 8;
    assert(llvm::isPowerOf2_32(vecBytes));
    if (vecBytes < 4) {
      return emitError(loc, "cp.async does not support transfers smaller than "
                            "4 bytes; calculated this as ")
             << vecBytes << " bytes";
    }

    for (int i = 0; i < shmemAddrs.size(); i++) {
      // It's possible that vecTy is larger than 128 bits, in which case we have
      // to use multiple cp.async instructions.
      int wordBytes = std::min(vecBytes, 16);
      int wordElems = wordBytes * 8 / vecTy.getElementTypeBitWidth();
      int numWordsInVec = std::max(1, vecBytes / wordBytes);
      for (int j = 0; j < numWordsInVec; j++) {
        int elemIdx = i * vecTy.getNumElements() + j * wordElems;

        // Tune CG and CA.
        CacheModifier srcCacheModifier =
            wordBytes == 16 ? CacheModifier::CG : CacheModifier::CA;
        assert(wordBytes == 16 || wordBytes == 8 || wordBytes == 4);

        PTXBuilder ptxBuilder;
        auto &copyAsyncOp =
            *ptxBuilder.create<PTXCpAsyncLoadInstr>(srcCacheModifier);
        auto *dstOperand = ptxBuilder.newAddrOperand(shmemAddrs[i], "r",
                                                     /*offset=*/j * wordBytes);
        auto *srcOperand = ptxBuilder.newAddrOperand(srcElems[elemIdx], "l");
        auto *copySize = ptxBuilder.newConstantOperand(wordBytes);
        auto *srcSize = copySize;
        if (op.getMask()) {
          // We don't use predicate in this case, setting src-size to 0
          // if there's any mask. cp.async will automatically fill the
          // remaining slots with 0 if cp-size > src-size.
          // XXX(Keren): Always assume other = 0 for now.
          auto selectOp =
              select(maskElems[elemIdx], i32_val(wordBytes), i32_val(0));
          srcSize = ptxBuilder.newOperand(selectOp, "r");
        }

        // When 'other != 0' is supported, we will need to fold the op.getMask()
        // and redundantDataMask() into the same predicate, the way it is done
        // for LoadOp.
        Value maskVal = redundantDataMask(srcTy, rewriter, loc, targetInfo);

        // TODO: Masking does not work for CTA multicast with cp.async. This is
        // a quick and dirty workaround to avoid the issue.
        bool skipMaskForMultiCTA = triton::gpu::getNumCTAs(srcLayout) > 1;
        if (!skipMaskForMultiCTA) {
          copyAsyncOp(dstOperand, srcOperand, copySize, srcSize)
              .predicate(maskVal);
        } else {
          copyAsyncOp(dstOperand, srcOperand, copySize, srcSize);
        }
        ptxBuilder.launch(rewriter, loc, void_ty(getContext()));
      }
    }

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct AsyncTMACopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op.getCache() == triton::CacheModifier::NONE &&
           "cache modifiers not supported yet.");
    assert(op.getEvict() == triton::EvictionPolicy::NORMAL &&
           "eviction policy not supported yet.");
    auto loc = op.getLoc();
    Type llvmElemTy =
        typeConverter->convertType(op.getResult().getType().getElementType());
    auto barrierMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getBarrier(),
        typeConverter->convertType(op.getBarrier().getType().getElementType()),
        rewriter);
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), llvmElemTy, rewriter);
    auto voidTy = void_ty(op->getContext());
    auto id = getThreadId(rewriter, loc);

    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value warpID = udiv(id, i32_val(warpSize));
    warpID = LLVM::NVIDIA::shuffleIdx(loc, rewriter, warpID, 0);
    Value pred = adaptor.getPred();
    // Select just one thread for the TMA copy. This also helps the compiler to
    // figure out that the op is uniform.
    pred = and_(pred, LLVM::NVIDIA::createElectPredicate(loc, rewriter));

    int elementSizeInBytes =
        op.getResult().getType().getElementType().getIntOrFloatBitWidth() / 8;
    int totalNumElements = product(op.getResult().getType().getShape());
    int64_t size = totalNumElements * elementSizeInBytes;

    int innerBlockSize = op.getResult().getType().getShape().back();
    int contigDimSizeInByte = innerBlockSize * elementSizeInBytes;
    int numCopies = 1;
    int rank = op.getCoord().size();
    if (rank > 1)
      numCopies = ceil<int>(contigDimSizeInByte, 128);

    // The bounding box inner dimension must be less than or equal to the
    // swizzle size.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    // We clamp the block size and the codegen will emit multiple copy
    // operations.
    for (int copyIdx = 0; copyIdx < numCopies; copyIdx += numWarps) {
      int numWarpsToCopy = std::min(numCopies - copyIdx, numWarps);
      if (numWarpsToCopy == 1)
        warpID = i32_val(0);
      Value boxPred =
          and_(pred, icmp_ult(id, i32_val(numWarpsToCopy * warpSize)));
      ::mlir::triton::PTXBuilder ptxBuilderTMA;
      Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
      Value copyIdxVal = add(warpID, i32_val(copyIdx));
      Value shMemOffset =
          mul(copyIdxVal, i32_val(totalNumElements / numCopies));
      Value shMemPtr =
          gep(elemPtrTy, llvmElemTy, dstMemObj.getBase(), shMemOffset);
      SmallVector<PTXBuilder::Operand *> operands = {
          ptxBuilderTMA.newOperand(boxPred, "b"),
          ptxBuilderTMA.newOperand(shMemPtr, "r"),
          ptxBuilderTMA.newOperand(adaptor.getDescPtr(), "l")};
      std::string tmaInst =
          "@$0 cp.async.bulk.tensor." + std::to_string(rank) +
          "d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {";
      int operandIdx = 3;
      for (int i = 0; i < rank; i++) {
        Value coord = adaptor.getCoord()[rank - i - 1];
        if (i == 0) {
          Value offset = mul(copyIdxVal, i32_val(128 / elementSizeInBytes));
          coord = add(coord, offset);
        }
        operands.push_back(ptxBuilderTMA.newOperand(coord, "r"));
        tmaInst += "$" + std::to_string(operandIdx++);
        if (i != rank - 1)
          tmaInst += ", ";
      }
      operands.push_back(
          ptxBuilderTMA.newOperand(barrierMemObj.getBase(), "r"));
      tmaInst += "}], [$" + std::to_string(operandIdx++) + "];";
      auto &tma = *ptxBuilderTMA.create<>(tmaInst);
      tma(operands, /*onlyAttachMLIRArgs=*/true);
      ptxBuilderTMA.launch(rewriter, loc, voidTy);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncTMACopyLocalToGlobalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type llvmElemTy =
        typeConverter->convertType(op.getSrc().getType().getElementType());
    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), llvmElemTy, rewriter);
    auto voidTy = void_ty(op->getContext());
    auto id = getThreadId(rewriter, loc);
    // Select just one thread for the TMA copy. This also helps the compiler to
    // figure out that the op is uniform.
    Value pred = LLVM::NVIDIA::createElectPredicate(loc, rewriter);
    int elementSizeInBytes =
        op.getSrc().getType().getElementType().getIntOrFloatBitWidth() / 8;
    int totalNumElements = product(op.getSrc().getType().getShape());
    int64_t size = totalNumElements * elementSizeInBytes;

    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value warpID = udiv(id, i32_val(warpSize));
    warpID = LLVM::NVIDIA::shuffleIdx(loc, rewriter, warpID, 0);
    int innerBlockSize = op.getSrc().getType().getShape().back();
    int contigDimSizeInByte = innerBlockSize * elementSizeInBytes;
    int numCopies = 1;
    int rank = op.getCoord().size();
    if (rank > 1)
      numCopies = ceil<int>(contigDimSizeInByte, 128);

    // The bounding box inner dimension must be less than or equal to the
    // swizzle size.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    // We clamp the block size and the codegen will emit multiple copy
    // operations.
    for (int copyIdx = 0; copyIdx < numCopies; copyIdx += numWarps) {
      int numWarpsToCopy = std::min(numCopies - copyIdx, numWarps);
      if (numWarpsToCopy == 1)
        warpID = i32_val(0);
      Value boxPred =
          and_(pred, icmp_ult(id, i32_val(numWarpsToCopy * warpSize)));
      ::mlir::triton::PTXBuilder ptxBuilderTMA;
      Type elemPtrTy = ptr_ty(rewriter.getContext(), 3);
      Value copyIdxVal = add(warpID, i32_val(copyIdx));
      Value shMemOffset =
          mul(copyIdxVal, i32_val(totalNumElements / numCopies));
      Value shMemPtr =
          gep(elemPtrTy, llvmElemTy, dstMemObj.getBase(), shMemOffset);
      SmallVector<PTXBuilder::Operand *> operands = {
          ptxBuilderTMA.newOperand(boxPred, "b"),
          ptxBuilderTMA.newOperand(adaptor.getDescPtr(), "l")};
      std::string tmaInst = "@$0 cp.async.bulk.tensor." + std::to_string(rank) +
                            "d.global.shared::cta.bulk_group [$1, {";
      int operandIdx = 2;
      for (int i = 0; i < rank; i++) {
        Value coord = adaptor.getCoord()[rank - i - 1];
        if (i == 0) {
          Value offset = mul(copyIdxVal, i32_val(128 / elementSizeInBytes));
          coord = add(coord, offset);
        }
        operands.push_back(ptxBuilderTMA.newOperand(coord, "r"));
        tmaInst += "$" + std::to_string(operandIdx++);
        if (i != rank - 1)
          tmaInst += ", ";
      }
      operands.push_back(ptxBuilderTMA.newOperand(shMemPtr, "r"));
      tmaInst += "}], [$" + std::to_string(operandIdx++) + "];";
      auto &tma = *ptxBuilderTMA.create<>(tmaInst);
      tma(operands, /*onlyAttachMLIRArgs=*/true);
      ptxBuilderTMA.launch(rewriter, loc, voidTy);
    }

    // TODO: Separate the syncronizations operations into separate TTGIR ops to
    // be able to schedule them at the high level.
    const std::string ptx = "cp.async.bulk.commit_group";
    PTXBuilder ptxBuilderSync;
    ptxBuilderSync.create<>(ptx)->operator()();
    ptxBuilderSync.launch(rewriter, op.getLoc(), void_ty(op.getContext()));

    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    auto &asyncWaitOp = *ptxBuilder.create<>("cp.async.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    PTXBuilder ptxBuilder;
    ptxBuilder.create<>("cp.async.commit_group")->operator()();
    ptxBuilder.launch(rewriter, op.getLoc(), void_ty(op.getContext()));

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct TMAStoreWaitConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMAStoreWait> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMAStoreWait op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    auto &asyncWaitOp = *ptxBuilder.create<>("cp.async.bulk.wait_group.read");
    auto num = op.getPendings();
    asyncWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<AsyncCopyGlobalToLocalOpConversion, AtomicCASOpConversion,
               AtomicRMWOpConversion, LoadOpConversion, StoreOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit);
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, benefit);
  patterns.add<AsyncTMACopyGlobalToLocalOpConversion,
               AsyncTMACopyLocalToGlobalOpConversion, TMAStoreWaitConversion>(
      typeConverter, benefit);
}
