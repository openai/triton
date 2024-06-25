#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace mlir::triton::gpu {
namespace {

using ::mlir::isLayoutMmaV1;
using ::mlir::LLVM::getMultiDimOffset;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;

struct LocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    // TODO: do we need to check if src is shared ?
    if (isa<SharedEncodingAttr>(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerSharedToDistributed(op, adaptor, getTypeConverter(),
                                      rewriter);
    }
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<BlockedEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      return lowerSharedToDotOpFMA(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  LogicalResult
  lowerSharedToDotOpFMA(LocalLoadOp op, LocalLoadOpAdaptor adaptor,
                        const LLVMTypeConverter *typeConverter,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType dstTy = op.getType();
    Attribute dstLayout = dstTy.getEncoding();
    auto dotLayout = cast<DotOperandEncodingAttr>(dstLayout);
    auto blockedLayout = cast<BlockedEncodingAttr>(
        cast<DotOperandEncodingAttr>(dstLayout).getParent());
    auto thread = getThreadId(rewriter, loc);
    Value res = SharedToDotOperandFMA::convertLayout(
        dotLayout.getOpIdx(), op.getSrc(), adaptor.getSrc(), blockedLayout,
        thread, loc, getTypeConverter(), rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }
  LogicalResult
  lowerSharedToDistributed(LocalLoadOp op, LocalLoadOpAdaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() <= 2 &&
           "Unexpected rank of ConvertLayout(shared->blocked)");
    auto srcSharedLayout = cast<SharedEncodingAttr>(srcTy.getEncoding());
    auto dstLayout = dstTy.getEncoding();
    auto inOrd = getOrder(srcSharedLayout);

    auto smemObj = getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(srcTy.getElementType()), rewriter);
    auto elemLlvmTy = typeConverter->convertType(dstTy.getElementType());

    SmallVector<Value> outVals = loadSharedToDistributed(
        dstTy, srcTy, elemLlvmTy, smemObj, loc, rewriter, targetInfo);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
public:
  ConvertLayoutOpConversion(LLVMTypeConverter &typeConverter,
                            const TargetInfoBase &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isSupported(srcLayout, dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    return failure();
  }

private:
  bool isSupported(Attribute srcLayout, Attribute dstLayout) const {
    return isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout) &&
           !isLayoutMmaV1(srcLayout) && !isLayoutMmaV1(dstLayout);
  }
  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> origRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const {
    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto rank = type.getRank();
    auto sizePerThread = getSizePerThread(layout);
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<unsigned> numCTATiles(rank);
    auto shapePerCTATile = getShapePerCTATile(layout);
    auto shapePerCTA = getShapePerCTA(layout, type.getShape());
    auto order = getOrder(layout);
    for (unsigned d = 0; d < rank; ++d) {
      numCTATiles[d] = ceil<unsigned>(shapePerCTA[d], shapePerCTATile[d]);
    }
    auto elemTy = type.getElementType();
    bool isInt1 = elemTy.isInteger(1);
    bool isPtr = isa<triton::PointerType>(elemTy);
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isInt1)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);

    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
      auto multiDimCTAInRepId =
          getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
      SmallVector<unsigned> multiDimCTAId(rank);
      for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
        auto d = it.index();
        multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
      }

      auto linearCTAId =
          getLinearIndex<unsigned>(multiDimCTAId, numCTATiles, order);
      // TODO: This is actually redundant index calculation, we should
      //       consider of caching the index calculation result in case
      //       of performance issue observed.
      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, targetInfo, elemId, type,
                              multiDimCTAInRepId, shapePerCTATile);
        SmallVector<Value> multiDimOffsetWrapped = getWrappedMultiDimOffset(
            rewriter, loc, multiDimOffset, origRepShape, shapePerCTATile,
            shapePerCTA);
        Value offset = linearize(rewriter, loc, multiDimOffsetWrapped,
                                 paddedRepShape, outOrd);
        auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
        Value ptr = gep(elemPtrTy, llvmElemTy, smemBase, offset);
        auto vecTy = vec_ty(llvmElemTy, vec);
        ptr = bitcast(ptr, ptr_ty(rewriter.getContext(), 3));
        if (stNotRd) {
          Value valVec = undef(vecTy);
          for (unsigned v = 0; v < vec; ++v) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread + v];
            if (isInt1)
              currVal = zext(llvmElemTy, currVal);
            else if (isPtr)
              currVal = ptrtoint(llvmElemTy, currVal);
            valVec = insert_element(vecTy, valVec, currVal, i32_val(v));
          }
          store(valVec, ptr);
        } else {
          Value valVec = load(vecTy, ptr);
          for (unsigned v = 0; v < vec; ++v) {
            Value currVal = extract_element(llvmElemTy, valVec, i32_val(v));
            if (isInt1)
              currVal = icmp_ne(currVal,
                                rewriter.create<LLVM::ConstantOp>(
                                    loc, i8_ty, rewriter.getI8IntegerAttr(0)));
            else if (isPtr)
              currVal = inttoptr(llvmElemTyOrig, currVal);
            vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
          }
        }
      }
    }
  }
  // blocked/mma -> blocked/mma.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(ConvertLayoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (product(srcTy.getShape()) == 1) {
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      SmallVector<Value> outVals(getTotalElemsPerThread(dstTy), inVals[0]);
      Value result =
          packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
      return success();
    }

    if (isMmaToMmaShortcut(srcTy, dstTy)) {
      auto srcMmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(srcLayout);
      auto dstMmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(dstLayout);
      assert(srcMmaLayout.getVersionMajor() == 3 &&
             dstMmaLayout.getVersionMajor() == 3 &&
             "mma shortcut conversion only supported on hopoper");
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      auto dstStructType =
          dyn_cast<LLVM::LLVMStructType>(typeConverter->convertType(dstTy));
      auto outValsSize = dstStructType.getBody().size();
      SmallVector<Value> outVals;
      outVals.reserve(outValsSize);
      auto inInstrShape =
          cast<NvidiaMmaEncodingAttr>(srcLayout).getInstrShape();
      auto outInstrShape =
          cast<NvidiaMmaEncodingAttr>(dstLayout).getInstrShape();

      if (inInstrShape[1] >= outInstrShape[1]) {
        // Reduce the N-dimension of input values to fit in the output shape.
        unsigned reduceBy = inInstrShape[1] / outInstrShape[1];
        unsigned inN = inInstrShape[1] / 4;
        unsigned inM = inVals.size() / inN;
        unsigned outN = outInstrShape[1] / 4;
        inM = inM / 2; // One warp owns two rows
        inN = inN * 2;
        outN = outN * 2;
        for (unsigned i = 0; i < inM; ++i) {
          for (unsigned j = 0; j < outN; j += 4) {
            for (unsigned k = 0; k < 4; ++k)
              outVals.push_back(inVals[i * inN + j + k]);
          }
        }
        assert(outVals.size() == outValsSize);
        Value result =
            packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
        rewriter.replaceOp(op, result);
        return success();
      } else {
        llvm_unreachable("unsupported mma shortcut conversion");
        return failure();
      }
    }

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    auto srcShapePerCTATile = getShapePerCTATile(srcLayout, srcTy.getShape());
    auto dstShapePerCTATile = getShapePerCTATile(dstLayout, shape);
    auto shapePerCTA = getShapePerCTA(srcLayout, shape);

    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA =
          std::min<unsigned>(shapePerCTA[d], srcShapePerCTATile[d]);
      unsigned outPerCTA =
          std::min<unsigned>(shapePerCTA[d], dstShapePerCTATile[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      numReplicates[d] = ceil<unsigned>(shapePerCTA[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], outPerCTA);
    }

    // Potentially we need to store for multiple CTAs in this replication
    auto accumNumReplicates = product<unsigned>(numReplicates);
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    unsigned inVec = 0;
    unsigned outVec = 0;
    auto origRepShape = getRepShapeForCvtLayout(op);
    auto paddedRepShape = getScratchConfigForCvtLayout(op, inVec, outVec);

    unsigned outElems = getTotalElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0) {
        barrier();
      }
      auto successful = targetInfo.processReplicaUsingStMatrix(
          rewriter, loc, smemBase, vals, srcTy,
          getTypeConverter()->convertType(srcTy.getElementType()),
          paddedRepShape, origRepShape, outOrd, accumNumReplicates);
      if (!successful) {
        processReplica(loc, rewriter, /*stNotRd*/ true, srcTy, inNumCTAsEachRep,
                       multiDimRepId, inVec, paddedRepShape, origRepShape,
                       outOrd, vals, smemBase);
      }
      barrier();
      processReplica(loc, rewriter, /*stNotRd*/ false, dstTy, outNumCTAsEachRep,
                     multiDimRepId, outVec, paddedRepShape, origRepShape,
                     outOrd, outVals, smemBase);
    }

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct ConvertLayoutOpUsingLinearLayoutsConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  // Set benefit to 2 so that this pattern applies before other convert-layout
  // conversions.  TODO(jlebar): Eventually we want this to be the only pattern.
  explicit ConvertLayoutOpUsingLinearLayoutsConversion(
      LLVMTypeConverter &typeConverter, PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    const auto &shape = op.getType().getShape();
    std::optional<LinearLayout> srcLayout =
        gpu::toLinearLayout(shape, op.getSrc().getType().getEncoding());
    std::optional<LinearLayout> dstLayout =
        gpu::toLinearLayout(shape, op.getType().getEncoding());
    if (!srcLayout.has_value() || !dstLayout.has_value()) {
      return failure();
    }

    // There are four cases to handle.
    //
    //  1. Transfer between values in the same thread, in which case we simply
    //     reorder the elements of adaptor.getSrc().
    //  2. Transfer between values in the same warp, in which case we try to
    //     move values using warp shuffles, though if the pattern is complicated
    //     enough we may fall back to using shared memory (case 3).
    //  3. Transfer between values in the same CTA, in which case we move values
    //     through shared memory.
    //  4. Transfer between values in different CTAs, in which case we move
    //     values through distributed shared memory.
    //
    // We can tell which case we're in by examining `conversion`.  If e.g. the
    // block -> block mapping is {1, 2, 4, ...} then there's no movement between
    // data in different CTAs and we know we're not in case 4.
    LinearLayout conversion = srcLayout->invertAndCompose(*dstLayout);

    int numLanes = conversion.getInDimSize(str_attr("lane"));
    int numWarps = conversion.getInDimSize(str_attr("warp"));
    int numBlocks = conversion.getInDimSize(str_attr("block"));

    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");

    // TODO(jlebar): These checks are overly-restrictive.  For example, we can
    // transfer by shuffling registers (case 1) if and only if all of the bases
    // for `register` have 0s for lane, warp, and block.  But the check below is
    // stronger than this, checking also that the choice of lane/warp/block does
    // not affect the permutation of registers.  If we allow different
    // lane/warp/blocks to have different permutations, we can generalize this.
    if (std::optional<LinearLayout> c = conversion.divideRight(
            LinearLayout::identity1D(numLanes, kLane, kLane) *
            LinearLayout::identity1D(numWarps, kWarp, kWarp) *
            LinearLayout::identity1D(numBlocks, kBlock, kBlock));
        c.has_value()) {
      return transferWithinThread(*c, op, adaptor, rewriter);
    }

    if (std::optional<LinearLayout> c = conversion.divideRight(
            LinearLayout::identity1D(numWarps, kWarp, kWarp) *
            LinearLayout::identity1D(numBlocks, kBlock, kBlock));
        c.has_value()) {
      return transferWithinLane(*c, op, adaptor, rewriter);
    }

    if (std::optional<LinearLayout> c = conversion.divideRight(
            LinearLayout::identity1D(numBlocks, kBlock, kBlock));
        c.has_value()) {
      return transferWithinBlock(*c, op, adaptor, rewriter);
    }

    return transferWithinBlockGroup(conversion, op, adaptor, rewriter);
  }

  LogicalResult
  transferWithinThread(const LinearLayout &conversion, ConvertLayoutOp op,
                       OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");

    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));
    assert(ArrayRef(to_vector(conversion.getInDimNames())) ==
           ArrayRef{kRegister});
    assert(ArrayRef(to_vector(conversion.getOutDimNames())) ==
           ArrayRef{kRegister});

    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> outVals(conversion.getOutDimSize(kRegister));
    for (int i = 0; i < conversion.getInDimSize(kRegister); i++) {
      auto dstIdx = conversion.apply({{kRegister, i}});
      outVals[dstIdx.begin()->second] = inVals[i];
    }
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult transferWithinLane(const LinearLayout &conversion,
                                   ConvertLayoutOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    // TODO(jlebar): Implement me.
    return failure();
  }

  LogicalResult transferWithinBlock(const LinearLayout &conversion,
                                    ConvertLayoutOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    // TODO(jlebar): Implement me.
    return failure();
  }

  LogicalResult
  transferWithinBlockGroup(const LinearLayout &conversion, ConvertLayoutOp op,
                           OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    // TODO(jlebar): Implement me.
    return failure();
  }
};

} // namespace
} // namespace mlir::triton::gpu

void mlir::triton::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // We prefer using the linear layout conversion, so it gets a higher benefit.
  // Eventually the LL conversion will subsume all of the others and be the only
  // one left.
  patterns.add<gpu::ConvertLayoutOpUsingLinearLayoutsConversion>(
      typeConverter, benefit.getBenefit() + 1);
  patterns.add<gpu::ConvertLayoutOpConversion>(typeConverter, targetInfo,
                                               benefit);
  patterns.add<gpu::LocalLoadOpConversion>(typeConverter, targetInfo, benefit);
}
