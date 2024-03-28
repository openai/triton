#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
namespace mlir::triton::NVIDIA {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(int computeCapability) : computeCapability(computeCapability) {}
  bool supportMaximumMinimum() const override;
  Value ballot(ConversionPatternRewriter &rewriter, Location loc, Type type,
               Value cmp) const override;
  Value storeShared(ConversionPatternRewriter &rewriter, Location loc,
                    Value ptr, Value val, Value pred) const override;
  Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                   Type elemTy, Value pred) const override;
  Value shuffleXor(Location loc, ConversionPatternRewriter &rewriter, Value val,
                   int i) const override;
  Value shuffleUp(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  int i) const override;
  Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                   int i) const override;
  Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter, Value val,
                   Value i) const override;
  Value programId(Location loc, ConversionPatternRewriter &rewriter,
                  ModuleOp moduleOp, int axis) const override;
  bool warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce) const override;
  bool processReplicaUsingStMatrix(
      ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
      SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
      ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
      ArrayRef<unsigned> outOrd, unsigned accumNumReplicates) const override;
  void printf(Value formatStrStart, int formatStrByteCount, ValueRange args,
              ConversionPatternRewriter &rewriter) const override;

private:
  int computeCapability;
};
} // namespace mlir::triton::NVIDIA
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFONVIDIA_H
