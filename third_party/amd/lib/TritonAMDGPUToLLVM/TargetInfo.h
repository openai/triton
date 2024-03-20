#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include <string>
namespace AMD {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo(std::string arch) : arch(arch) {}
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
  void printf(Value msg, ValueRange args,
              ConversionPatternRewriter &rewriter) const override;

private:
  std::string arch;
};
} // namespace AMD
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOAMD_H
