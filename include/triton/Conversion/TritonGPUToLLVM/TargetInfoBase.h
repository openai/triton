#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H

#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
namespace mlir::triton {
class TargetInfoBase {
public:
  virtual bool supportMaximumMinimum() const = 0;
  virtual Value ballot(ConversionPatternRewriter &rewriter, Location loc,
                       Type type, Value cmp) const = 0;
  virtual Value storeShared(ConversionPatternRewriter &rewriter, Location loc,
                            Value ptr, Value val, Value pred) const = 0;
  virtual Value loadShared(ConversionPatternRewriter &rewriter, Location loc,
                           Value ptr, Type elemTy, Value pred) const = 0;
  virtual Value shuffleXor(Location loc, ConversionPatternRewriter &rewriter,
                           Value val, int i) const = 0;
  virtual Value shuffleUp(Location loc, ConversionPatternRewriter &rewriter,
                          Value val, int i) const = 0;
  virtual Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                           Value val, int i) const = 0;
  virtual Value shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                           Value val, Value i) const = 0;
  virtual Value programId(Location loc, ConversionPatternRewriter &rewriter,
                          ModuleOp moduleOp, int axis) const = 0;
  virtual bool warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                          SmallVector<Value> &acc, triton::ReduceOp op,
                          unsigned numLaneToReduce) const = 0;
  virtual bool processReplicaUsingStMatrix(
      ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
      SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
      ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
      ArrayRef<unsigned> outOrd, unsigned accumNumReplicates) const = 0;

  // Prints a message following the given format from the device.
  // |formatStrStart| is the pointer to the start of the format string global
  // variable; |args| are the arguments to fill placeholders in the format
  // string.
  virtual void printf(Value formatStrStart, int formatStrByteCount,
                      ValueRange args,
                      ConversionPatternRewriter &rewriter) const = 0;

  virtual ~TargetInfoBase() {}
};
} // namespace mlir::triton
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOBASE_H
