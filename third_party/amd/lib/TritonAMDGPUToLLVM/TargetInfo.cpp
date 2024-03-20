#include "TargetInfo.h"
#include "Utility.h"
#include "amd/include/TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
using namespace mlir;
namespace AMD {

bool TargetInfo::supportMaximumMinimum() const { return false; }
Value TargetInfo::ballot(ConversionPatternRewriter &rewriter, Location loc,
                         Type type, Value cmp) const {
  auto stringAttr = rewriter.getStringAttr("llvm.amdgcn.ballot");
  SmallVector<Value> operands = {cmp};
  Value asmResult =
      rewriter.create<LLVM::CallIntrinsicOp>(loc, type, stringAttr, operands)
          ->getResult(0);
  return asmResult;
}

Value TargetInfo::storeShared(ConversionPatternRewriter &rewriter, Location loc,
                              Value ptr, Value val, Value pred) const {
  rewriter.create<scf::IfOp>(
      loc, pred,
      [&](OpBuilder &builder, Location loc) {
        auto storeOp = builder.create<LLVM::StoreOp>(loc, val, ptr);
        builder.create<scf::YieldOp>(loc);
      },
      nullptr);
  return val;
}

Value TargetInfo::loadShared(ConversionPatternRewriter &rewriter, Location loc,
                             Value ptr, Type elemTy, Value pred) const {
  auto width = elemTy.getIntOrFloatBitWidth();
  auto loaded = rewriter.create<scf::IfOp>(
      loc, pred,
      [&](OpBuilder &builder, Location loc) {
        auto loadVal = builder.create<LLVM::LoadOp>(loc, elemTy, ptr);
        builder.create<mlir::scf::YieldOp>(loc, ValueRange({loadVal}));
      },
      [&](OpBuilder &builder, Location loc) {
        Value falseVal = builder.create<arith::ConstantOp>(
            loc, elemTy, builder.getZeroAttr(elemTy));
        builder.create<mlir::scf::YieldOp>(loc, ValueRange({falseVal}));
      });
  return loaded.getResult(0);
}

Value TargetInfo::shuffleXor(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return LLVM::AMD::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, int i) const {
  return LLVM::AMD::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return LLVM::AMD::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, Value i) const {
  return LLVM::AMD::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::programId(Location loc, ConversionPatternRewriter &rewriter,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::AMD::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce) const {
  return false;
}

bool TargetInfo::processReplicaUsingStMatrix(
    ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
    SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
    ArrayRef<unsigned> outOrd, unsigned accumNumReplicates) const {
  return false;
}

void TargetInfo::printf(Value msg, ValueRange args,
                        ConversionPatternRewriter &rewriter) const {}

} // namespace AMD
