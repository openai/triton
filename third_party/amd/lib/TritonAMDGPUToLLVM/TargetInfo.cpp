#include "TargetInfo.h"
#include "Utility.h"
#include "amd/include/TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
using namespace mlir;

namespace {
template <typename T>
static LLVM::LLVMFuncOp getOrInsertFunction(T &moduleOp, const Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            StringRef name,
                                            LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                            LLVM::Linkage::External);
  }
  return ret;
}

// Extend all values to 64-bit per printf call requirements.
Value printfPromoteValue(ConversionPatternRewriter &rewriter, Value value) {
  auto *context = rewriter.getContext();
  auto loc = UnknownLoc::get(context);
  auto type = value.getType();

  if (auto floatType = dyn_cast<FloatType>(type)) {
    Value newValue = value;
    if (!floatType.isF64())
      newValue = fpext(f64_ty, newValue);
    return bitcast(newValue, i64_ty);
  }

  llvm::errs() << "current value: " << value << "\n";
  assert(type.isIntOrIndex());
  if (type.getIntOrFloatBitWidth() < 64) {
    if (type.isUnsignedInteger()) {
      return zext(ui64_ty, value);
    } else {
      return sext(i64_ty, value);
    }
  }

  return value;
}
} // namespace
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

void TargetInfo::printf(Value formatStrStart, int formatStrByteCount,
                        ValueRange args,
                        ConversionPatternRewriter &rewriter) const {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto *ctx = rewriter.getContext();
  mlir::Location loc = UnknownLoc::get(ctx);

  // See
  // https://github.com/ROCm/ROCm-Device-Libs/blob/rocm-6.0.x/ockl/src/services.cl#L263-L361
  // for details about the following HIP device print functions.
  LLVM::LLVMFuncOp printBeginFn =
      getOrInsertFunction(moduleOp, loc, rewriter, "__ockl_printf_begin",
                          LLVM::LLVMFunctionType::get(i64_ty, {i64_ty}));
  LLVM::LLVMFuncOp printStrFn = getOrInsertFunction(
      moduleOp, loc, rewriter, "__ockl_printf_append_string_n",
      LLVM::LLVMFunctionType::get(
          i64_ty, {i64_ty, ptr_ty(ctx), /*length=*/i64_ty, /*isLast=*/i32_ty}));
  LLVM::LLVMFuncOp printArgsFn;
  if (!args.empty()) {
    printArgsFn = getOrInsertFunction(
        moduleOp, loc, rewriter, "__ockl_printf_append_args",
        LLVM::LLVMFunctionType::get(
            i64_ty, {i64_ty, /*numArgs=*/i32_ty, i64_ty, i64_ty, i64_ty, i64_ty,
                     i64_ty, i64_ty, i64_ty, /*isLast=*/i32_ty}));
  }

  // Emit the intrinsic function call to begin the printf.
  Value zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, i64_ty, 0);
  auto printfBeginCall =
      rewriter.create<LLVM::CallOp>(loc, printBeginFn, zeroI64);
  Value printfDesc = printfBeginCall.getResult();

  // Emit the intrinsic function call to handle the printf format string.
  Value oneI32 = i32_val(1);
  Value zeroI32 = i32_val(0);
  Value formatStrLen =
      rewriter.create<LLVM::ConstantOp>(loc, i64_ty, formatStrByteCount);
  auto printfFormatCall = rewriter.create<LLVM::CallOp>(
      loc, printStrFn,
      ValueRange{printfDesc, formatStrStart, formatStrLen,
                 args.empty() ? oneI32 : zeroI32});
  printfDesc = printfFormatCall.getResult();

  // Emit the intrinsic function call to handle arguments iteratively.
  // We can only handle at most 7 values each time.
  constexpr size_t kArgsPerGroup = 7;
  for (size_t group = 0; group < args.size(); group += kArgsPerGroup) {
    size_t bound = std::min(group + kArgsPerGroup, args.size());
    size_t numArgs = bound - group;

    SmallVector<mlir::Value, 2 + kArgsPerGroup + 1> arguments;
    arguments.push_back(printfDesc);
    arguments.push_back(i32_val(numArgs));
    for (size_t i = group; i < bound; ++i) {
      arguments.push_back(printfPromoteValue(rewriter, args[i]));
    }
    // Pad out to 7 arguments since the function always needs 7 args.
    for (size_t extra = numArgs; extra < kArgsPerGroup; ++extra) {
      arguments.push_back(zeroI64);
    }

    auto isLast = (bound == args.size()) ? oneI32 : zeroI32;
    arguments.push_back(isLast);
    printfDesc = call(printArgsFn, arguments).getResult();
  }
}

} // namespace AMD
