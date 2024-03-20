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
  mlir::Type llvmI8 = i8_ty;
  auto ptrType = ptr_ty(ctx);
  mlir::Type llvmI32 = i32_ty;
  mlir::Type llvmI64 = i64_ty;
  mlir::Location loc = UnknownLoc::get(ctx);

  // See
  // https://github.com/ROCm/ROCm-Device-Libs/blob/rocm-6.0.x/ockl/src/services.cl#L263-L361
  // for details about the following HIP device print functions.
  LLVM::LLVMFuncOp printBeginFn =
      getOrInsertFunction(moduleOp, loc, rewriter, "__ockl_printf_begin",
                          LLVM::LLVMFunctionType::get(llvmI64, {llvmI64}));
  LLVM::LLVMFuncOp printStrFn = getOrInsertFunction(
      moduleOp, loc, rewriter, "__ockl_printf_append_string_n",
      LLVM::LLVMFunctionType::get(
          llvmI64, {llvmI64, ptrType, /*length=*/llvmI64, /*isLast=*/llvmI32}));
  LLVM::LLVMFuncOp printArgsFn;
  if (!args.empty()) {
    printArgsFn = getOrInsertFunction(
        moduleOp, loc, rewriter, "__ockl_printf_append_args",
        LLVM::LLVMFunctionType::get(
            llvmI64, {llvmI64, /*numArgs=*/llvmI32, llvmI64, llvmI64, llvmI64,
                      llvmI64, llvmI64, llvmI64, llvmI64, /*isLast=*/llvmI32}));
  }

  /// Start the printf hostcall
  Value zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, llvmI64, 0);
  auto printfBeginCall =
      rewriter.create<LLVM::CallOp>(loc, printBeginFn, zeroI64);
  Value printfDesc = printfBeginCall.getResult();

  Value formatStrLen =
      rewriter.create<LLVM::ConstantOp>(loc, llvmI64, formatStrByteCount);
  Value oneI32 = i32_val(1);
  Value zeroI32 = i32_val(0);

  auto appendFormatCall = rewriter.create<LLVM::CallOp>(
      loc, printStrFn,
      ValueRange{printfDesc, formatStrStart, formatStrLen,
                 args.empty() ? oneI32 : zeroI32});
  printfDesc = appendFormatCall.getResult();

  // __ockl_printf_append_args takes 7 values per append call
  constexpr size_t argsPerAppend = 7;
  size_t nArgs = args.size();
  for (size_t group = 0; group < nArgs; group += argsPerAppend) {
    size_t bound = std::min(group + argsPerAppend, nArgs);
    size_t numArgsThisCall = bound - group;

    SmallVector<mlir::Value, 2 + argsPerAppend + 1> arguments;
    arguments.push_back(printfDesc);
    arguments.push_back(i32_val(numArgsThisCall));
    for (size_t i = group; i < bound; ++i) {
      Value arg = args[i];
      if (auto floatType = dyn_cast<FloatType>(arg.getType())) {
        if (!floatType.isF64())
          arg = fpext(f64_ty, arg);
        arg = bitcast(arg, llvmI64);
      }
      if (arg.getType().getIntOrFloatBitWidth() != 64)
        arg = zext(llvmI64, arg);

      arguments.push_back(arg);
    }
    // Pad out to 7 arguments since the hostcall always needs 7
    for (size_t extra = numArgsThisCall; extra < argsPerAppend; ++extra) {
      arguments.push_back(zeroI64);
    }

    auto isLast = (bound == nArgs) ? oneI32 : zeroI32;
    arguments.push_back(isLast);
    printfDesc = call(printArgsFn, arguments).getResult();
  }
}

} // namespace AMD
