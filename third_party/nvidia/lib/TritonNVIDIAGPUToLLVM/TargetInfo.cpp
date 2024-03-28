#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;

using mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
namespace {
Value computeStMatrixAddr(Value laneId, int matStride, Location loc,
                          ConversionPatternRewriter &rewriter) {
  Value rowInMat = urem(laneId, i32_val(8)); // row in the 8x8 matrix
  // linear index of the matrix in the 2x2 matrices
  // Decompose matIndex => s_0, s_1, that is the coordinate in 2x2 matrices in
  // a warp.
  Value matIndex = udiv(laneId, i32_val(8));
  Value s0 = urem(matIndex, i32_val(2));
  Value s1 = udiv(matIndex, i32_val(2));
  Value mIndex = add(rowInMat, mul(s0, i32_val(8)));
  int m8n8Stride = 8;
  Value offset =
      add(mul(mIndex, i32_val(matStride)), mul(s1, i32_val(m8n8Stride)));
  return offset;
}

void stMatrixm8n8x4(Value offset, ArrayRef<Value> vals, int indexOffset,
                    Value smemBase, Type elemTy, Location loc,
                    ConversionPatternRewriter &rewriter) {
  SmallVector<Value> inputs;
  auto prTy = ptr_ty(rewriter.getContext(), 3);
  // Pack the input into 2xf16
  Type packedTy = vec_ty(vals[0].getType(), 2);
  for (int i = 0; i < 4; i++) {
    Value input = undef(packedTy);
    for (int j = 0; j < 2; j++) {
      input = insert_element(packedTy, input, vals[indexOffset + i * 2 + j],
                             i32_val(j));
    }
    inputs.push_back(bitcast(input, i32_ty));
  }
  Value addr = gep(smemBase.getType(), elemTy, smemBase, offset);
  rewriter.create<triton::nvgpu::StoreMatrixOp>(loc, addr, inputs);
}
void storeDistributedToSharedWithStMatrix(
    RankedTensorType tensorTy, Type elemTy, SmallVector<Value> &inVals,
    Value smemBase, ArrayRef<unsigned> paddedRepShape,
    ArrayRef<unsigned> origRepShape, Location loc,
    ConversionPatternRewriter &rewriter) {
  auto shapePerCTA = getShapePerCTA(tensorTy);
  auto mmaLayout = tensorTy.getEncoding().cast<NvidiaMmaEncodingAttr>();
  auto order = triton::gpu::getOrder(mmaLayout);
  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
  auto shapePerCTATile = getShapePerCTATile(mmaLayout);
  ArrayRef<unsigned> mmaShape = mmaLayout.getInstrShape();
  // 4xm8n8 matches exactly the size of 1 warp of wgmma layout for 16bit type
  // and has a shape of 16x16.
  int instrN = mmaShape[1] * warpsPerCTA[1];
  int instrM = mmaShape[0] * warpsPerCTA[0];
  std::array<int, 2> numRep = {ceil((int)origRepShape[0], instrM),
                               ceil((int)origRepShape[1], instrN)};

  Value thread = getThreadId(rewriter, loc);
  Value warp = udiv(thread, i32_val(32));
  Value lane = urem(thread, i32_val(32));

  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warp, warpsPerCTA);

  // Compute the relative offset for each lane.
  Value stMatrixLaneOffset =
      computeStMatrixAddr(lane, paddedRepShape[1], loc, rewriter);
  multiDimWarpId[0] = mul(multiDimWarpId[0], i32_val(mmaShape[0]));
  multiDimWarpId[1] = mul(multiDimWarpId[1], i32_val(mmaShape[1]));
  SmallVector<Value> multiDimOffsetWrapped =
      getWrappedMultiDimOffset(rewriter, loc, multiDimWarpId, origRepShape,
                               shapePerCTATile, shapePerCTA);
  Value relativeOffset =
      linearize(rewriter, loc, multiDimOffsetWrapped, paddedRepShape, order);
  relativeOffset = add(relativeOffset, stMatrixLaneOffset);
  int indexOffset = 0;
  int m8n8x4Stride = 16;
  int numNChunk = mmaShape[1] / m8n8x4Stride;
  for (int m = 0; m < numRep[0]; m++) {
    for (int n = 0; n < numRep[1]; n++) {
      for (int k = 0; k < numNChunk; k++) {
        Value addr =
            add(relativeOffset, i32_val(k * m8n8x4Stride + n * instrN +
                                        m * instrM * paddedRepShape[1]));
        stMatrixm8n8x4(addr, inVals, indexOffset, smemBase, elemTy, loc,
                       rewriter);
        indexOffset += 8;
      }
    }
  }
}

bool isStMatrixCompatible(RankedTensorType tensorTy) {
  auto mmaLayout = tensorTy.getEncoding().dyn_cast<NvidiaMmaEncodingAttr>();
  if (!mmaLayout || !mmaLayout.isHopper())
    return false;
  if (tensorTy.getElementType().getIntOrFloatBitWidth() != 16)
    return false;
  return true;
}

// declare vprintf(i8*, i8*) as external function
LLVM::LLVMFuncOp getVprintfDeclaration(ConversionPatternRewriter &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("vprintf");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();

  SmallVector<Type> argsType{ptr_ty(context), ptr_ty(context)};
  auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                           funcType);
}

// extend integer to int32, extend float to float64
// this comes from vprintf alignment requirements.
std::pair<Type, Value> printfPromoteValue(ConversionPatternRewriter &rewriter,
                                          Value value) {
  auto *context = rewriter.getContext();
  auto type = value.getType();
  Value newOp = value;
  Type newType = type;
  auto loc = UnknownLoc::get(context);

  bool bUnsigned = type.isUnsignedInteger();
  if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
    if (bUnsigned) {
      newType = ui32_ty;
      newOp = zext(newType, value);
    } else {
      newType = i32_ty;
      newOp = sext(newType, value);
    }
  } else if (type.isBF16() || type.isF16() || type.isF32()) {
    newType = f64_ty;
    newOp = fpext(newType, value);
  }

  return {newType, newOp};
}
} // namespace

namespace mlir::triton::NVIDIA {

// Check if the reduction can use a redux op and return the kind.
static std::optional<NVVM::ReduxKind> matchReduxKind(triton::ReduceOp op,
                                                     int computeCapability) {
  if (computeCapability < 80)
    return std::nullopt;
  if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    return std::nullopt;
  Block *block = &(*op.getCombineOp().begin());
  Operation *yield = block->getTerminator();
  Operation *reduceOp = yield->getOperand(0).getDefiningOp();
  if (!reduceOp || reduceOp->getNumOperands() != 2 ||
      reduceOp->getNumResults() != 1)
    return std::nullopt;
  auto intType = reduceOp->getResultTypes()[0].dyn_cast<IntegerType>();
  if (!intType || intType.getWidth() > 32)
    return std::nullopt;
  if (reduceOp->getOperand(0) != block->getArgument(0) ||
      reduceOp->getOperand(1) != block->getArgument(1))
    return std::nullopt;
  if (isa<arith::AddIOp>(reduceOp))
    return NVVM::ReduxKind::ADD;
  if (isa<arith::AndIOp>(reduceOp))
    return NVVM::ReduxKind::AND;
  if (isa<arith::OrIOp>(reduceOp))
    return NVVM::ReduxKind::OR;
  if (isa<arith::XOrIOp>(reduceOp))
    return NVVM::ReduxKind::XOR;
  if (isa<arith::MinSIOp>(reduceOp))
    return NVVM::ReduxKind::MIN;
  if (isa<arith::MinUIOp>(reduceOp))
    return NVVM::ReduxKind::UMIN;
  if (isa<arith::MaxSIOp>(reduceOp))
    return NVVM::ReduxKind::MAX;
  if (isa<arith::MaxUIOp>(reduceOp))
    return NVVM::ReduxKind::UMAX;
  return std::nullopt;
}

bool TargetInfo::supportMaximumMinimum() const {
  return computeCapability >= 80;
}
Value TargetInfo::ballot(ConversionPatternRewriter &rewriter, Location loc,
                         Type type, Value cmp) const {
  Value threadMask = int_val(type.getIntOrFloatBitWidth(), -1);
  return rewriter.create<NVVM::VoteBallotOp>(loc, type, threadMask, cmp);
}
Value TargetInfo::storeShared(ConversionPatternRewriter &rewriter, Location loc,
                              Value ptr, Value val, Value pred) const {
  MLIRContext *ctx = rewriter.getContext();
  unsigned bits = std::max(8u, val.getType().getIntOrFloatBitWidth());
  const char *c = bits == 64 ? "l" : (bits == 16 ? "h" : "r");

  PTXBuilder builder;
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto *valOpr = builder.newOperand(val, c);
  auto &st = builder.create<>("st")->shared().b(bits);
  st(ptrOpr, valOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, void_ty(ctx));
}

Value TargetInfo::loadShared(ConversionPatternRewriter &rewriter, Location loc,
                             Value ptr, Type elemTy, Value pred) const {
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = ptr.getType().cast<LLVM::LLVMPointerType>();
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for loadShared");
  unsigned bitwidth = std::max(8u, elemTy.getIntOrFloatBitWidth());

  const char *c = bitwidth == 64 ? "=l" : (bitwidth == 16 ? "=h" : "=r");

  PTXBuilder builder;
  auto *dOpr = builder.newOperand(c);
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto &ld = builder.create<>("ld")->shared().b(bitwidth);
  ld(dOpr, ptrOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, elemTy);
}

Value TargetInfo::shuffleXor(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return LLVM::NVIDIA::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, int i) const {
  return LLVM::NVIDIA::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return LLVM::NVIDIA::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, Value i) const {
  return LLVM::NVIDIA::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::programId(Location loc, ConversionPatternRewriter &rewriter,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::NVIDIA::llGetPid(loc, rewriter, moduleOp, axis);
}
bool TargetInfo::warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce) const {
  if (auto kind = matchReduxKind(op, computeCapability)) {
    // Based on benchmarking on A100 redux op gives a speed up only when doing
    // a single reduction (not partitioned) and when the mask is static.
    // Therefore we currently only enable it to reduce across all the lanes.
    if (numLaneToReduce == 32) {
      assert(acc.size() == 1);
      Value mask = i32_val(0xFFFFFFFF);
      // Even though we currently don't use redux for partitioned reduction
      // the code below supports it in case we want to tweak the heuristic.
      if (numLaneToReduce < 32) {
        // For partitioned reduction we need to calculate the mask so that
        // each group of numLaneToReduce threads has the correct mask.
        unsigned bitmask = (1 << numLaneToReduce) - 1;
        Value threadId = getThreadId(rewriter, loc);
        Value laneId = urem(threadId, i32_val(32));
        mask = shl(i32_val(bitmask),
                   and_(laneId, i32_val(~(numLaneToReduce - 1))));
      }
      for (unsigned i = 0; i < acc.size(); ++i) {
        unsigned bitwidth = acc[i].getType().cast<IntegerType>().getWidth();
        if (bitwidth < 32) {
          if (*kind == NVVM::ReduxKind::MIN || *kind == NVVM::ReduxKind::MAX)
            acc[i] = sext(i32_ty, acc[i]);
          else
            acc[i] = zext(i32_ty, acc[i]);
        }
        acc[i] = rewriter.create<NVVM::ReduxOp>(loc, acc[i].getType(), acc[0],
                                                *kind, mask);
        if (bitwidth < 32)
          acc[i] = trunc(int_ty(bitwidth), acc[i]);
      }
      return true;
    }
  }
  return false;
}
bool TargetInfo::processReplicaUsingStMatrix(
    ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
    SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
    ArrayRef<unsigned> outOrd, unsigned accumNumReplicates) const {
  if (isStMatrixCompatible(srcTy) && accumNumReplicates == 1 &&
      outOrd[0] == 1 && paddedRepShape[1] % 8 == 0) {
    storeDistributedToSharedWithStMatrix(srcTy, elemTy, vals, smemBase,
                                         paddedRepShape, origRepShape, loc,
                                         rewriter);
    return true;
  }
  return false;
}

void TargetInfo::printf(Value formatStrStart, int /*formatStrByteCount*/,
                        ValueRange args,
                        ConversionPatternRewriter &rewriter) const {
  auto *ctx = rewriter.getContext();
  Type ptr = ptr_ty(ctx);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto funcOp = getVprintfDeclaration(rewriter);
  auto loc = UnknownLoc::get(ctx);

  Value one = i32_val(1);
  Value zero = i32_val(0);

  Value bufferPtr = null(ptr);

  SmallVector<Value, 16> newArgs;
  if (args.size() >= 1) {
    SmallVector<Type> argTypes;
    for (auto arg : args) {
      Type newType;
      Value newArg;
      std::tie(newType, newArg) = printfPromoteValue(rewriter, arg);
      argTypes.push_back(newType);
      newArgs.push_back(newArg);
    }

    Type structTy = LLVM::LLVMStructType::getLiteral(ctx, argTypes);
    auto allocated =
        rewriter.create<LLVM::AllocaOp>(loc, ptr_ty(ctx), structTy, one,
                                        /*alignment=*/0);

    for (const auto &entry : llvm::enumerate(newArgs)) {
      auto index = i32_val(entry.index());
      auto fieldPtr =
          gep(ptr_ty(ctx), structTy, allocated, ArrayRef<Value>{zero, index});
      store(entry.value(), fieldPtr);
    }
    bufferPtr = bitcast(allocated, ptr);
  }

  SmallVector<Value> operands{formatStrStart, bufferPtr};
  call(funcOp, operands);
}

} // namespace mlir::triton::NVIDIA
