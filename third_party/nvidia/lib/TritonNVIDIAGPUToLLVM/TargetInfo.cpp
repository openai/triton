#include "TargetInfo.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

using mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
namespace {
Value computeStMatrixAddr(Value laneId, int matStride, Location loc,
                          RewriterBase &rewriter, int swizzleByteWidth) {
  Value rowInMat = urem(laneId, i32_val(8)); // row in the 8x8 matrix
  // linear index of the matrix in the 2x2 matrices
  // Decompose matIndex => s_0, s_1, that is the coordinate in 2x2 matrices in
  // a warp.
  Value matIndex = udiv(laneId, i32_val(8));
  Value s0 = urem(matIndex, i32_val(2));
  Value s1 = udiv(matIndex, i32_val(2));
  if (swizzleByteWidth >= 32)
    s1 = xor_(s1, and_(laneId, i32_val(1)));
  Value mIndex = add(rowInMat, mul(s0, i32_val(8)));
  int m8n8Stride = 8;
  Value offset =
      add(mul(mIndex, i32_val(matStride)), mul(s1, i32_val(m8n8Stride)));
  return offset;
}

void stMatrixm8n8x4(Value offset, ArrayRef<Value> vals, int indexOffset,
                    Value smemBase, Type elemTy, Location loc,
                    RewriterBase &rewriter) {
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
    ArrayRef<unsigned> origRepShape, Location loc, RewriterBase &rewriter,
    int swizzlingByteWidth) {
  auto shapePerCTA = getShapePerCTA(tensorTy);
  auto mmaLayout = mlir::cast<NvidiaMmaEncodingAttr>(tensorTy.getEncoding());
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
  int numBoxes = 1;
  if (swizzlingByteWidth == 128) {
    int contigDimSizeInByte =
        origRepShape[1] * elemTy.getIntOrFloatBitWidth() / 8;
    numBoxes = ceil<int>(contigDimSizeInByte, 128);
  }
  SmallVector<unsigned> boxShape = {paddedRepShape[0], paddedRepShape[1]};
  boxShape[1] = boxShape[1] / numBoxes;
  Value thread = getThreadId(rewriter, loc);
  Value warp = udiv(thread, i32_val(32));
  Value lane = urem(thread, i32_val(32));

  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warp, warpsPerCTA);

  // Compute the relative offset for each lane.
  Value stMatrixLaneOffset =
      computeStMatrixAddr(lane, boxShape[1], loc, rewriter, swizzlingByteWidth);
  multiDimWarpId[0] = mul(multiDimWarpId[0], i32_val(mmaShape[0]));
  multiDimWarpId[1] = mul(multiDimWarpId[1], i32_val(mmaShape[1]));
  SmallVector<Value> multiDimOffsetWrapped = getWrappedMultiDimOffset(
      rewriter, loc, multiDimWarpId, boxShape, shapePerCTATile, shapePerCTA);
  Value relativeOffset =
      linearize(rewriter, loc, multiDimOffsetWrapped, boxShape, order);
  relativeOffset = add(relativeOffset, stMatrixLaneOffset);
  int indexOffset = 0;
  int m8n8x4Stride = 16;
  int numNChunk = mmaShape[1] / m8n8x4Stride;
  unsigned totalNumElements = product(origRepShape);
  numNChunk = numNChunk / numBoxes;
  for (int m = 0; m < numRep[0]; m++) {
    for (int n = 0; n < numRep[1]; n++) {
      for (int box = 0; box < numBoxes; box++) {
        for (int k = 0; k < numNChunk; k++) {
          Value kOffset;
          if (swizzlingByteWidth >= 64) {
            int swizzleBits = swizzlingByteWidth == 128 ? 6 : 2;
            Value o = lshr(and_(lane, i32_val(swizzleBits)), i32_val(1));
            Value kV = xor_(o, i32_val(k));
            kOffset = mul(kV, i32_val(m8n8x4Stride));
          } else {
            kOffset = i32_val(k * m8n8x4Stride);
          }
          Value addr = add(relativeOffset,
                           i32_val(n * instrN + m * instrM * boxShape[1] +
                                   box * (totalNumElements / numBoxes)));
          addr = add(addr, kOffset);

          stMatrixm8n8x4(addr, inVals, indexOffset, smemBase, elemTy, loc,
                         rewriter);
          indexOffset += 8;
        }
      }
    }
  }
}

bool isStMatrixCompatible(RankedTensorType tensorTy, int swizzlingByteWidth) {
  auto mmaLayout =
      mlir::dyn_cast<NvidiaMmaEncodingAttr>(tensorTy.getEncoding());
  if (!mmaLayout || !mmaLayout.isHopper())
    return false;
  if (tensorTy.getElementType().getIntOrFloatBitWidth() != 16)
    return false;
  if (swizzlingByteWidth > 0 && mmaLayout.getInstrShape()[1] < 64)
    return false;
  if (swizzlingByteWidth != 0 && swizzlingByteWidth != 32 &&
      swizzlingByteWidth != 64 && swizzlingByteWidth != 128)
    return false;
  return true;
}

// declare vprintf(i8*, i8*) as external function
LLVM::LLVMFuncOp getVprintfDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("vprintf");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *context = rewriter.getContext();

  SmallVector<Type> argsType{ptr_ty(context), ptr_ty(context)};
  auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                           funcType);
}

// extend integer to int32, extend float to float64
// this comes from vprintf alignment requirements.
std::pair<Type, Value> printfPromoteValue(RewriterBase &rewriter, Value value) {
  auto *context = rewriter.getContext();
  auto type = value.getType();
  Value newOp = value;
  Type newType = type;
  auto loc = UnknownLoc::get(context);

  bool isUnsigned = type.isUnsignedInteger();
  if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
    if (isUnsigned) {
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

LLVM::LLVMFuncOp getAssertfailDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("__assertfail");
  {
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);
  }
  // void __assert_fail(const char * assertion, const char * file, unsigned
  // int line, const char * function);
  auto *ctx = rewriter.getContext();
  SmallVector<Type> argsType{ptr_ty(ctx), ptr_ty(ctx), i32_ty, ptr_ty(ctx),
                             rewriter.getIntegerType(sizeof(size_t) * 8)};
  auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto funcOp = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx),
                                                  funcName, funcType);

  funcOp.setPassthroughAttr(
      ArrayAttr::get(ctx, StringAttr::get(ctx, "noreturn")));
  return funcOp;
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
  auto intType = dyn_cast<IntegerType>(reduceOp->getResultTypes()[0]);
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

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return rewriter.create<triton::nvgpu::ClusterCTAIdOp>(loc,
                                                        rewriter.getI32Type());
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  Value threadMask = int_val(type.getIntOrFloatBitWidth(), -1);
  return rewriter.create<NVVM::VoteBallotOp>(loc, type, threadMask, cmp);
}

static Value mapa(RewriterBase &rewriter, Location loc, Value ptr, Value ctaid,
                  Value pred) {
  PTXBuilder builder;
  (*builder.create<>("mapa.shared::cluster.u32"))(
      builder.newOperand("=r"), //
      builder.newAddrOperand(ptr, "r"), builder.newAddrOperand(ctaid, "r"))
      .predicate(pred, "b");
  return builder.launch(rewriter, loc, i32_ty, /*hasSideEffects=*/false);
}

static std::string getConstraintForBitwidth(unsigned bitwidth) {
  switch (bitwidth) {
  case 8:
  case 16:
    return "h";
  case 32:
    return "r";
  case 64:
    return "l";
  default:
    llvm_unreachable("unsupported bitwidth");
  }
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");

  if (!isa<VectorType>(val.getType())) {
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, {val}, rewriter),
                 pred);
    return;
  }

  auto vecTy = cast<VectorType>(val.getType());
  Type elemTy = vecTy.getElementType();
  unsigned vec = vecTy.getNumElements();
  unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_32(vec));

  if (elemBitwidth < 8) {
    assert(vec == 1 &&
           "don't know how to load/store vectors of sub-byte elems");
    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (Value &v : vals) {
      v = zext(int_ty(8), bitcast(v, int_ty(elemBitwidth)));
    }
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, vals, rewriter),
                 pred);
    return;
  }

  if (!elemTy.isInteger()) {
    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (Value &v : vals) {
      v = bitcast(v, int_ty(elemBitwidth));
    }
    storeDShared(rewriter, loc, ptr, ctaId, packLLVector(loc, vals, rewriter),
                 pred);
    return;
  }

  // load/store ops only support v2 and v4.  If the vector width is larger than
  // 4, we have two strategies for dealing with it.
  //  1. If the element type is smaller than b32, store b32's instead.
  //  2. Otherwise, split the store into multiple stores.
  if (vec > 4 && elemBitwidth < 32) {
    assert(llvm::isPowerOf2_32(vec));
    int elemsPerPack = 32 / elemBitwidth;
    SmallVector<Value> oldVals = unpackLLVector(loc, val, rewriter);

    SmallVector<Value> newVals;
    for (int i = 0; i < vec / elemsPerPack; i++) {
      Value v = packLLVector(
          loc, ArrayRef(oldVals).slice(i * elemsPerPack, elemsPerPack),
          rewriter);
      newVals.push_back(bitcast(v, i32_ty));
    }
    storeDShared(rewriter, loc, ptr, ctaId,
                 packLLVector(loc, newVals, rewriter), pred);
    return;
  }

  if (vec * elemBitwidth > 128) {
    assert(llvm::isPowerOf2_32(vec));
    assert(elemBitwidth == 32 || elemBitwidth == 64);
    int maxVec = 128 / elemBitwidth;

    auto newVecTy = vec_ty(elemTy, maxVec);
    SmallVector<Value> vals = unpackLLVector(loc, val, rewriter);
    for (int i = 0; i < vec / maxVec; i++) {
      auto newPtr = gep(ptr.getType(), elemTy, ptr, i32_val(i * maxVec),
                        /*inbounds=*/true);
      storeDShared(
          rewriter, loc, newPtr, ctaId,
          packLLVector(loc, ArrayRef(vals).slice(i * maxVec, maxVec), rewriter),
          pred);
    }
    return;
  }

  // At this point we're committed to doing the store!
  assert(elemBitwidth >= 8);
  assert(elemTy.isInteger());
  assert(1 <= vec && vec <= 4);
  assert(vec * elemBitwidth <= 128);

  // Get pointer to remote shared memory if needed.
  if (ctaId.has_value()) {
    ptr = mapa(rewriter, loc, ptr, *ctaId, pred);
  }

  PTXBuilder builder;
  auto st = builder.create<>("st")
                ->o("shared::cta", ctaId.has_value())
                .o("shared", !ctaId.has_value())
                .b(elemBitwidth)
                .v(vec, /*predicate=*/vec > 1);
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");

  PTXBuilder::Operand *valOpr;
  std::string constraint = getConstraintForBitwidth(elemBitwidth);
  if (vec > 1) {
    SmallVector<std::pair<Value, std::string>> vecVals;
    for (int i = 0; i < vec; i++) {
      vecVals.push_back({extract_element(val, i32_val(i)), constraint});
    }
    valOpr = builder.newListOperand(vecVals);
  } else {
    valOpr = builder.newOperand(val, constraint);
  }
  st(ptrOpr, valOpr).predicate(pred, "b");
  builder.launch(rewriter, loc, void_ty(ctx));
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type loadTy,
                              Value pred) const {
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for load_dsmem");

  if (!isa<VectorType>(loadTy)) {
    SmallVector<Value> values = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, vec_ty(loadTy, 1), pred),
        rewriter);
    assert(values.size() == 1);
    return values[0];
  }

  auto vecTy = cast<VectorType>(loadTy);
  Type elemTy = vecTy.getElementType();
  unsigned vec = vecTy.getNumElements();
  unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
  assert(llvm::isPowerOf2_32(vec));

  if (elemBitwidth < 8) {
    assert(vec == 1 &&
           "don't know how to load/store vectors of sub-byte elems");
    SmallVector<Value> vals = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, int_ty(8), pred), rewriter);
    assert(vals.size() == 1);
    return bitcast(trunc(int_ty(elemBitwidth), vals[0]), elemTy);
  }

  // We only know how to load integers.
  if (!elemTy.isInteger()) {
    Type newLoadTy = vec_ty(int_ty(elemBitwidth), vec);
    SmallVector<Value> vals = unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, newLoadTy, pred), rewriter);
    for (Value &v : vals) {
      v = bitcast(v, elemTy);
    }
    return packLLVector(loc, vals, rewriter);
  }

  // load/store ops only support v2 and v4.  If the vector width is larger than
  // 4, we have two strategies for dealing with it.
  //  1. If the element type is smaller than b32, load b32's instead.
  //  2. Otherwise, split the load into multiple loads.
  if (vec > 4 && elemBitwidth < 32) {
    int newVec = vec / (32 / elemBitwidth);
    auto newVecTy = vec_ty(i32_ty, newVec);
    auto res = loadDShared(rewriter, loc, ptr, ctaId, newVecTy, pred);

    // Unpack the b32's into the original vector type.
    SmallVector<Value> vals;
    for (Value v : unpackLLVector(loc, res, rewriter)) {
      Value vv = bitcast(v, vec_ty(elemTy, 32 / elemBitwidth));
      for (Value vvv : unpackLLVector(loc, vv, rewriter)) {
        vals.push_back(vvv);
      }
    }
    return packLLVector(loc, vals, rewriter);
  }

  if (vec * elemBitwidth > 128) {
    assert(elemBitwidth == 32 || elemBitwidth == 64);
    assert(llvm::isPowerOf2_32(vec));
    int maxVec = 128 / elemBitwidth;

    SmallVector<Value> vals;
    for (int i = 0; i < vec / maxVec; i++) {
      auto newPtr = gep(ptr.getType(), elemTy, ptr, i32_val(i * maxVec),
                        /*inbounds=*/true);
      auto newVal = loadDShared(rewriter, loc, newPtr, ctaId,
                                vec_ty(elemTy, maxVec), pred);
      for (Value v : unpackLLVector(loc, newVal, rewriter)) {
        vals.push_back(v);
      }
    }
    return packLLVector(loc, vals, rewriter);
  }

  // At this point we're committed to actually do the load!
  assert(elemBitwidth >= 8);
  assert(elemTy.isInteger());
  assert(1 <= vec && vec <= 4);
  assert(vec * elemBitwidth <= 128);

  // Get pointer to remote shared memory if needed.
  if (ctaId.has_value()) {
    ptr = mapa(rewriter, loc, ptr, *ctaId, pred);
  }

  PTXBuilder builder;
  auto ld = builder.create<>("ld")
                ->o("shared::cta", ctaId.has_value())
                .o("shared", !ctaId.has_value())
                .v(vec, /*predicate=*/vec > 1)
                .b(elemBitwidth);

  std::string elemConstraint = "=" + getConstraintForBitwidth(elemBitwidth);
  auto *outOpr = vec == 1 ? builder.newOperand(elemConstraint)
                          : builder.newListOperand(vec, elemConstraint);
  ld(outOpr, builder.newAddrOperand(ptr, "r")).predicate(pred, "b");

  Type resultTy =
      vec == 1 ? Type(int_ty(elemBitwidth))
               : Type(struct_ty(SmallVector<Type>(vec, int_ty(elemBitwidth))));
  Value load = builder.launch(rewriter, loc, resultTy, /*hasSideEffects=*/true);

  SmallVector<Value> resultVals = unpackLLElements(loc, load, rewriter);
  return packLLVector(loc, resultVals, rewriter);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::NVIDIA::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::NVIDIA::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::NVIDIA::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::NVIDIA::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, int axis) const {
  return LLVM::NVIDIA::llGetPid(loc, rewriter, moduleOp, axis);
}
bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
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
        unsigned bitwidth = cast<IntegerType>(acc[i].getType()).getWidth();
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
    RewriterBase &rewriter, Location loc, Value smemBase,
    SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
    ArrayRef<unsigned> outOrd, unsigned accumNumReplicates,
    int swizzlingByteWidth) const {
  if (isStMatrixCompatible(srcTy, swizzlingByteWidth) &&
      accumNumReplicates == 1 && outOrd[0] == 1 && paddedRepShape[1] % 8 == 0) {
    storeDistributedToSharedWithStMatrix(srcTy, elemTy, vals, smemBase,
                                         paddedRepShape, origRepShape, loc,
                                         rewriter, swizzlingByteWidth);
    return true;
  }
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__nv_umulhi" : "__nv_umul64hi";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args) const {
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

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto funcOp = getAssertfailDeclaration(rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  llvm::SmallString<64> messageString(message), fileString(file),
      funcString(func);
  messageString.push_back('\0');
  fileString.push_back('\0');
  funcString.push_back('\0');
  Value messageStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertMessage_", messageString);
  Value fileStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertFile_", fileString);
  Value funcStringVal =
      LLVM::addStringToModule(loc, rewriter, "assertFunc_", funcString);
  Value lineNumber = i32_val(line);
  Value charSize = int_val(sizeof(size_t) * 8, sizeof(char));
  SmallVector<Value> operands = {messageStringVal, fileStringVal, lineNumber,
                                 funcStringVal, charSize};
  call(funcOp, operands);
}

} // namespace mlir::triton::NVIDIA
