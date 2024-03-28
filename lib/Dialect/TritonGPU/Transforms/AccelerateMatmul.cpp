#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace {
using tt::DotOp;
using ttg::BlockedEncodingAttr;
using ttg::ConvertLayoutOp;
using ttg::DotOperandEncodingAttr;
using ttg::NvidiaMmaEncodingAttr;
using ttg::SliceEncodingAttr;

// Get the highest version supported for the hardware and the dot.
static int getMMAVersionSafe(int computeCapability, tt::DotOp op) {
  int baseVersion = 0;
  if (computeCapability < 75) {
    baseVersion = 1;
  } else if (computeCapability < 90) {
    baseVersion = 2;
  } else if (computeCapability < 100) {
    baseVersion = 3;
  } else {
    assert(false && "computeCapability not supported");
  }

  for (; baseVersion >= 1; baseVersion--) {
    if (supportMMA(op, baseVersion)) {
      return baseVersion;
    }
  }

  return 0;
}

SmallVector<unsigned>
warpsPerTileV2(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps) {
  auto rank = shape.size();
  // Early exit for batched matmul
  if (rank == 3)
    return {(unsigned)numWarps, 1, 1};

  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion() &&
           !isa<tt::TransOp>(op);
  };
  auto slices = multiRootGetSlice(dotOp, {filter}, {filter});
  bool hasChainedDot = false;
  for (Operation *op : slices) {
    if (isa<tt::DotOp>(op) && (op != dotOp)) {
      auto chainedDot = cast<tt::DotOp>(op);
      auto resTy = chainedDot.getResult().getType();
      if (resTy.getRank() != rank) {
        continue;
      }
      if (auto mmaEncoding =
              resTy.getEncoding().dyn_cast<NvidiaMmaEncodingAttr>()) {
        return ttg::getWarpsPerCTA(mmaEncoding);
      }
      hasChainedDot = true;
    }
  }
  if (hasChainedDot) {
    if (shape[0] >= shape[1]) {
      return {(unsigned)numWarps, 1};
    } else {
      return {1, (unsigned)numWarps};
    }
  }

  SmallVector<unsigned> ret(rank, 1);
  SmallVector<int64_t> shapePerWarp(rank, 1);
  shapePerWarp[rank - 1] = 8;
  shapePerWarp[rank - 2] = 16;
  // TODO (@daadaada): double-check.
  // original logic in
  // https://github.com/openai/triton/blob/master/lib/codegen/analysis/layout.cc#L252
  // seems buggy for shape = [32, 16] ?
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

SmallVector<unsigned, 2>
warpsPerTileV3(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps,
               const SmallVector<unsigned, 3> &instrShape) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getResult(), &slices);
  if (llvm::find_if(slices, [](Operation *op) { return isa<tt::DotOp>(op); }) !=
      slices.end())
    return {(unsigned)numWarps, 1};

  // For MMAv3, the smallest indivisible unit of warp shape is (4, 1).
  SmallVector<unsigned, 2> ret = {4, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, instrShape[1]};
  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] > shapePerWarp[0] * ret[0]) {
      ret[0] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

class BlockedToMMA : public mlir::RewritePattern {
  int computeCapability;
  mutable int mmaV1Counter{}; // used to generate ID for MMAv1 encoding
  mutable llvm::DenseMap<Operation *, unsigned> dotOpInstNs;

  static bool bwdFilter(Operation *op) {
    return op->getNumOperands() == 1 &&
           (isa<tt::FpToFpOp, tt::BitcastOp, ttg::ConvertLayoutOp>(op) ||
            op->getDialect()->getTypeID() ==
                mlir::TypeID::get<arith::ArithDialect>());
  }

  // Finds the first different bitwidth in the chain of shape-preserving
  // unary ops that x depends on.
  // There are two primary scenarios:
  // (1) Upcasting: A sequence such as loading an fp16, followed by arithmetic
  // operations, then bitcasting to fp32, and finally computing in fp32.
  // (2) Downcasting: This might involve loading an fp32, performing arithmetic
  // operations, bitcasting to fp16, and finally computing in fp16.
  // In the upcasting scenario, element reordering converts the original
  // elements distribution to the order of higher precision primitives. As a
  // result, kwidth can be the bitwidth of the lower precision primitive.
  // Conversely, in the downcasting scenario, no reordering is performed,
  // making it directory use the lower precision primitive.
  static int computeOrigBitWidth(Value x) {
    int finalBitWidth = getElementTypeOrSelf(x).getIntOrFloatBitWidth();
    int origBitWidth = finalBitWidth;
    SetVector<Operation *> slice;
    mlir::BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = bwdFilter;
    getBackwardSlice(x, &slice, opt);
    for (auto op : slice) {
      if (Value arg = op->getOperand(0))
        if (auto argTy = arg.getType().dyn_cast<RankedTensorType>()) {
          auto argBitWidth = argTy.getElementType().getIntOrFloatBitWidth();
          if (argBitWidth != origBitWidth) {
            origBitWidth = std::min<int>(origBitWidth, argBitWidth);
            break;
          }
        }
    }
    return origBitWidth;
  }

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context),
        computeCapability(computeCapability) {}

  static SmallVector<unsigned, 3>
  getWarpsPerTile(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int version,
                  int numWarps, const SmallVector<unsigned, 3> &instrShape) {
    switch (version) {
    case 2:
      return warpsPerTileV2(dotOp, shape, numWarps);
    case 3:
      return warpsPerTileV3(dotOp, shape, numWarps, instrShape);
    default:
      assert(false && "not supported version");
      return {0, 0};
    }
  }

  static Value getMMAv3Operand(Value v, mlir::PatternRewriter &rewriter,
                               int opIdx) {
    OpBuilder::InsertionGuard g(rewriter);
    Value arg = v;
    if (auto cvtOp = v.getDefiningOp<ttg::ConvertLayoutOp>())
      arg = cvtOp.getSrc();
    auto argType = arg.getType().cast<RankedTensorType>();
    auto eltType = argType.getElementType();
    assert(argType.getEncoding() && "unexpected tensor type");
    auto newOrder = ttg::getOrder(argType.getEncoding());

    // MMAv3 with transpose only supports f16 and bf16 data type
    // fallback to MMAv3 without transpose for other data types
    if (!eltType.isF16() && !eltType.isBF16()) {
      if (opIdx == 1) {
        newOrder = {0, 1};
      } else {
        newOrder = {1, 0};
      }
    }

    auto CTALayout = ttg::getCTALayout(argType.getEncoding());
    auto newLayout = ttg::SharedEncodingAttr::get(
        argType.getContext(), argType.getShape(), newOrder, CTALayout,
        argType.getElementType());
    auto newType = tt::MemDescType::get(argType.getShape(),
                                        argType.getElementType(), newLayout);
    rewriter.setInsertionPointAfterValue(arg);
    return rewriter.create<ttg::LocalAllocOp>(arg.getLoc(), newType, arg);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (computeCapability < 70)
      return failure();
    auto dotOp = cast<tt::DotOp>(op);
    auto ctx = op->getContext();
    // TODO: Check data-types and SM compatibility
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        oldRetType.getEncoding().isa<ttg::NvidiaMmaEncodingAttr>())
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = ttg::getShapePerCTA(oldRetType);
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());

    int versionMajor = getMMAVersionSafe(computeCapability, dotOp);
    if (!versionMajor)
      return failure();

    auto instrShape = mmaVersionToInstrShape(versionMajor, retShapePerCTA,
                                             dotOp.getA().getType());
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = dotOp.getA().getType();
    auto oldBType = dotOp.getB().getType();

    ttg::NvidiaMmaEncodingAttr mmaEnc;
    if (versionMajor == 1) {
      SetVector<Operation *> aBwdSlices, bBwdSlices;
      auto isCvt = [](Operation *op) { return isa<ConvertLayoutOp>(op); };
      mlir::BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      opt.filter = isCvt;
      getBackwardSlice(a, &aBwdSlices, opt);
      getBackwardSlice(b, &bBwdSlices, opt);
      // get the source of the first conversion found in slices
      auto getCvtArgOrder = [](Operation *op) {
        return cast<ConvertLayoutOp>(op)
            .getSrc()
            .getType()
            .getEncoding()
            .cast<BlockedEncodingAttr>()
            .getOrder();
      };
      bool isARow = true;
      bool isBRow = true;
      Operation *aOp = a.getDefiningOp();
      Operation *bOp = b.getDefiningOp();
      if (!aBwdSlices.empty())
        aOp = aBwdSlices[0];
      if (!bBwdSlices.empty())
        bOp = bBwdSlices[0];
      if (aOp)
        isARow = getCvtArgOrder(aOp)[0] == 1;
      if (bOp)
        isBRow = getCvtArgOrder(bOp)[0] == 1;

      mmaEnc = ttg::NvidiaMmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor, numWarps, CTALayout,
          instrShape, oldAType.getShape(), oldBType.getShape(), retShapePerCTA,
          isARow, isBRow, mmaV1Counter++);
    } else if (versionMajor == 2 || versionMajor == 3) {
      int versionMinor = computeCapability == 75 ? 1 : 0;
      auto warpsPerTile = getWarpsPerTile(dotOp, retShapePerCTA, versionMajor,
                                          numWarps, instrShape);
      mmaEnc = ttg::NvidiaMmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor, versionMinor, warpsPerTile,
          CTALayout, instrShape);
    }
    auto newRetType = RankedTensorType::get(
        oldRetType.getShape(), oldRetType.getElementType(), mmaEnc);
    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(oldAcc.getLoc(),
                                                        newRetType, oldAcc);

    if (versionMajor == 3) {
      a = getMMAv3Operand(a, rewriter, 0);
      b = getMMAv3Operand(b, rewriter, 1);
    } else {

      // convert operands
      int minBitwidth =
          std::min(computeOrigBitWidth(a), computeOrigBitWidth(b));
      Type minType = IntegerType::get(ctx, minBitwidth);
      // convert A operand
      auto newAEncoding = ttg::DotOperandEncodingAttr::get(
          oldAType.getContext(), 0, newRetType.getEncoding(),
          minBitwidth > 0 ? minType : oldAType.getElementType());
      auto newAType = RankedTensorType::get(
          oldAType.getShape(), oldAType.getElementType(), newAEncoding);
      a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
      // convert B operand
      auto newBEncoding = ttg::DotOperandEncodingAttr::get(
          oldBType.getContext(), 1, newRetType.getEncoding(),
          minBitwidth > 0 ? minType : oldBType.getElementType());
      auto newBType = RankedTensorType::get(
          oldBType.getShape(), oldBType.getElementType(), newBEncoding);
      b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    }
    // convert dot instruction
    auto newDot = rewriter.create<tt::DotOp>(dotOp.getLoc(), newRetType, a, b,
                                             newAcc, dotOp.getAllowTF32(),
                                             dotOp.getMaxNumImpreciseAcc());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, oldRetType,
                                                      newDot.getResult());
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType =
      operand.getType().cast<RankedTensorType>().cloneWith(std::nullopt,
                                                           promotedType);
  return builder.create<tt::FpToFpOp>(loc, tensorPromotedType, operand);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod, int computeCapability) {
  mod.walk([=](tt::DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    NvidiaMmaEncodingAttr mmaLayout =
        D.getType().getEncoding().dyn_cast<NvidiaMmaEncodingAttr>();
    if (mmaLayout) {
      bool isNativeFP8 = AElType.isFloat8E5M2() || AElType.isFloat8E4M3FNUZ();
      if (!isNativeFP8 || (isNativeFP8 && computeCapability >= 89))
        return;
      promoteType = builder.getF16Type();
    } else {
      // FMA case.
      Type AElType = dotOp.getA().getType().getElementType();
      Type DElType = D.getType().getElementType();
      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUAccelerateMatmulPass
    : public TritonGPUAccelerateMatmulBase<TritonGPUAccelerateMatmulPass> {
public:
  TritonGPUAccelerateMatmulPass() = default;
  TritonGPUAccelerateMatmulPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<::BlockedToMMA>(context, computeCapability);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    // Now that we have picked the mma type, decompose dot that are not natively
    // supported.
    decomposeMixedModeDotOp(m, computeCapability);
  }
};

std::unique_ptr<Pass>
mlir::triton::gpu::createAccelerateMatmulPass(int computeCapability) {
  return std::make_unique<TritonGPUAccelerateMatmulPass>(computeCapability);
}
