#include "TritonAMDGPUTransforms/MfmaGroup.h"
#include "TritonAMDGPUTransforms/Passes.h"
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
using ttg::AMDMfmaEncodingAttr;
using ttg::AMDWmmaEncodingAttr;
using ttg::BlockedEncodingAttr;
using ttg::ConvertLayoutOp;
using ttg::DotOperandEncodingAttr;
using ttg::SliceEncodingAttr;

enum class MatrixCoreVersion {
  CDNA_MFMA1,
  CDNA_MFMA2,
  CDNA_MFMA3,
  RDNA_WMMA,
  UNKNOWN
};

MatrixCoreVersion getMatrixCoreVersion(StringRef archGen) {
  if (archGen.contains("gfx11"))
    return MatrixCoreVersion::RDNA_WMMA;
  if (archGen.contains("gfx908"))
    return MatrixCoreVersion::CDNA_MFMA1;
  if (archGen.contains("gfx90a"))
    return MatrixCoreVersion::CDNA_MFMA2;
  if (archGen.contains("gfx940") || archGen.contains("gfx941") ||
      archGen.contains("gfx942"))
    return MatrixCoreVersion::CDNA_MFMA3;
  return MatrixCoreVersion::UNKNOWN;
}

int getMfmaVersion(MatrixCoreVersion matrixCoreVer) {
  if (MatrixCoreVersion::CDNA_MFMA1 == matrixCoreVer)
    return 1;
  if (MatrixCoreVersion::CDNA_MFMA2 == matrixCoreVer)
    return 2;
  if (MatrixCoreVersion::CDNA_MFMA3 == matrixCoreVer)
    return 3;
  return 0;
}

SmallVector<unsigned, 2> warpsPerTile(tt::DotOp dotOp,
                                      const ArrayRef<int64_t> shape,
                                      int numWarps,
                                      SmallVector<int64_t, 2> shapePerWarp) {
  // TODO: needs to be updated with appropriate shapePerWarp etc.
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  mlir::ForwardSliceOptions fwdOpt;
  fwdOpt.filter = filter;
  mlir::BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = filter;
  auto slices = mlir::getSlice(dotOp, bwdOpt, fwdOpt);
  for (Operation *op : slices)
    if (isa<tt::DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<int64_t, 2> tensorShape = {shape[0], shape[1]};
  SmallVector<unsigned, 2> ret = {1, 1};

  do {
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (tensorShape[0] / (shapePerWarp[0] * 2) / ret[0] >=
        tensorShape[1] / shapePerWarp[1] / ret[1]) {
      if (ret[0] < tensorShape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);

  if (ret[1] * shapePerWarp[1] > tensorShape[1]) {
    return {ret[1], ret[0]};
  }

  return ret;
}

SmallVector<unsigned, 2>
warpsPerTileMFMA(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps) {
  return warpsPerTile(dotOp, shape, numWarps, {32, 32});
}

SmallVector<unsigned, 2>
warpsPerTileWMMA(tt::DotOp dotOp, const ArrayRef<int64_t> shape, int numWarps) {
  return warpsPerTile(dotOp, shape, numWarps,
                      {AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr()[0],
                       AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr()[1]});
}

/**
 * @brief Convert layout and cast element type of a given tensor
 *
 * If old element type is different from new element type, this function
 * creates two new operations:
 * 1. %converted_value = layout_convert %value, newEncoding
 * 2. %casted_value = cast(fext, ftrunc, etc.) %value, newElemType
 *
 * If old element type is same as new element type, this function creates only
 * one operation: %converted_value = layout_convert %value, newEncoding
 *
 * @param rewriter
 * @param value original tensor value, which we need to convert and cast
 * @param newEncoding new encoding for the tenosr
 * @param newElemType new element type for the tensor
 * @return converted and optionaly casted tensor value
 */
Value convertAndCastTensor(mlir::PatternRewriter &rewriter, Value value,
                           ::mlir::Attribute newEncoding, Type newElemType) {
  assert(newElemType.isIntOrFloat());

  auto loc = value.getLoc();
  auto oldType = value.getType().cast<RankedTensorType>();
  auto oldElemType = oldType.getElementType();

  assert(oldElemType.isIntOrFloat());
  assert(oldElemType.isIntOrIndex() == newElemType.isIntOrIndex());

  auto convertedType =
      RankedTensorType::get(oldType.getShape(), oldElemType, newEncoding);

  Value convertedTensor =
      rewriter.create<ttg::ConvertLayoutOp>(loc, convertedType, value);

  if (newElemType == oldElemType)
    return convertedTensor;

  Type castedType = convertedType.cloneWith(std::nullopt, newElemType);

  Value castedTensor;

  if (newElemType.isIntOrIndex()) {
    unsigned oldWidth = oldElemType.getIntOrFloatBitWidth();
    unsigned newWidth = newElemType.getIntOrFloatBitWidth();
    if (oldWidth == newWidth)
      castedTensor = rewriter.create<mlir::arith::BitcastOp>(loc, convertedType,
                                                             convertedTensor);
    else if (oldWidth > newWidth)
      castedTensor = rewriter.create<mlir::arith::TruncIOp>(loc, castedType,
                                                            convertedTensor);
    else if (oldElemType.isSignedInteger())
      castedTensor = rewriter.create<mlir::arith::ExtSIOp>(loc, castedType,
                                                           convertedTensor);
    else
      castedTensor = rewriter.create<mlir::arith::ExtUIOp>(loc, castedType,
                                                           convertedTensor);
  } else {
    if (oldElemType.isF16() && newElemType.isF32())
      castedTensor = rewriter.create<mlir::arith::ExtFOp>(loc, castedType,
                                                          convertedTensor);
    else if (oldElemType.isF32() && newElemType.isF16())
      castedTensor = rewriter.create<mlir::arith::TruncFOp>(loc, castedType,
                                                            convertedTensor);
    else
      castedTensor =
          rewriter.create<tt::FpToFpOp>(loc, castedType, convertedTensor);
  }
  return castedTensor;
}

class BlockedToMFMA : public mlir::RewritePattern {
  int mfmaVersion;
  int enforcedNonKDim;

public:
  BlockedToMFMA(mlir::MLIRContext *context, int mfmaVersion, int nonKDim)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context),
        mfmaVersion(mfmaVersion), enforcedNonKDim(nonKDim) {}

  bool isChainDot(tt::DotOp &dotOp) const {
    auto filter = [&dotOp](Operation *op) {
      return op->getParentRegion() == dotOp->getParentRegion();
    };
    mlir::ForwardSliceOptions fwdOpt;
    fwdOpt.filter = filter;
    mlir::BackwardSliceOptions bwdOpt;
    bwdOpt.omitBlockArguments = true;
    bwdOpt.filter = filter;
    auto slices = mlir::getSlice(dotOp, bwdOpt, fwdOpt);
    for (Operation *op : slices) {
      if (isa<tt::DotOp>(op) && (op != dotOp))
        return true;
    }
    return false;
  }

  /// @brief Choose MFMA instruction parameters
  /// @param dot target dot operation
  /// @return pair {mDim, nDim, kDim} sizes of one MFMA instruction arguments
  std::tuple<unsigned, unsigned, unsigned>
  chooseMfmaDimensions(tt::DotOp dot) const {
    // number of matrix elements along k dim per one MFMA intruction
    unsigned kDim = 0;
    auto opType = dot.getA().getType().cast<RankedTensorType>();
    auto dataTypeA = opType.getElementType();
    auto dataTypeB =
        dot.getB().getType().cast<RankedTensorType>().getElementType();

    auto resType = dot.getD().getType().cast<RankedTensorType>();
    auto resShape = resType.getShape();

    unsigned mDim = 0;
    unsigned nDim = 0;
    if (enforcedNonKDim != 0) {
      mDim = enforcedNonKDim;
      nDim = enforcedNonKDim;
    } else {
      int minSize = std::min(resShape[0], resShape[1]);
      if (minSize >= 32) {
        mDim = 32;
        nDim = 32;
      }
      if (minSize >= 16 && minSize < 32) {
        mDim = 16;
        nDim = 16;
      }
      if (minSize < 16) {
        if (resShape[0] < 16 && resShape[1] >= 64) {
          mDim = 4;
          nDim = 64;
        } else if (resShape[0] >= 64 && resShape[1] < 16) {
          mDim = 64;
          nDim = 4;
        } else {
          assert(opType.getShape()[1] >= 64 &&
                 "k should be at least 64 to use this layout");
          mDim = 4;
          nDim = 4;
        }
      }
    }
    assert(mDim != 0 && nDim != 0);

    auto maybeMfmaInsn =
        MfmaInsn::selectMfma(mDim, nDim, dataTypeA, dataTypeB, mfmaVersion);
    if (failed(maybeMfmaInsn))
      llvm::report_fatal_error("No match found in MFMA database\n");
    else
      kDim = (*maybeMfmaInsn).getKDim();
    assert(kDim != 0);

    assert(resShape[0] % mDim == 0 && resShape[1] % nDim == 0);
    assert(opType.getShape()[1] % kDim == 0);
    return {mDim, nDim, kDim};
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<tt::DotOp>(op);

    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        !oldRetType.getEncoding().isa<ttg::BlockedEncodingAttr>())
      return failure();

    if (!supportMFMA(dotOp))
      return failure();

    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());

    // get MFMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();
    auto ctx = oldAType.getContext();

    ttg::AMDMfmaEncodingAttr mfmaEnc;

    auto [mDim, nDim, kDim] = chooseMfmaDimensions(dotOp);

    auto warpsPerTile = warpsPerTileMFMA(dotOp, retShape, numWarps);

    bool isTransposed = isChainDot(dotOp);
    mfmaEnc = ttg::AMDMfmaEncodingAttr::get(
        oldRetType.getContext(),
        /*versionMajor*/ mfmaVersion, /*versionMinor*/ 0, warpsPerTile,
        /*instrShape*/ mDim, nDim, isTransposed, CTALayout);

    Type mfmaAccType;
    if (oldRetType.getElementType().isIntOrIndex())
      mfmaAccType = rewriter.getIntegerType(32);
    else
      mfmaAccType = rewriter.getF32Type();

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = convertAndCastTensor(rewriter, oldAcc, mfmaEnc, mfmaAccType);

    // kWidth is a number of consecutive elements per one instruction per one
    // thread
    auto kWidth = -1;
    // in mfma 32x32 case argument matrix groups elements in 2 groups
    // in mfma 16x16 case argument matrix groups elements in 4 groups
    // in mfma 4x4 case argument matrix groups in 16 groups
    if (mDim == 32 && nDim == 32)
      kWidth = kDim / 2;
    if (mDim == 16 && nDim == 16)
      kWidth = kDim / 4;
    if (mDim == 4 && nDim == 4)
      kWidth = kDim / 16;
    if (mDim == 4 && nDim == 64 || mDim == 64 && nDim == 4)
      kWidth = kDim;
    assert(kWidth != -1);
    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 0, mfmaEnc, kWidth));
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 1, mfmaEnc, kWidth));
    a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<tt::DotOp>(dotOp.getLoc(), newAcc.getType(),
                                             a, b, newAcc, dotOp.getInputPrecision(),
                                             dotOp.getMaxNumImpreciseAcc());

    Value dotOutput =
        convertAndCastTensor(rewriter, newDot, oldRetType.getEncoding(),
                             oldRetType.getElementType());

    rewriter.replaceOp(op, dotOutput);

    return success();
  }
};
static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType =
      operand.getType().cast<RankedTensorType>().cloneWith(std::nullopt,
                                                           promotedType);
  return builder.create<triton::FpToFpOp>(loc, tensorPromotedType, operand);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod) {
  mod.walk([](triton::DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType;
    if (D.getType().getEncoding().isa<AMDMfmaEncodingAttr>()) {
      Type BElType = dotOp.getB().getType().getElementType();

      auto maxBitWidth = std::max(AElType.getIntOrFloatBitWidth(),
                                  BElType.getIntOrFloatBitWidth());

      // TODO check mfma tensor core version compatibility
      if (maxBitWidth == 8)
        return;

      if (AElType == BElType)
        return;

      if (maxBitWidth < 16)
        promoteType = builder.getF16Type();
      else if (maxBitWidth <= 32)
        promoteType = builder.getF32Type();
    } else if (D.getType().getEncoding().isa<AMDWmmaEncodingAttr>()) {
      Type BElType = dotOp.getB().getType().getElementType();

      if (AElType == BElType)
        return;

      // Other cases must be filtered earlier
      promoteType =
          AElType.getIntOrFloatBitWidth() > BElType.getIntOrFloatBitWidth()
              ? AElType
              : BElType;
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

class BlockedToWMMA : public mlir::RewritePattern {
public:
  BlockedToWMMA(mlir::MLIRContext *context)
      : mlir::RewritePattern(tt::DotOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<tt::DotOp>(op);

    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        !oldRetType.getEncoding().isa<ttg::BlockedEncodingAttr>())
      return failure();

    // TODO: Support different operand types
    if (!supportWMMA(dotOp))
      return failure();

    // get WMMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();
    auto ctx = oldAType.getContext();

    AMDWmmaEncodingAttr wmmaEnc;

    auto mnkDim = AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr();
    auto warpsPerTile = warpsPerTileWMMA(dotOp, retShape, numWarps);
    // Not supported yet
    // if (retShape[0] < warpsPerTile[0] * mnkDim[0] || retShape[1] <
    // warpsPerTile[1] * mnkDim[1])
    //  return failure();
    auto CTALayout = ttg::getCTALayout(oldRetType.getEncoding());
    wmmaEnc = AMDWmmaEncodingAttr::get(oldRetType.getContext(), warpsPerTile,
                                       CTALayout);

    Type wmmaAccType;
    auto oldRetElemType = oldRetType.getElementType();
    auto aElemType = oldAType.getElementType();
    if (oldRetElemType.isIntOrIndex())
      wmmaAccType = rewriter.getIntegerType(32);
    else if (oldRetElemType.isa<mlir::Float16Type, mlir::BFloat16Type>() &&
             aElemType == oldRetElemType)
      wmmaAccType = oldRetElemType;
    else
      wmmaAccType = rewriter.getF32Type();

    auto newRetType = RankedTensorType::get(retShape, oldRetElemType, wmmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = convertAndCastTensor(rewriter, oldAcc, wmmaEnc, wmmaAccType);

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), aElemType,
        ttg::DotOperandEncodingAttr::get(ctx, 0, wmmaEnc, mnkDim[2]));
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(),
        ttg::DotOperandEncodingAttr::get(ctx, 1, wmmaEnc, mnkDim[2]));
    a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<tt::DotOp>(dotOp.getLoc(), newRetType, a, b,
                                             newAcc, dotOp.getInputPrecision(),
                                             dotOp.getMaxNumImpreciseAcc());

    Value dotOutput = convertAndCastTensor(
        rewriter, newDot, oldRetType.getEncoding(), oldRetElemType);
    rewriter.replaceOp(op, dotOutput);
    return success();
  }
};
} // namespace

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPUAccelerateMatmulPass
    : public TritonAMDGPUAccelerateMatmulBase<
          TritonAMDGPUAccelerateMatmulPass> {
public:
  TritonAMDGPUAccelerateMatmulPass() = default;
  TritonAMDGPUAccelerateMatmulPass(StringRef archGen,
                                   int matrixInstructionSize) {
    this->archGenerationName = archGen.data();
    this->matrixInstructionSize = matrixInstructionSize;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    auto matrixCoreVer = getMatrixCoreVersion(archGenerationName);
    if (MatrixCoreVersion::CDNA_MFMA1 == matrixCoreVer ||
        MatrixCoreVersion::CDNA_MFMA2 == matrixCoreVer ||
        MatrixCoreVersion::CDNA_MFMA3 == matrixCoreVer) {
      patterns.add<::BlockedToMFMA>(context, getMfmaVersion(matrixCoreVer),
                                    matrixInstructionSize);
    } else if (matrixCoreVer == MatrixCoreVersion::RDNA_WMMA) {
      patterns.add<::BlockedToWMMA>(context);
    }
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    decomposeMixedModeDotOp(m);
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUAccelerateMatmulPass(std::string archGen,
                                             int matrixInstructionSize) {
  return std::make_unique<TritonAMDGPUAccelerateMatmulPass>(
      archGen, matrixInstructionSize);
}
