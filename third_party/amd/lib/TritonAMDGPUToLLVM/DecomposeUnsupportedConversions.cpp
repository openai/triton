#include "TargetInfo.h"
#include "OptimizeLDSUtility.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <numeric>

using namespace mlir;
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DECOMPOSEUNSUPPORTEDAMDCONVERSIONS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

constexpr int LDSSize = 65536;

static void promoteReduceOpResult(OpBuilder &builder, triton::ReduceOp op,
                                  Value result, Type promotedType) {
  // save original type
  auto originalType = result.getType();
  auto elemType = isa<RankedTensorType>(originalType)
                      ? cast<RankedTensorType>(originalType).getElementType()
                      : originalType;

  // promote result type
  result.setType(promotedType);

  // set insertion point after reduce op
  builder.setInsertionPointAfter(op);

  // truncate result back to original type
  mlir::Operation *truncResult = nullptr;
  if (elemType.isInteger(16) || elemType.isInteger(8)) {
    truncResult = builder.create<mlir::arith::TruncIOp>(result.getLoc(),
                                                        originalType, result);
  } else if (elemType.isF16()) {
    truncResult = builder.create<mlir::arith::TruncFOp>(result.getLoc(),
                                                        originalType, result);
  }

  // replace all uses except for the truncOp above
  if (truncResult != nullptr) {
    result.replaceAllUsesWith(truncResult->getResult(0));
    truncResult->setOperand(0, result);
  }
}

struct DecomposeUnsupportedAMDConversions
    : public mlir::triton::impl::DecomposeUnsupportedAMDConversionsBase<
          DecomposeUnsupportedAMDConversions> {
  explicit DecomposeUnsupportedAMDConversions(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  void runOnOperation() override {
    triton::AMD::TargetInfo targetInfo(this->arch.getValue());
    int sharedMemoryLimit = targetInfo.getSharedMemorySize();

    ModuleOp mod = getOperation();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    triton::gpu::decomposeSplatOpToSharedLayoutConversion(mod);

    triton::gpu::decomposeTensorCoreToDotLayoutConversion<
        triton::gpu::AMDMfmaEncodingAttr>(mod, isMfmaToDotShortcut);

    /* -------------------------------- */
    // Replace `wmma -> dot_op` with `wmma -> blocked -> dot_op`
    /* -------------------------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getSrc().getType();
      auto dstType = cvtOp.getType();
      auto srcWmma =
          dyn_cast<triton::gpu::AMDWmmaEncodingAttr>(srcType.getEncoding());
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (srcWmma && dstDotOp) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcWmma),
                getOrder(srcWmma), numWarps, threadsPerWarp, numCTAs));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
    // Try to reduce LDS usage of cvt(mfma->blocked) op by changing the shape of
    // WarpsPerCta attribute in mfma layout. The implicit LDS usage of
    // cvt(mfma->blocked) op depends on the number of warps per CTA that mfma
    // layout uses along x dimension and block layout uses across y dimension.
    //
    // clang-format off
    //
    // LDS usage of this op is roughly calculated as:
    // LDS_USAGE = getShapePerCTA(mfma_layout)[0] * getShapePerCTA(blocked_layout)[1] * sizeof(data_type)
    // LDS_USAGE = warpsPerCTA(mfma_layout)[0] * warpsPerCta(blocked_layout)[1] * C,
    // where C = 32 * sizePerWarp(blocked_layout)[1] * threadsPerWarp(blocked_layout)[1] * sizeof(data_type)
    //
    // clang-format on
    //
    // When LDS_USAGE exceeds the size of LDS, try to lower LDS usage by
    // decomposing cvt(mfma->blocked) op into 2 conversions: cvt(mfma->mfma_tmp)
    // and cvt(mfma_tmp->blocked), where mfma_tmp has WarpsPerCta attribute that
    // minimizes uses of LDS for these conversions.
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);

      auto srcType = cvtOp.getSrc().getType();
      auto dstType = cvtOp.getType();

      auto srcEnc = srcType.getEncoding();
      auto dstBlocked =
          dstType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();

      // TODO: Reduce LDS usage for WMMA dots
      if (!srcEnc.isa<triton::gpu::AMDMfmaEncodingAttr>() || !dstBlocked) {
        return;
      }

      auto currLDSUsage = mlir::triton::AMD::getCvtOpLDSUsage(cvtOp);
      if (currLDSUsage <= LDSSize) {
        return;
      }

      unsigned numWarps = triton::gpu::getNumWarpsPerCTA(srcEnc);

      // Find all possible shapes of WarpsPerCTA by finding all possible
      // factorizations of numWarps. Pick shape for which both conversions in
      // decomposition use LDS less than LDSSize and for which sum of LDS usage
      // is minimal. If no such shape exists, do not decompose.
      unsigned minLDSUsage = 2 * LDSSize;
      int minIdx = -1;
      int rank = dstBlocked.getWarpsPerCTA().size();
      auto factorizedNumWarps =
          mlir::triton::AMD::factorizePowerOf2(numWarps, rank);

      SmallVector<Attribute> tmpLayouts;
      for (int i = 0; i < factorizedNumWarps.size(); i++) {
        auto warpsPerCTA = factorizedNumWarps[i];
        tmpLayouts.push_back(
            mlir::triton::AMD::createTmpLayout(srcEnc, warpsPerCTA));
      }

      for (int i = 0; i < tmpLayouts.size(); i++) {
        auto resources = mlir::triton::AMD::estimateResourcesForReplacement(
            builder, cvtOp, tmpLayouts[i]);
        if (resources.LDS <= LDSSize && resources.LDS < minLDSUsage) {
          minLDSUsage = resources.LDS;
          minIdx = i;
        }
      }

      if (minIdx == -1 || minLDSUsage > LDSSize) {
        return;
      }

      assert(minIdx >= 0 && minIdx < tmpLayouts.size());
      auto replacementCvts = mlir::triton::AMD::createNewConvertOps(
          builder, cvtOp, tmpLayouts[minIdx]);

      cvtOp.replaceAllUsesWith(replacementCvts.second.getResult());
      cvtOp.erase();
    });

    triton::gpu::decomposeBlockedToDotLayoutConversion(mod);

    // promote reduce ops
    mod.walk([&](triton::ReduceOp op) -> void {
      OpBuilder builder(op);

      // promote operands
      SmallVector<Value> newOperands;
      for (OpOperand &operand : op->getOpOperands()) {
        auto val = operand.get();
        auto oldType = cast<RankedTensorType>(val.getType());
        auto elemType = oldType.getElementType();
        if (elemType.isInteger(16) || elemType.isInteger(8)) {
          auto newType =
              oldType.cloneWith(std::nullopt, builder.getIntegerType(32));
          auto promotedVal =
              builder.create<mlir::arith::ExtSIOp>(op->getLoc(), newType, val);
          newOperands.push_back(promotedVal);
        } else if (elemType.isF16()) {
          auto newType = oldType.cloneWith(std::nullopt, builder.getF32Type());
          auto promotedVal =
              builder.create<mlir::arith::ExtFOp>(op->getLoc(), newType, val);
          newOperands.push_back(promotedVal);
        } else {
          newOperands.push_back(val);
        }
      }
      op->setOperands(newOperands);

      // promote results
      for (Value result : op.getResults()) {
        auto type = result.getType();
        if (type.isInteger(16) || type.isInteger(8)) {
          promoteReduceOpResult(builder, op, result,
                                builder.getIntegerType(32));
        } else if (type.isF16()) {
          promoteReduceOpResult(builder, op, result, builder.getF32Type());
        } else if (isa<RankedTensorType>(type)) {
          auto oldType = cast<RankedTensorType>(type);
          auto elemType = oldType.getElementType();
          if (elemType.isInteger(16) || elemType.isInteger(8)) {
            promoteReduceOpResult(
                builder, op, result,
                oldType.cloneWith(std::nullopt, builder.getIntegerType(32)));
          } else if (elemType.isF16()) {
            promoteReduceOpResult(
                builder, op, result,
                oldType.cloneWith(std::nullopt, builder.getF32Type()));
          }
        }
      }

      // promote combine op
      for (Block &oldBlock : op.getCombineOp().getBlocks()) {
        // update block args
        for (auto arg : oldBlock.getArguments()) {
          auto type = arg.getType();
          if (type.isInteger(16) || type.isInteger(8)) {
            arg.setType(builder.getIntegerType(32));
          } else if (type.isF16()) {
            arg.setType(builder.getF32Type());
          }
        }

        for (Operation &oldOp : oldBlock.getOperations()) {
          // update operands
          for (OpOperand &operand : oldOp.getOpOperands()) {
            auto val = operand.get();
            auto type = val.getType();
            if (type.isInteger(16) || type.isInteger(8)) {
              val.setType(builder.getIntegerType(32));
            } else if (type.isF16()) {
              val.setType(builder.getF32Type());
            }
          }

          // update results
          for (Value result : oldOp.getResults()) {
            auto type = result.getType();
            if (type.isInteger(16) || type.isInteger(8)) {
              result.setType(builder.getIntegerType(32));
            } else if (type.isF16()) {
              result.setType(builder.getF32Type());
            }
          }
        }
      }
    });
  }
};

} // namespace

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedConversionsPass(StringRef targetArch) {
  return std::make_unique<DecomposeUnsupportedAMDConversions>(targetArch);
}

} // namespace mlir::triton::AMD
