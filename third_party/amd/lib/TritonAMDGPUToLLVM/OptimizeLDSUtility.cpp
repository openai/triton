#include "OptimizeLDSUtility.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <numeric>

using namespace mlir;

namespace mlir {
namespace triton {
namespace AMD {

constexpr int kPtrBitWidth = 64;

int getCvtOpLDSUsage(RankedTensorType srcTy, RankedTensorType dstTy) {
  unsigned inVec = 0;
  unsigned outVec = 0;
  auto smemShape =
      triton::getScratchConfigForCvtLayout(srcTy, dstTy, inVec, outVec);
  unsigned elems =
      std::accumulate(smemShape.begin(), smemShape.end(), 1, std::multiplies{});
  auto bytes =
      srcTy.getElementType().isa<triton::PointerType>()
          ? elems * kPtrBitWidth / 8
          : elems * std::max<int>(8, srcTy.getElementTypeBitWidth()) / 8;

  return bytes;
}

int getCvtOpLDSUsage(triton::gpu::ConvertLayoutOp op) {
  return getCvtOpLDSUsage(op.getSrc().getType(), op.getType());
}

bool isPowerOfTwo(unsigned x) { return x && (x & (x - 1)) == 0; }

static void stepFactorizationPow2(std::vector<SmallVector<unsigned>> &factors,
                                  SmallVector<unsigned> &curFactor,
                                  int restTwos, int dim) {
  if (dim == curFactor.size()) {
    if (restTwos == 0)
      factors.push_back(curFactor);
    return;
  }
  curFactor[dim] = 1;
  for (int i = 0; i <= restTwos; ++i) {
    stepFactorizationPow2(factors, curFactor, restTwos - i, dim + 1);
    curFactor[dim] *= 2;
  }
}

std::vector<SmallVector<unsigned>> factorizePowerOf2(int n, int rank) {
  assert(isPowerOfTwo(n));
  int x = log2(n);
  std::vector<SmallVector<unsigned>> factors;
  SmallVector<unsigned> curFactor(rank, 1);
  stepFactorizationPow2(factors, curFactor, x, 0);
  return factors;
}

Attribute createTmpLayout(Attribute layout, ArrayRef<unsigned> warpsPerCTA) {
  auto ctx = layout.getContext();
  if (auto src = layout.dyn_cast<triton::gpu::AMDMfmaEncodingAttr>())
    return triton::gpu::AMDMfmaEncodingAttr::get(
        ctx, src.getVersionMajor(), src.getVersionMinor(), warpsPerCTA,
        src.getMDim(), src.getNDim(), src.getIsTransposed(),
        src.getCTALayout());
  if (auto src = layout.dyn_cast<triton::gpu::AMDWmmaEncodingAttr>())
    return triton::gpu::AMDWmmaEncodingAttr::get(ctx, warpsPerCTA,
                                                 src.getCTALayout());
  if (auto src = layout.dyn_cast<triton::gpu::BlockedEncodingAttr>())
    return triton::gpu::BlockedEncodingAttr::get(
        ctx, src.getSizePerThread(), src.getThreadsPerWarp(), warpsPerCTA,
        src.getOrder(), src.getCTALayout());
  if (auto src = layout.dyn_cast<triton::gpu::DotOperandEncodingAttr>()) {
    return triton::gpu::DotOperandEncodingAttr::get(
        ctx, src.getOpIdx(), createTmpLayout(src.getParent(), warpsPerCTA),
        src.getKWidth());
  }
  if (auto src = layout.dyn_cast<triton::gpu::SliceEncodingAttr>())
    return triton::gpu::SliceEncodingAttr::get(
        ctx, src.getDim(), createTmpLayout(src.getParent(), warpsPerCTA));
  assert("Encountered unsupported layout");
  return Attribute();
}

std::pair<triton::gpu::ConvertLayoutOp, triton::gpu::ConvertLayoutOp>
createNewConvertOps(OpBuilder &builder, triton::gpu::ConvertLayoutOp &cvtOp,
                    Attribute tmpLayout) {
  auto srcType = cvtOp.getSrc().getType();
  auto dstType = cvtOp.getType();

  auto newDstType = RankedTensorType::get(
      dstType.getShape(), dstType.getElementType(), dstType.getEncoding());
  RankedTensorType newSrcType = RankedTensorType::get(
      srcType.getShape(), srcType.getElementType(), tmpLayout);

  auto tmpCvt = builder.create<triton::gpu::ConvertLayoutOp>(
      cvtOp.getLoc(), newSrcType, cvtOp.getSrc());
  auto newEpilogueCvt = builder.create<triton::gpu::ConvertLayoutOp>(
      cvtOp.getLoc(), newDstType, tmpCvt);

  return std::make_pair(tmpCvt, newEpilogueCvt);
}

Resources
estimateResourcesForReplacement(OpBuilder builder,
                                mlir::triton::gpu::ConvertLayoutOp cvtOp,
                                Attribute tmpLayout) {
  Resources res;
  RankedTensorType srcTy = cvtOp.getSrc().getType();
  RankedTensorType dstTy = cvtOp.getType();
  RankedTensorType intermediateTy = RankedTensorType::get(
      srcTy.getShape(), srcTy.getElementType(), tmpLayout);

  int tmpCvtLDS = mlir::triton::AMD::getCvtOpLDSUsage(srcTy, intermediateTy);
  int newCvtLDS = mlir::triton::AMD::getCvtOpLDSUsage(intermediateTy, dstTy);
  res.LDS = std::max(tmpCvtLDS, newCvtLDS);
  return res;
}

} // namespace AMD
} // namespace triton
} // namespace mlir
