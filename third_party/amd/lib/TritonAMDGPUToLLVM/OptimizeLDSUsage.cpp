/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Analysis/Liveness.h"
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
#define GEN_PASS_DEF_OPTIMIZEAMDLDSUSAGE
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

constexpr int kPtrBitWidth = 64;

class OptimizeAMDLDSUsage
    : public mlir::triton::impl::OptimizeAMDLDSUsageBase<OptimizeAMDLDSUsage> {

  static int getCvtOpLDSUsage(triton::gpu::ConvertLayoutOp &cvtOp) {
    unsigned inVec = 0;
    unsigned outVec = 0;
    auto smemShape = triton::getScratchConfigForCvtLayout(cvtOp, inVec, outVec);
    unsigned elems = std::accumulate(smemShape.begin(), smemShape.end(), 1,
                                     std::multiplies{});
    auto srcType = cvtOp.getSrc().getType();
    auto bytes =
        srcType.getElementType().isa<triton::PointerType>()
            ? elems * kPtrBitWidth / 8
            : elems * std::max<int>(8, srcType.getElementTypeBitWidth()) / 8;

    return bytes;
  }

  static bool isPowerOfTwo(unsigned x) { return x && (x & (x - 1)) == 0; }

  static std::vector<SmallVector<unsigned>> factorizePowerOf2(int n) {
    assert(isPowerOfTwo(n));
    int x = log2(n);
    std::vector<SmallVector<unsigned>> pairs;

    for (int i = 0; i <= x / 2; ++i) {
      int j = x - i;
      SmallVector<unsigned> sample(2);
      sample[0] = pow(2, i);
      sample[1] = pow(2, j);
      pairs.push_back(sample);
      std::swap(sample[0], sample[1]);
      pairs.push_back(sample);
    }

    return pairs;
  }

  /**
   * @brief Copy given layout with different warpsPerCTA parameter
   * @param layout original layout
   * @param warpsPerCTA new warpsPerCTA
   * @return create layout
   */
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

  /**
   * Creates two chained convert layout operations
   *
   * %1 = cvtOp %0 (srcLayout -> dstLayout)
   * ->
   * %1 = cvtOp %0 (srcLayout -> dstLayout)
   * %2 = cvtOp %0 (srcLayout -> tmpLayout)
   * %3 = cvtOp %1 (tmpLayout -> dstLayout)
   *
   * @param builder
   * @param cvtOp original operation
   * @param tmpLayout
   * @return pair of created operations
   */
  static std::pair<triton::gpu::ConvertLayoutOp, triton::gpu::ConvertLayoutOp>
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

  // Try to reduce LDS usage of cvt op.
  //
  // Consider mfma->blocked conversion as an example.
  // LDS reduction is possible by changing the shape of WarpsPerCta attribute in
  // mfma layout. The implicit LDS usage of cvt(mfma->blocked) op depends on the
  // number of warps per CTA that mfma layout uses along x dimension and block
  // layout uses across y dimension.
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
  void tryMinimizeLDS(triton::gpu::ConvertLayoutOp cvtOp) {
    OpBuilder builder(cvtOp);

    auto srcType = cvtOp.getSrc().getType();
    auto dstType = cvtOp.getType();

    auto srcEnc = srcType.getEncoding();
    auto dstEnc = dstType.getEncoding();

    auto currLDSUsage = getCvtOpLDSUsage(cvtOp);
    if (currLDSUsage <= LDSSize) {
      return;
    }

    auto ctx = srcEnc.getContext();
    auto rank = srcType.getShape().size();
    unsigned numWarps = triton::gpu::getNumWarpsPerCTA(srcEnc);
    auto warpSize = triton::gpu::getWarpSize(srcEnc);

    triton::gpu::ConvertLayoutOp tmpCvt;
    triton::gpu::ConvertLayoutOp newEpilogueCvt;

    // Find all possible shapes of WarpsPerCTA by finding all possible
    // factorizations of numWarps. Pick shape for which both conversions in
    // decomposition use LDS less than LDSSize and for which sum of LDS usage
    // is minimal. If no such shape exists, do not decompose.
    auto factorizedNumWarps = factorizePowerOf2(numWarps);
    // Create a list of temporary layouts
    SmallVector<Attribute> tmpLayouts;
    for (int i = 0; i < factorizedNumWarps.size(); i++) {
      auto warpsPerCTA = factorizedNumWarps[i];
      tmpLayouts.push_back(createTmpLayout(srcEnc, warpsPerCTA));
      tmpLayouts.push_back(createTmpLayout(dstEnc, warpsPerCTA));
      SmallVector<unsigned> elemsPerThread(rank, 1);
      SmallVector<unsigned> threadsPerWarp(rank, 1);
      threadsPerWarp[rank - 1] = warpSize / 8;
      threadsPerWarp[rank - 2] = warpSize / threadsPerWarp[rank - 1];
      auto order = triton::gpu::getOrder(srcEnc);
      auto layoutCTA = triton::gpu::getCTALayout(srcEnc);
      auto fallbackLayout = triton::gpu::BlockedEncodingAttr::get(
          ctx, elemsPerThread, threadsPerWarp, warpsPerCTA, order, layoutCTA);
      tmpLayouts.push_back(fallbackLayout);
    }

    unsigned minLDSUsage = 2 * LDSSize;
    int minIdx = -1;
    for (int i = 0; i < tmpLayouts.size(); i++) {
      auto tmpLayout = tmpLayouts[i];
      std::tie(tmpCvt, newEpilogueCvt) =
          createNewConvertOps(builder, cvtOp, tmpLayout);

      int tmpCvtLDS = getCvtOpLDSUsage(tmpCvt);
      int newCvtLDS = getCvtOpLDSUsage(newEpilogueCvt);
      if (tmpCvtLDS <= LDSSize && newCvtLDS <= LDSSize) {
        int LDSUsage = std::max(tmpCvtLDS, newCvtLDS);
        if (LDSUsage < minLDSUsage) {
          minLDSUsage = LDSUsage;
          minIdx = i;
        }
      }
      newEpilogueCvt.erase();
      tmpCvt.erase();
    }

    if (minIdx == -1) {
      return;
    }

    assert(minIdx >= 0 && minIdx < tmpLayouts.size());
    auto tmpLayout = tmpLayouts[minIdx];
    std::tie(tmpCvt, newEpilogueCvt) =
        createNewConvertOps(builder, cvtOp, tmpLayout);

    cvtOp.replaceAllUsesWith(newEpilogueCvt.getResult());
    cvtOp.erase();
  }

  /**
   * @brief Get live information for LDS buffers in function
   * @return mapping from operation to list of live LDS buffers
   */
  std::map<mlir::Operation *, SmallVector<Allocation::BufferId>>
  analyzeBufferLiveness(FunctionOpInterface func,
                        const Allocation *allocations) {
    std::map<mlir::Operation *, SmallVector<Allocation::BufferId>> liveBuffers;

    mlir::Liveness liveness(func);
    auto analyzeOperation = [&](mlir::Operation *op) -> void {
      auto scratchBuffer = allocations->getBufferId(op);
      if (scratchBuffer != Allocation::InvalidBufferId)
        liveBuffers[op].push_back(scratchBuffer);
      for (auto result : op->getOpResults()) {
        auto bufferId = allocations->getBufferId(result);
        if (bufferId == Allocation::InvalidBufferId)
          continue;
        auto liveOperations = liveness.resolveLiveness(result);
        for (auto depOp : liveOperations)
          liveBuffers[depOp].push_back(bufferId);
      }
    };
    func.walk(analyzeOperation);
    return liveBuffers;
  }

  SmallVector<triton::gpu::ConvertLayoutOp>
  findLDSBottleneck(ModuleAllocation &allocAnalysis, FunctionOpInterface func) {
    SmallVector<triton::gpu::ConvertLayoutOp> candidates;
    auto funcAnalysis = allocAnalysis.getFuncData(func);
    auto liveBuffers = analyzeBufferLiveness(func, funcAnalysis);

    func.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      auto opBuffer = funcAnalysis->getBufferId(cvtOp.getOperation());
      assert(opBuffer != Allocation::InvalidBufferId);
      for (auto bufId : liveBuffers[cvtOp]) {
        auto offset = funcAnalysis->getOffset(bufId);
        auto size = funcAnalysis->getAllocatedSize(bufId);
        if (offset + size > LDSSize) {
          candidates.push_back(cvtOp);
          break;
        }
      }
    });
    return candidates;
  }

public:
  OptimizeAMDLDSUsage(int32_t LDSSize)
      : OptimizeAMDLDSUsageBase<OptimizeAMDLDSUsage>() {
    this->LDSSize = LDSSize;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    ModuleAllocation allocAnalysis(mod);
    if (allocAnalysis.getSharedMemorySize() > LDSSize) {
      auto rootFunctions = allocAnalysis.getRoots();
      for (auto rootFunc : rootFunctions) {
        // Find operations with peak LDS consumption
        auto candidates = findLDSBottleneck(allocAnalysis, rootFunc);
        // Try to transform candidates to minimize LDS usage
        for (auto candidate : candidates)
          tryMinimizeLDS(candidate);
      }
    }
  }
};

} // namespace

namespace mlir {

namespace triton {

namespace AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createOptimizeLDSUsagePass(int32_t LDSSize) {
  return std::make_unique<OptimizeAMDLDSUsage>(LDSSize);
}

} // namespace AMD

} // namespace triton

} // namespace mlir
