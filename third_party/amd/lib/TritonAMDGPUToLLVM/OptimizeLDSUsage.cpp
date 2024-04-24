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
#include "OptimizeLDSUtility.h"
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

class OptimizeAMDLDSUsage
    : public mlir::triton::impl::OptimizeAMDLDSUsageBase<OptimizeAMDLDSUsage> {

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
  void tryFitCvtIntoLDS(triton::gpu::ConvertLayoutOp cvtOp, int targetLDSSize) {
    OpBuilder builder(cvtOp);

    auto srcType = cvtOp.getSrc().getType();
    auto dstType = cvtOp.getType();

    auto srcEnc = srcType.getEncoding();
    auto dstEnc = dstType.getEncoding();

    auto ctx = srcEnc.getContext();
    auto rank = srcType.getShape().size();
    unsigned numWarps = triton::gpu::getNumWarpsPerCTA(srcEnc);
    auto warpSize = triton::gpu::getWarpSize(srcEnc);

    // Find all possible shapes of WarpsPerCTA by finding all possible
    // factorizations of numWarps. Pick shape for which both conversions in
    // decomposition use LDS less than LDSSize and for which sum of LDS usage
    // is minimal. If no such shape exists, do not decompose.
    auto factorizedNumWarps =
        mlir::triton::AMD::factorizePowerOf2(numWarps, rank);
    // Create a list of temporary layouts
    SmallVector<Attribute> tmpLayouts;
    for (int i = 0; i < factorizedNumWarps.size(); i++) {
      auto warpsPerCTA = factorizedNumWarps[i];
      tmpLayouts.push_back(
          mlir::triton::AMD::createTmpLayout(srcEnc, warpsPerCTA));
      tmpLayouts.push_back(
          mlir::triton::AMD::createTmpLayout(dstEnc, warpsPerCTA));
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
      auto resources = mlir::triton::AMD::estimateResourcesForReplacement(
          builder, cvtOp, tmpLayouts[i]);
      // TODO analyze performance along with LDS consumption
      if (resources.LDS < minLDSUsage) {
        minLDSUsage = resources.LDS;
        minIdx = i;
      }
    }

    if (minIdx == -1 || minLDSUsage > targetLDSSize) {
      return;
    }

    triton::gpu::ConvertLayoutOp tmpCvt;
    triton::gpu::ConvertLayoutOp newEpilogueCvt;

    assert(minIdx >= 0 && minIdx < tmpLayouts.size());
    auto tmpLayout = tmpLayouts[minIdx];
    std::tie(tmpCvt, newEpilogueCvt) =
        mlir::triton::AMD::createNewConvertOps(builder, cvtOp, tmpLayout);

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

  struct LDSBottleneckOperation {
    triton::gpu::ConvertLayoutOp op;
    int64_t LDSSizeTarget;
  };

  /**
   * Assuming that all buffer above scratch buffer in memory space can be
   * shifted down in memory, this function gives optimistic estimation of memory
   * space available for scratch buffer.
   */
  int64_t
  computeMaxScratchBufferSize(triton::gpu::ConvertLayoutOp op,
                              Allocation *allocation,
                              ArrayRef<Allocation::BufferId> liveBuffers) {
    int totalSize = 0;
    auto scratchBufferId = allocation->getBufferId(op.getOperation());
    int64_t scratchBufferSize = allocation->getAllocatedSize(scratchBufferId);
    size_t totalLDSConsumption = 0;
    for (auto buf : liveBuffers)
      totalLDSConsumption = std::max(
          totalLDSConsumption, allocation->getAllocatedInterval(buf).end());
    int64_t freeRequired = totalLDSConsumption - LDSSize;
    auto maxScratchSize = std::max(0l, scratchBufferSize - freeRequired);
    return maxScratchSize;
  }

  SmallVector<LDSBottleneckOperation>
  findLDSBottleneckLayoutConvert(ModuleAllocation &allocAnalysis,
                                 FunctionOpInterface func) {
    SmallVector<LDSBottleneckOperation> candidates;
    auto funcAnalysis = allocAnalysis.getFuncData(func);
    auto liveBuffers = analyzeBufferLiveness(func, funcAnalysis);

    func.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      auto opBuffer = funcAnalysis->getBufferId(cvtOp.getOperation());
      assert(opBuffer != Allocation::InvalidBufferId);
      for (auto bufId : liveBuffers[cvtOp]) {
        auto offset = funcAnalysis->getOffset(bufId);
        auto size = funcAnalysis->getAllocatedSize(bufId);
        if (offset + size > LDSSize) {
          auto maxScratchBufferSize = computeMaxScratchBufferSize(
              cvtOp, funcAnalysis, liveBuffers[cvtOp]);
          candidates.push_back({cvtOp, maxScratchBufferSize});
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
        auto candidates =
            findLDSBottleneckLayoutConvert(allocAnalysis, rootFunc);
        // Try to transform candidate operations to fit them into LDS
        for (auto candidate : candidates)
          tryFitCvtIntoLDS(candidate.op, candidate.LDSSizeTarget);
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
