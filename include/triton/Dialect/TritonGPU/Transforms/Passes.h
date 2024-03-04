#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gpu {

std::unique_ptr<Pass> createPipelinePass(int numStages = 3, int numWarps = 4,
                                         int numCTAs = 1,
                                         int computeCapability = 80);

std::unique_ptr<Pass> createAccelerateMatmulPass(int computeCapability = 80);

std::unique_ptr<Pass> createF32DotTCPass();

std::unique_ptr<Pass> createPrefetchPass();

std::unique_ptr<Pass> createCoalescePass();

std::unique_ptr<Pass> createReorderInstructionsPass();

std::unique_ptr<Pass> createReduceDataDuplicationPass();

std::unique_ptr<Pass> createRemoveLayoutConversionsPass();

std::unique_ptr<Pass> createVerifier();

std::unique_ptr<Pass> createOptimizeDotOperandsPass();

std::unique_ptr<Pass> createOptimizeThreadLocalityPass();

} // namespace gpu
} // namespace triton

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

} // namespace mlir
#endif
