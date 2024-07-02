#ifndef TRITON_DIALECT_TRITONAMDGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONAMDGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

std::unique_ptr<Pass> createTritonAMDGPUStreamPipelinePass();

std::unique_ptr<Pass>
createTritonAMDGPUAccelerateMatmulPass(std::string archGenName = std::string(),
                                       int matrixInstructionSize = 0,
                                       int kpack = 1);

std::unique_ptr<Pass> createTritonAMDGPUCanonicalizeLoopsPass();

std::unique_ptr<Pass> createTritonAMDGPUReorderInstructionsPass();

std::unique_ptr<Pass> createTritonAMDGPUVerifier();

std::unique_ptr<Pass> createTritonAMDGPUOptimizeEpiloguePass();

std::unique_ptr<Pass> createTritonAMDGPUCanonicalizePointersPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonAMDGPUTransforms/Passes.h.inc"

} // namespace mlir
#endif
