#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonAMDGPUStreamPipeline : Pass<"tritonamdgpu-stream-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Pipeline global loads through registers to shared memory while computing on previous
    tile
  }];

  let constructor = "mlir::createTritonAMDGPUStreamPipelinePass()";

  let dependentDialects = [];
}

def TritonAMDGPUAccelerateMatmul : Pass<"tritonamdgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., AMD matrix cores)
  }];

  let constructor = "mlir::createTritonAMDGPUAccelerateMatmulPass()";

  let dependentDialects = [];

  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">,
    Option<"matrixInstructionSize", "matrix-instruction-size",
           "int32_t", /*default*/"0",
           "enforce matrix instruction MN size">,
    Option<"kPack", "kPack",
           "int32_t", /*default*/"1",
           "KWidth / kBase">
  ];
}

def TritonAMDGPUOptimizeEpilogue : Pass<"tritonamdgpu-optimize-epilogue", "mlir::ModuleOp"> {
  let summary = "Optimize epilogue: (1) Store accumulators directly without going thorough SMEM in epilogue.";

  let description = [{
  }];

  let constructor = "mlir::createTritonAMDGPUOptimizeEpiloguePass()";

  let dependentDialects = [];

}

def TritonAMDGPUCanonicalizePointers : Pass<"tritonamdgpu-canonicalize-pointers", "mlir::ModuleOp"> {
  let summary = "Canonicalize pointers: rewrite pointers passed to load/store operation as an <basePtr, offset> pair.";

  let description = [{
  }];

  let constructor = "mlir::createTritonAMDGPUCanonicalizePointersPass()";

  let dependentDialects = [];

}

def TritonAMDGPUReorderInstructions: Pass<"tritonamdgpu-reorder-instructions", "mlir::ModuleOp"> {
  let summary = "Reorder instructions";

  let description = "This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving "
                    "conversions from shared memory before their first use) and (2) promote LLVM instruction "
                    "order more friendly to `ptxas`.";

  let constructor = "mlir::createTritonAMDGPUReorderInstructionsPass()";

  let dependentDialects = [];
}

#endif
