#ifndef TRITONCOMMONGPU_CONVERSION_PASSES
#define TRITONCOMMONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateSharedMemory : Pass<"allocate-shared-memory", "mlir::ModuleOp"> {
    let summary = "Add metadata for shared memory allocation";
    let constructor = "mlir::triton::gpu::createAllocateSharedMemoryPass()";
}
def PerfCollection : Pass<"perf-collection", "mlir::ModuleOp"> {
    let summary = "Collect perf information";
    let constructor = "mlir::triton::gpu::createPerfCollectionPass()";
}

#endif
