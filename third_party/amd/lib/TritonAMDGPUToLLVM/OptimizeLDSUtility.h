#ifndef TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_OPTIMIZE_LDS_UTILITY_H
#define TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_OPTIMIZE_LDS_UTILITY_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
namespace mlir::triton::AMD {

int getCvtOpLDSUsage(RankedTensorType srcTy, RankedTensorType dstTy);

int getCvtOpLDSUsage(triton::gpu::ConvertLayoutOp op);

bool isPowerOfTwo(unsigned x);

std::vector<SmallVector<unsigned>> factorizePowerOf2(int n, int rank);

/**
 * @brief Copy given layout with different warpsPerCTA parameter
 * @param layout original layout
 * @param warpsPerCTA new warpsPerCTA
 * @return create layout
 */
Attribute createTmpLayout(Attribute layout, ArrayRef<unsigned> warpsPerCTA);

/**
 * Creates two chained convert layout operations
 *
 * %1 = cvtOp %0 (srcLayout -> dstLayout) // original operation
 * ->
 * %1 = cvtOp %0 (srcLayout -> dstLayout) // original operation
 * %2 = cvtOp %0 (srcLayout -> tmpLayout) // <returned pair>.first
 * %3 = cvtOp %2 (tmpLayout -> dstLayout) // <returned pair>.second
 *
 * @param builder
 * @param cvtOp original operation
 * @param tmpLayout
 * @return pair of created operations
 */
std::pair<triton::gpu::ConvertLayoutOp, triton::gpu::ConvertLayoutOp>
createNewConvertOps(OpBuilder &builder, triton::gpu::ConvertLayoutOp &cvtOp,
                    Attribute tmpLayout);

struct Resources {
  int LDS;
};

Resources
estimateResourcesForReplacement(OpBuilder builder,
                                mlir::triton::gpu::ConvertLayoutOp cvtOp,
                                Attribute tmpLayout);

} // namespace mlir::triton::AMD

#endif // TRITON_CONVERSION_TRITONAMDGPU_TO_LLVM_OPTIMIZE_LDS_UTILITY_H
