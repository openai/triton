#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PERFCOLLECTION
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct PerfCollection
    : public mlir::triton::impl::PerfCollectionBase<PerfCollection> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    mod.walk([&](FunctionOpInterface funcOp) {
      funcOp.walk([&](Operation *op) {
        // Go through key operations convert_layout, load/store, dot.
        // LoadOp, StoreOp, AtomicCASOp, AtomicRMWOp, AsyncCopyGlobalToLocalOp,
        // AsyncTMACopyGlobalToLocalOp
        if (auto convertOp = dyn_cast<mlir::triton::gpu::ConvertLayoutOp>(op)) {
          // size of conversion, does it use smem?
          op->emitRemark() << "has convertOp with size";
        }
        if (auto loadOp = dyn_cast<mlir::triton::LoadOp>(op)) {
          // How can we tell if it's coalesced? We need to wait till lowering
          // when vectorization is decided.
        }
        // DotAsyncOp, DotOp, DotWaitOp
        if (auto dotOp = dyn_cast<mlir::triton::DotOp>(op)) {
          // Show the shape of the dot: [M, N, K], also mma version.
          op->emitRemark() << "has dot";
        }
        if (auto dotOp = dyn_cast<mlir::triton::nvidia_gpu::DotAsyncOp>(op)) {
          // Show the shape of the dot: [M, N, K], also mma version.
          auto dotEnc = dyn_cast<mlir::triton::gpu::NvidiaMmaEncodingAttr>(
              cast<RankedTensorType>(dotOp->getResult(0).getType())
                  .getEncoding());
          if (dotEnc && dotEnc.getVersionMajor() == 3)
            op->emitRemark() << "has async_dot v3";
          else
            op->emitRemark() << "has async_dot";
        }
      });
    });
  }
};

} // namespace

namespace mlir {

namespace triton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>> createPerfCollectionPass() {
  return std::make_unique<PerfCollection>();
}

} // namespace gpu

} // namespace triton

} // namespace mlir
