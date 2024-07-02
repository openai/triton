#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#include <list>

using namespace mlir;

static bool willIncreaseRegisterPressure(Operation *op) {
  if (isa<triton::gpu::LocalLoadOp>(op))
    return true;
  if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op))
    return isa<triton::gpu::DotOperandEncodingAttr>(
        cvt.getType().getEncoding());
  return false;
}

// Gather cone of DFG from the op's basic block.
// - Collect dfg breadth first to keep relative order and
//   reverse order for insertion after. An op may be captured
//   multiple times if DFG reconverges and it will be moved multiple
//   times to keep dominance correctness.
// - Returns bool if this DFG leads to a load op. This
//   condition is not desirable for moving ttg.local_stores
//   early.
static bool gatherDFG(Operation *op, Block *block,
                      SmallVector<Operation *> &dfg) {
  bool leadsToLoad = false;

  std::list<Operation *> oprs{op};
  auto checkOperands = [&](Operation *cop) {
    for (auto operand : cop->getOperands()) {
      if (Operation *oprOp = operand.getDefiningOp()) {
        Block *oprBlk = oprOp->getBlock();
        if (block->findAncestorOpInBlock(*oprOp)) {
          // only move ops that reside in same block
          if (oprBlk == block)
            dfg.push_back(oprOp);
          oprs.push_back(oprOp);
          leadsToLoad |= isa<triton::LoadOp>(oprOp);
        } else {
          // should always be in parent block
          assert(oprBlk->findAncestorOpInBlock(*block->getParentOp()));
        }
      }
    }
  };

  // BFS (filo)
  while (oprs.size()) {
    Operation *nop = oprs.front();
    oprs.pop_front();
    // check next op and sub-regions
    nop->walk(checkOperands);
  }
  return leadsToLoad;
}

// Search thru block to find earliest insertion point for move
// op. This can be either an atomic op or last usage of source pointer.
// Search ends when move op encountered.
static llvm::ilist<Operation>::iterator
findEarlyInsertionPoint(Block *block, Operation *move, Value src) {
  auto loc = block->begin();
  for (auto bi = block->begin(); bi != block->end(); ++bi) {
    auto *op = &*bi;
    if (op == move) // don't move later than current location
      break;
    if (src) {
      // check for ops accessing src
      for (auto opr : op->getOperands()) {
        if (opr == src)
          loc = bi;
      }
    }
    // atomics used for syncronization?
    op->walk([&](Operation *wop) {
      if (isa<triton::AtomicRMWOp, triton::AtomicCASOp>(wop))
        loc = bi;
    });
  }
  return loc;
}

class TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
public:
  TritonAMDGPUReorderInstructionsPass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);
    // Sink conversions into loops when they will increase
    // register pressure
    DenseMap<Operation *, Operation *> opToMove;
    auto moveAfter = [](Operation *lhs, Operation *rhs) {
      lhs->moveAfter(rhs);
    };
    m.walk([&](Operation *op) {
      if (!willIncreaseRegisterPressure(op))
        return;
      if (!op->hasOneUse())
        return;
      Operation *user = op->getUses().begin()->getOwner();
      if (user->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opToMove.insert({op, user});
    });
    for (auto &kv : opToMove)
      kv.first->moveBefore(kv.second);
    opToMove.clear();
    // Move LocalLoadOp and LocalAllocOp immediately after their operands.
    m.walk([&](Operation *op) {
      if (!isa<triton::gpu::LocalLoadOp, triton::gpu::LocalAllocOp>(op) ||
          op->getNumOperands() < 1) {
        return;
      }
      if (Operation *argOp = op->getOperand(0).getDefiningOp())
        moveAfter(op, argOp);
    });
    // Move transpositions just after their definition
    m.walk([&](triton::TransOp op) {
      Operation *argOp = op.getSrc().getDefiningOp();
      if (!argOp)
        return;
      moveAfter(op, argOp);
    });
    SmallVector<Operation *> moveOps;
    // Move global loads early to prefetch.
    m.walk([&](triton::LoadOp op) { moveOps.push_back(op); });
    // Move local_stores early if dependence distance greater than
    // one iteration. Best perf on GEMM when these precede global loads.
    m.walk([&](triton::gpu::LocalStoreOp op) { moveOps.push_back(op); });
    for (auto op : moveOps) {
      // 0. Gather use-def chain in block.
      Block *block = op->getBlock();
      SmallVector<Operation *> dfg{op};
      bool leadsToLoad = gatherDFG(op, block, dfg);
      if (!isa<triton::gpu::LocalStoreOp>(op) || !leadsToLoad) {
        Value src;
        if (auto ld = dyn_cast<triton::LoadOp>(op))
          src = ld.getPtr();
        auto ip = findEarlyInsertionPoint(block, op, src);
        // Remove ops that already precede the insertion point. This
        // is done before moves happen to avoid N^2 complexity in
        // `Operation::isBeforeInBlock`.
        llvm::erase_if(dfg,
                       [&](Operation *op) { return !ip->isBeforeInBlock(op); });
        // Move ops to insertion point.
        for (auto *op : dfg)
          op->moveAfter(block, ip);
      }
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
