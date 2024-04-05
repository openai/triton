#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <iterator>
#include <numeric>

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct CoalescePass : public TritonGPUCoalesceBase<CoalescePass> {

  void groupOps(llvm::EquivalenceClasses<Operation *> &slices,
                ModuleAxisInfoAnalysis &axisInfoAnalysis) {

    auto matchesShape = [](const Value &val, RankedTensorType refTensorType) {
      auto rttType = val.getType().dyn_cast<RankedTensorType>();
      return rttType && rttType.getShape() == refTensorType.getShape();
    };

    DenseSet<Operation *> unprocessed;
    for (auto I = slices.begin(), E = slices.end(); I != E; ++I) {
      if (I->isLeader())
        for (auto MI = slices.member_begin(I); MI != slices.member_end(); ++MI)
          unprocessed.insert(*MI);
    }

    while (!unprocessed.empty()) {
      Operation *op = *unprocessed.begin();
      unprocessed.erase(unprocessed.begin());
      Value ptr = getMemAccessPtr(op);
      LDBG("Grouping op: " << *op);
      LLVM_DEBUG({
        llvm::dbgs() << "axis info of pointer: ";
        axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
        llvm::dbgs() << "\n";
      });

      auto tensorType = ptr.getType().cast<RankedTensorType>();
      auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
      SmallVector<unsigned> order = argSort(contiguity);
      LDBG("order=[" << triton::join(order, ", ") << "]");

      for (Operation *use : mlir::multiRootGetSlice(op)) {
        Value val = getMemAccessPtr(use);
        if (!val || !matchesShape(val, tensorType))
          continue;
        auto currOrder =
            argSort(axisInfoAnalysis.getAxisInfo(val)->getContiguity());
        if (order != currOrder)
          continue;
        slices.unionSets(op, use);
        unprocessed.erase(use);
        LDBG("merged with " << *use);
      }
    }
  }

  void
  setCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                       llvm::EquivalenceClasses<Operation *> &slices,
                       int numWarps, int threadsPerWarp,
                       llvm::MapVector<Operation *, Attribute> &layoutMap) {

    for (auto I = slices.begin(), E = slices.end(); I != E; ++I) {
      if (!I->isLeader())
        continue;
      // Loop over each slice.
      LDBG("Considering slice: ");
      struct Layout {
        unsigned maxPerThread = 0;
        unsigned minPerThread = 0;
        unsigned elemNumBytes = 0;
      };

      unsigned sharedPerThread = 0;
      llvm::SmallVector<Layout, 10> layouts;
      for (auto MI = slices.member_begin(I); MI != slices.member_end(); ++MI) {
        auto op = *MI;
        Value ptr = getMemAccessPtr(op);
        auto tensorType = ptr.getType().cast<RankedTensorType>();
        auto shapePerCTA = triton::gpu::getShapePerCTA(tensorType);
        auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
        SmallVector<unsigned> order = argSort(contiguity);
        int numElems = product<int64_t>(shapePerCTA);
        int numThreads = numWarps * threadsPerWarp;
        unsigned elemNumBits = getElementBitWidth(tensorType);
        unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);

        // maximum number of consecutive elements to avoid breaking cross-thread
        // memory coalesing.
        unsigned maxMemopSize = 16;
        unsigned perThread =
            getNumElementsPerThread(op, order, axisInfoAnalysis);
        unsigned maxPerThread =
            std::max(perThread, maxMemopSize / elemNumBytes);

        // minimum number of consecutive elements to ensure a saturation of one
        // memory transaction.
        unsigned memTranscationSize = 128;
        unsigned minPerThread =
            memTranscationSize / (threadsPerWarp * elemNumBytes);
        // minimum number of consecutive elements to ensure a four-byte
        // alignment.
        minPerThread = std::max(minPerThread, 4 / elemNumBytes);

        LLVM_DEBUG({
          DBGS() << "  " << *op;
          DBGS() << "      order=[" << triton::join(order, ", ") << "]\n";
          DBGS() << "     axis info of pointer: ";
          axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
          LDBG("    shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");
          LDBG("    maxPerThread for op: " << maxPerThread);
          LDBG("    minPerThread for op: " << minPerThread);
          DBGS() << "\n";
        });

        layouts.push_back({maxPerThread, minPerThread, elemNumBytes});
        sharedPerThread = std::max(sharedPerThread, perThread);
      }

      // Ensure cross-thread coalescing for every memop.
      for (unsigned I = 0; I < layouts.size(); I++)
        sharedPerThread = std::min(sharedPerThread, layouts[I].maxPerThread);

      // Ensure bandwidth saturation and correct alignment for every memop.
      for (unsigned I = 0; I < layouts.size(); I++)
        sharedPerThread = std::max(sharedPerThread, layouts[I].minPerThread);

      LDBG("sharedPerThread: " << sharedPerThread);

      for (auto MI = slices.member_begin(I); MI != slices.member_end(); ++MI) {
        auto op = *MI;
        Value ptr = getMemAccessPtr(op);
        auto tensorType = ptr.getType().cast<RankedTensorType>();
        auto shapePerCTA = triton::gpu::getShapePerCTA(tensorType);
        auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
        SmallVector<unsigned> order = argSort(contiguity);
        int numElems = product<int64_t>(shapePerCTA);
        int numThreads = numWarps * threadsPerWarp;

        unsigned perThread =
            std::min<int>(sharedPerThread, std::max(numElems / numThreads, 1));

        if (!dyn_cast<triton::LoadOp>(op)) {
          // For ops that can result in a global memory write, we should enforce
          // that each thread handles at most 128 bits, which is the widest
          // available vectorized store op; otherwise, the store will have
          // "gaps" in the memory write at the warp level, resulting in worse
          // performance. For loads, we can expect that the gaps won't matter
          // due to the L1 cache.
          unsigned elemNumBits = getElementBitWidth(tensorType);
          perThread = std::min<int>(
              perThread, getNumElementsPerThread(op, order, axisInfoAnalysis));
        }

        SmallVector<unsigned> sizePerThread(tensorType.getRank(), 1);
        sizePerThread[order[0]] = perThread;
        auto CTALayout = triton::gpu::getCTALayout(tensorType.getEncoding());
        layoutMap[op] = triton::gpu::BlockedEncodingAttr::get(
            &getContext(), tensorType.getShape(), sizePerThread, order,
            numWarps, threadsPerWarp, CTALayout);
      }
    }
  }

  static Type getNewType(Type type, Attribute encoding) {
    RankedTensorType tensorType = type.cast<RankedTensorType>();
    return RankedTensorType::get(tensorType.getShape(),
                                 tensorType.getElementType(), encoding);
  }

  void coalesceOp(Attribute encoding, Operation *op) {
    OpBuilder builder(op);
    // Convert operands
    // For load/store with tensor pointers, we don't have to change the
    // operands' type, we do this by changing the outputs' type of
    // `make_tensor_ptr`
    SmallVector<Value, 4> newArgs;
    for (auto operand : op->getOperands()) {
      auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
      if (tensorType &&
          !tensorType.getEncoding().isa<triton::gpu::SharedEncodingAttr>()) {
        Type newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = isa<triton::gpu::AsyncCopyGlobalToLocalOp>(op);
      newTypes.push_back(isAsync ? t : getNewType(t, encoding));
    }

    // Construct new op with the new encoding
    Operation *newOp =
        builder.create(op->getLoc(), op->getName().getIdentifier(), newArgs,
                       newTypes, op->getAttrs());

    // Cast the results back to the original layout
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();
  }

  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // Collect memops
    llvm::EquivalenceClasses<Operation *> MemopSlices;
    moduleOp.walk([&](Operation *curr) {
      Value ptr = getMemAccessPtr(curr);
      if (!ptr)
        return;
      bool isPtrTensor = false, isTensorPointer = false;
      if (auto tensorType = ptr.getType().dyn_cast<RankedTensorType>())
        isPtrTensor = tensorType.getElementType().isa<PointerType>();
      if (auto ptrType = ptr.getType().dyn_cast<PointerType>())
        isTensorPointer = ptrType.getPointeeType().isa<RankedTensorType>();
      if (!isPtrTensor && !isTensorPointer)
        return;
      MemopSlices.insert(curr);
    });

    // Group memops into slices
    groupOps(MemopSlices, axisInfoAnalysis);

    // For each slice group, we determine what layout the pointers should have
    // for best memory coalescing.
    llvm::MapVector<Operation *, Attribute> layoutMap;
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(moduleOp);
    int threadsPerWarp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    setCoalescedEncoding(axisInfoAnalysis, MemopSlices, numWarps,
                         threadsPerWarp, layoutMap);

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    for (auto &kv : layoutMap) {
      coalesceOp(kv.second, kv.first);
    }
  }
};

std::unique_ptr<Pass> mlir::triton::gpu::createCoalescePass() {
  return std::make_unique<CoalescePass>();
}
