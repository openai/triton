/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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
#include <memory>
#include <stack>

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------
// This class iterates throught the argument of the `funcOp`, if the argument is
// a pointer, it starts a walk to replace the pointer through all the IR with an
// offset. Only when the pointer is really needed the offset is added to the
// base pointer and passed to the operation that needs the pointer (usually a
// load or a store)
//
// Let's suppose that `arg0` is an offset. The algorithm works like that:
// a) At the beginning the offset is zero, and we associate with `arg0` a
// `FatPtr{arg0, offset}` b) Follow the pointer through the IR, and replace any
// `tt.addptr(%ptr, %offset)` with
//    `add(%fatPoniters[ptr].basePtr, %fatPointers[ptr].offset)`.
// c) When you meet the `tt.load(%ptr)` or `tt.store(%ptr)` instructions,
// replace that instruction with:
//    %fat_ptr = tt.addptr(%fatPointers[ptr].basePtr, %fatPointers[ptr].offset)
//    %data = tt.load(%fat_ptr)
//
class FatPtrPropagation {
private:
  template <typename OpTy> OpTy resolveOp(Operation *op) {
    OpTy resolvedOp = dyn_cast<OpTy>(op);
    if (rewriteOpMap.contains(op))
      resolvedOp = dyn_cast<OpTy>(rewriteOpMap[op]);
    return resolvedOp;
  }

  // Rewrite a store/load operation
  LogicalResult rewriteLoadStoreOp(Operation *op, Location loc, Value ptr) {
    auto fatPtr = pointers[ptr];
    Value basePtr = fatPtr.basePtr;
    Value offset = fatPtr.offset;
    // This is creating `tt.addptr(%fatPtr.bastPtr, %fatPtr.offset)
    Value newPtr = rewriter.create<triton::AddPtrOp>(loc, basePtr.getType(),
                                                     basePtr, offset);
    // Map and replace the load
    IRMapping mapper;
    mapper.map(ptr, newPtr);
    Operation *newOp = rewriter.clone(*op, mapper);
    rewriter.replaceAllOpUsesWith(op, newOp);
    // Delete the old load operation
    opToDelete.insert(op);
    return success();
  }

  // Start from an argument of a function and propagate its
  // fat pointers
  LogicalResult rewritePointer(Value argPtr) {
    SmallVector<OpOperand *> queue;

    // Start the visit
    for (OpOperand &use : argPtr.getUses())
      queue.push_back(&use);

    while (!queue.empty()) {
      OpOperand *curOperand = queue.back();
      queue.pop_back();

      Operation *curOp = curOperand->getOwner();
      Location curLoc = curOp->getLoc();
      Value nextPtr;

      rewriter.setInsertionPoint(curOp);
      LogicalResult curRes = LogicalResult::success();
      if (auto splatOp = dyn_cast<triton::SplatOp>(curOp)) {
        nextPtr = splatOp.getResult();
        auto fatPtr = pointers[splatOp.getSrc()];
        auto ptrShape = splatOp.getResult().getType().getShape();
        auto newOffsetType =
            RankedTensorType::get(ptrShape, fatPtr.offset.getType());
        Value offset = rewriter.create<triton::SplatOp>(curLoc, newOffsetType,
                                                        fatPtr.offset);
        pointers[nextPtr] = FatPtr{nextPtr, offset};
      } else if (auto addPtr = dyn_cast<triton::AddPtrOp>(curOp)) {
        nextPtr = addPtr.getResult();
        auto fatPtr = pointers[addPtr.getPtr()];
        Value offset = rewriter.create<arith::AddIOp>(curLoc, fatPtr.offset,
                                                      addPtr.getOffset());
        pointers[nextPtr] = FatPtr{fatPtr.basePtr, offset};
        opToDelete.insert(curOp);
      } else if (auto loadOp = dyn_cast<triton::LoadOp>(curOp)) {
        curRes = rewriteLoadStoreOp(curOp, curLoc, loadOp.getPtr());
      } else if (auto storeOp = dyn_cast<triton::StoreOp>(curOp)) {
        curRes = rewriteLoadStoreOp(curOp, curLoc, storeOp.getPtr());
      } else if (auto forOp = resolveOp<scf::ForOp>(curOp)) {
        size_t operandNum = curOperand->getOperandNumber();
        FatPtr fatPtr = pointers[curOperand->get()];
        Value offset = fatPtr.offset;
        Value basePtr = fatPtr.basePtr;

        // Replace the forOp with an additional argument (i.e., the curOperand's
        // offset)
        auto newForOp = replaceForOpWithNewSignature(rewriter, forOp, offset);
        rewriteOpMap[forOp] = newForOp;

        newForOp->setOperand(operandNum, basePtr);
        OpOperand *forOperand = &newForOp->getOpOperand(operandNum);
        // This is making sure we propagate the visit from the forOp result
        nextPtr = newForOp.getTiedLoopResult(forOperand);

        // This is making sure we visit the uses withint the forOp region
        Value arg = newForOp.getTiedLoopRegionIterArg(forOperand);
        pointers[arg] = FatPtr{arg, newForOp.getRegionIterArgs().back()};
        for (OpOperand &use : arg.getUses())
          queue.push_back(&use);

        // This is setting the fat pointer for the users of the loop
        // and then propagatin the result
        pointers[nextPtr] = FatPtr{nextPtr, newForOp.getResults().back()};

        opToDelete.insert(forOp);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(curOp)) {
        // Rewriting the yield op is a bit more complicated, because a
        // yield op can be inside of a ForOp, WhileOp(in the AfterRegion) or
        // IfOp
        size_t operandNum = curOperand->getOperandNumber();
        FatPtr fatPtr = pointers[curOperand->get()];
        yieldOp.getResultsMutable().append(fatPtr.offset);
        yieldOp->setOperand(operandNum, fatPtr.basePtr);

        if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
          // Case 1: the yieldOp is contained within an IfOp. One of the
          // two branches is responsible to rewrite the operation. The other
          // branch only update the yieldOp with the right parameters
          if (yieldOp->getBlock() == &ifOp.getThenRegion().front()) {
            auto newIfOp = replaceIfOpWithNewSignature(rewriter, ifOp,
                                                       fatPtr.offset.getType());
            nextPtr = newIfOp.getResult(operandNum);
            pointers[nextPtr] = FatPtr{nextPtr, newIfOp.getResults().back()};
            opToDelete.insert(ifOp);
          }
        } else if (auto whileOp =
                       resolveOp<scf::WhileOp>(yieldOp->getParentOp())) {
          // Case 2: the yieldOp is contained within the AfterRegion of a
          // WhileOp. In this case, we know that the before region should have
          // already been replaced (when we met the WhileOp), hence we can
          // simply replace the WhileOp with a new AfterRegion (and hance a new
          // set of return types)
          auto newWhileOp = replaceWhileOpWithNewSignature(
              rewriter, whileOp, {}, fatPtr.offset.getType());
          nextPtr = newWhileOp.getResult(operandNum);
          pointers[nextPtr] = FatPtr{nextPtr, newWhileOp.getResults().back()};
          rewriteOpMap[whileOp] = newWhileOp;
          opToDelete.insert(whileOp.getOperation());
        }
      } else if (auto whileOp = resolveOp<scf::WhileOp>(curOp)) {
        // WhileOp rewrite happens in two phases: first rewrite the operand list
        // and then rewrite the types when we meet the yieldOp
        size_t operandNum = curOperand->getOperandNumber();
        FatPtr fatPtr = pointers[curOperand->get()];
        Value offset = fatPtr.offset;
        Value basePtr = fatPtr.basePtr;
        // Rewrite the while op with a new set of operands (but with the same
        // set of return types)
        auto newWhileOp =
            replaceWhileOpWithNewSignature(rewriter, whileOp, offset, {});
        newWhileOp->setOperand(operandNum, fatPtr.basePtr);
        Value arg = newWhileOp.getBeforeBody()->getArgument(operandNum);
        // Propagate inside the BeforeRegion
        pointers[arg] =
            FatPtr{arg, newWhileOp.getBeforeBody()->getArguments().back()};
        nextPtr = arg;
        rewriteOpMap[whileOp] = newWhileOp;
        opToDelete.insert(curOp);
      } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(curOp)) {
        // ConditionOp can only be contained within the BeforeRegion of a
        // WhileOp. We already rewrotw the WhileOp with the right operands, so
        // we need only to add the offset the the current operand to be the base
        // pointer and continue the walk inside the AfterRegion
        size_t operandNum = curOperand->getOperandNumber();
        FatPtr fatPtr = pointers[curOperand->get()];
        Value offset = fatPtr.offset;
        Value basePtr = fatPtr.basePtr;
        auto whileOp = dyn_cast<scf::WhileOp>(conditionOp->getParentOp());
        assert(whileOp &&
               "ConditionOp inside an operation different from WhileOp");

        // Update the condition op
        auto afterBlock = whileOp.getAfterBody();
        conditionOp.getArgsMutable().append(offset);

        // Propagate through the after region
        afterBlock->addArgument(offset.getType(), curLoc);
        nextPtr = afterBlock->getArgument(operandNum - 1);
        pointers[nextPtr] = FatPtr{nextPtr, afterBlock->getArguments().back()};
      } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(curOp)) {
        // CondBranchOp is a bit tricky to handle. Because we might be inserting
        // the offset as a TrueDestOperand, which is not the end of
        // `condBrOp.getOperands()`

        auto falseOperands = llvm::to_vector(condBrOp.getFalseDestOperands());
        auto trueOperands = llvm::to_vector(condBrOp.getTrueOperands());
        auto it = llvm::find(falseOperands, curOperand->get());
        bool isFalseOperand = (it != falseOperands.end());
        size_t operandNum = curOperand->getOperandNumber();

        if (rewriteOpMap.contains(condBrOp)) {
          // If we need to use a different condBrOp, we might also need to
          // update `operandNum`
          auto condBranchReplacement =
              dyn_cast<cf::CondBranchOp>(rewriteOpMap[condBrOp]);
          if (isFalseOperand) {
            // The offset needs to be added if we are on the FalseOperands side,
            // but the true operands have been rewritten
            int maybeOffset =
                (condBranchReplacement.getTrueDestOperands().size() !=
                 condBrOp.getTrueDestOperands().size());
            operandNum += maybeOffset;
            curOperand = &condBranchReplacement->getOpOperand(operandNum);
          }
          // Now we need to recompute the currentOperation and its {true,false}
          // operands
          curOp = condBranchReplacement.getOperation();
          falseOperands =
              llvm::to_vector(condBranchReplacement.getFalseDestOperands());
          trueOperands =
              llvm::to_vector(condBranchReplacement.getTrueDestOperands());
          condBrOp = condBranchReplacement;
        }

        // Now we can proceed almost normally
        FatPtr fatPtr = pointers[curOperand->get()];
        Value offset = fatPtr.offset;
        Value basePtr = fatPtr.basePtr;

        Block *falseDest = condBrOp.getFalseDest();
        Block *trueDest = condBrOp.getTrueDest();
        // We need indeed to map the operandNum to the false/true operand
        if (isFalseOperand) {
          falseOperands.push_back(offset);
          falseDest->addArgument(offset.getType(), curLoc);
          nextPtr = falseDest->getArgument(operandNum -
                                           condBrOp.getNumTrueOperands() - 1);
          pointers[nextPtr] = FatPtr{nextPtr, falseDest->getArguments().back()};
        } else {
          trueOperands.push_back(offset);
          trueDest->addArgument(offset.getType(), curLoc);
          nextPtr = trueDest->getArgument(operandNum - 1);
          pointers[nextPtr] = FatPtr{nextPtr, trueDest->getArguments().back()};
        }

        // Create a new condBranch. We cannot simply extend the operands,
        // because this would invalidate other operands pointing at the same
        // cond branch
        auto newCondBranch = rewriter.create<cf::CondBranchOp>(
            curLoc, condBrOp.getCondition(), trueDest, trueOperands, falseDest,
            falseOperands);

        newCondBranch.setOperand(operandNum, basePtr);
        rewriteOpMap[condBrOp] = newCondBranch;
        opToDelete.insert(condBrOp);
      } else if (auto branchOp = dyn_cast<cf::BranchOp>(curOp)) {
        size_t operandNum = curOperand->getOperandNumber();

        FatPtr fatPtr = pointers[curOperand->get()];
        Value offset = fatPtr.offset;
        Value basePtr = fatPtr.basePtr;

        branchOp.getDestOperandsMutable().append(fatPtr.offset);
        branchOp->setOperand(operandNum, basePtr);
        Block *dest = branchOp.getDest();
        dest->addArgument(offset.getType(), curLoc);
        nextPtr = dest->getArgument(operandNum);
        pointers[nextPtr] = {nextPtr, dest->getArguments().back()};
      } else {
        // In theory, we should not meet any other operation. If we do, it'll be
        // unsupported
        curRes = LogicalResult::failure();
      }

      if (failed(curRes))
        return failure();

      // Keep propagating the fat pointer down the IR
      if (nextPtr)
        for (OpOperand &use : nextPtr.getUses())
          queue.push_back(&use);
    }
    return success();
  }

  LogicalResult rewriteRegion(Region &region) {
    for (Value arg : region.getArguments()) {
      if (!isa<triton::PointerType>(arg.getType()))
        continue;
      rewriter.setInsertionPointToStart(&region.front());
      Value zeroOffset =
          rewriter.create<arith::ConstantIntOp>(region.getLoc(), 0, 32);
      pointers[arg] = FatPtr{arg, zeroOffset};
      if (failed(rewritePointer(arg)))
        return failure();
    }
    return success();
  }

public:
  FatPtrPropagation(triton::FuncOp funcOp, MLIRContext *ctx)
      : funcOp(funcOp), rewriter(ctx) {}

  LogicalResult rewrite() {
    auto res = rewriteRegion(funcOp->getRegion(0));

    // Remove operations that are not needed anymore
    for (Operation *op : llvm::reverse(opToDelete))
      op->erase();

    return res;
  }

private:
  // This is the internal representation of a fat pointer: `fatPtr = basePtr +
  // offset`
  struct FatPtr {
    Value basePtr;
    Value offset;
  };

  // This structure is used to point to the right operation during the traversal
  DenseMap<Operation *, Operation *> rewriteOpMap;

  // Rewriter
  mlir::IRRewriter rewriter;

  // Actual funcOp
  triton::FuncOp funcOp;

  // Symbol table: association between pointers and fatPointers
  llvm::MapVector<Value, FatPtr> pointers;

  // List of IR to delete
  SetVector<Operation *> opToDelete;
};

} // namespace

class TritonAMDGPUCanonicalizePointersPass
    : public TritonAMDGPUCanonicalizePointersBase<
          TritonAMDGPUCanonicalizePointersPass> {
public:
  TritonAMDGPUCanonicalizePointersPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    llvm::SmallVector<triton::FuncOp> funcOps;

    // For now, we don't support non-root functions
    // within the same module
    auto callGraph = CallGraph<void *>(m);
    auto walk = m.walk([&](triton::FuncOp funcOp) -> WalkResult {
      if (!callGraph.isRoot(funcOp))
        return WalkResult::interrupt();
      funcOps.push_back(funcOp);
      return WalkResult::advance();
    });

    if (walk.wasInterrupted())
      signalPassFailure();

    for (triton::FuncOp funcOp : funcOps) {
      FatPtrPropagation fatPtrPropagation(funcOp, context);
      auto result = fatPtrPropagation.rewrite();
      if (failed(result))
        signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUCanonicalizePointersPass() {
  return std::make_unique<TritonAMDGPUCanonicalizePointersPass>();
}
