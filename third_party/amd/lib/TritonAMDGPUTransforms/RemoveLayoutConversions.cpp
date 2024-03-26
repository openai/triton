#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#include "TritonAMDGPUTransforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>

using namespace mlir;
namespace {
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::NvidiaMmaEncodingAttr;
using triton::gpu::SliceEncodingAttr;

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// dot(a, b, load(ptr)) -> add(load(ptr), dot(a, b, 0))
class ConvertDotConvert : public mlir::RewritePattern {
public:
  ConvertDotConvert(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dstOp = cast<triton::gpu::ConvertLayoutOp>(op);
    auto dotOp = dstOp.getSrc().getDefiningOp<triton::DotOp>();
    if (!dotOp)
      return mlir::failure();
    if (std::distance(dstOp->user_begin(), dstOp->user_end()) != 1 ||
        std::distance(dotOp->user_begin(), dotOp->user_end()) != 1)
      return mlir::failure();
    auto cvtOp =
        dotOp.getOperand(2).getDefiningOp<triton::gpu::ConvertLayoutOp>();
    if (!cvtOp)
      return mlir::failure();
    if (!cvtOp.getSrc().getDefiningOp<triton::LoadOp>())
      return failure();
    RankedTensorType dstTy = dstOp.getType();
    RankedTensorType srcTy = cvtOp.getSrc().getType();
    if (dstTy != srcTy)
      return mlir::failure();

    auto _0f = rewriter.create<arith::ConstantOp>(
        op->getLoc(), dstTy.getElementType(),
        rewriter.getZeroAttr(dstTy.getElementType()));
    auto _0 = rewriter.create<triton::SplatOp>(
        op->getLoc(), dotOp.getType(), _0f);
    auto newDot = rewriter.create<triton::DotOp>(
        op->getLoc(), dotOp.getType(), dotOp.getOperand(0),
        dotOp.getOperand(1), _0, dotOp.getInputPrecision(),
        dotOp.getMaxNumImpreciseAcc());
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op->getLoc(), dstTy, newDot.getResult());
    rewriter.replaceOpWithNewOp<arith::AddFOp>(op, newCvt, cvtOp.getSrc());
    return mlir::success();
  }
};

// The current algorithm works by analyzing the IR and doing a one-shot rewrite
// based on the analysis. The algorithm is as follows.
//
// 1. Find all the anchor ops. These are ops that have a layout we want to
//    preserve.
//
// 2. For each anchor, propagate its layout to all its descendants.
//    An op can have multiple ancestors that are anchors, so at this stage an op
//    may have multiple layouts associated with it.
//
// 3. Resolve conflicts by deciding which of the multiple layouts the op should
//    keep, inserting convert-layout ops to resolve conflicts.  After this
//    stage, each value has only one layout associated with it.
//
// 4. Rewrite the IR by walking the function in dominance order. Since we
//    assume the IR is structured we just need to process the regions in the
//    correct order. For each op, rewrite it using the layout decided by the
//    analysis phase.
class LayoutPropagation {
public:
  // Structure to keep track of the layout associated to a value.
  struct LayoutInfo {
    LayoutInfo(Attribute encoding) { encodings.insert(encoding); }
    LayoutInfo() {}
    llvm::SmallSetVector<Attribute, 8> encodings;
  };
  LayoutPropagation(triton::FuncOp F) : funcOp(F) {}
  // Find the anchor ops and set their layout in the data structure.
  void initAnchorLayout();
  // Recursively Propagate the layout to all the users of the anchor ops until
  // we reach a fix point.
  void propagateLayout();
  // Add layouts given in `Info` to the uses of `value`.
  SmallVector<Value> propagateToUsers(Value value, LayoutInfo &info);
  // Set the encoding to all the values and fill out the values with new layout
  // in `changed`.
  void setEncoding(ValueRange values, LayoutInfo &info,
                   SmallVector<Value> &changed, Operation *op);
  // Resolve cases where a value has multiple layouts associated to it.
  void resolveConflicts();
  // Rewrite the IR for the full module.
  void rewrite();
  // Rewrite the IR for a region.
  void rewriteRegion(Region &R);
  // Rewrite an op based on the layout picked by the analysis.
  Operation *rewriteOp(Operation *op);
  // Rewrite a for op based on the layout picked by the analysis.
  Operation *rewriteForOp(scf::ForOp forOp);
  Operation *rewriteWhileOp(scf::WhileOp whileOp);
  Operation *rewriteIfOp(scf::IfOp ifOp);
  void rewriteYieldOp(scf::YieldOp yieldOp);
  void rewriteConditionOp(scf::ConditionOp conditionOp);
  void rewriteReduceToScalar(Operation *reduceOp);
  Operation *cloneElementwise(OpBuilder &rewriter, Operation *op,
                              Attribute encoding);
  // Map the original value to the rewritten one.
  void map(Value old, Value newV);
  // Return the mapped value in the given encoding. This will insert a convert
  // if the encoding is different than the encoding decided at resolve time.
  Value getValueAs(Value value, Attribute encoding);
  // Dump the current stage of layout information.
  void dump();

private:
  // map from value to layout information.
  llvm::MapVector<Value, LayoutInfo> layouts;
  // map of the values rewrite based on their encoding.
  DenseMap<std::pair<Value, Attribute>, Value> rewriteMapping;
  SetVector<Operation *> opToDelete;
  triton::FuncOp funcOp;
};

} // namespace

// Look ahead to at the transitive uses and see if there is a convert to mma
// operations.
static bool hasConvertToMMATransisitiveUse(Operation *op, Attribute encoding) {
  SmallVector<Value> queue = {op->getResult(0)};
  SetVector<Operation *> forwardSlice;
  llvm::SmallDenseSet<Value> seen;
  while (!queue.empty()) {
    Value currentValue = queue.back();
    queue.pop_back();
    getForwardSlice(currentValue, &forwardSlice);
    for (Operation *op : forwardSlice) {
      if (auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
        Attribute dstEncoding = convertOp.getResult()
                                    .getType()
                                    .cast<RankedTensorType>()
                                    .getEncoding();
        if (auto mmaLayout =
                dstEncoding.dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>())
          return (mmaLayout.getVersionMajor() > 1) ? true
                                                   : mmaLayout == encoding;
        if (dstEncoding.isa<triton::gpu::DotOperandEncodingAttr>())
          return encoding.cast<triton::gpu::NvidiaMmaEncodingAttr>()
                     .getVersionMajor() > 1;
      }
      auto yield = dyn_cast<scf::YieldOp>(op);
      if (!yield)
        continue;
      auto forOp = dyn_cast<scf::ForOp>(yield.getOperation()->getParentOp());
      if (!forOp)
        continue;
      for (OpOperand &operand : yield->getOpOperands()) {
        Operation *def = operand.get().getDefiningOp();
        if (def && forwardSlice.count(def) &&
            (seen.insert(operand.get()).second == true))
          queue.push_back(forOp.getRegionIterArg(operand.getOperandNumber()));
      }
    }
  }
  return false;
}

#ifdef USE_ROCM
// Look ahead to at the transitive uses and see if there is a convert to mfma or
// wmma operations.
// TODO: unify with hasConvertToMMATransisitiveUse?
static bool hasConvertToAmdMmaTransisitiveUse(Operation *op,
                                              Attribute encoding) {
  SmallVector<Value> queue = {op->getResult(0)};
  SetVector<Operation *> forwardSlice;
  llvm::SmallDenseSet<Value> seen;
  while (!queue.empty()) {
    Value currentValue = queue.back();
    queue.pop_back();
    getForwardSlice(currentValue, &forwardSlice);
    for (Operation *op : forwardSlice) {
      if (auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
        if (convertOp.getResult()
                .getType()
                .cast<RankedTensorType>()
                .getEncoding() == encoding)
          return true;
      }
      auto yield = dyn_cast<scf::YieldOp>(op);
      if (!yield)
        continue;
      auto forOp = dyn_cast<scf::ForOp>(yield.getOperation()->getParentOp());
      if (!forOp)
        continue;
      for (OpOperand &operand : yield->getOpOperands()) {
        Operation *def = operand.get().getDefiningOp();
        if (def && forwardSlice.count(def) &&
            (seen.insert(operand.get()).second == true))
          queue.push_back(forOp.getRegionIterArg(operand.getOperandNumber()));
      }
    }
  }
  return false;
}
#endif


// Return true if the op is an op with a layout we don't want to change. We will
// propagate the layout starting from anchor ops.
static bool isLayoutAnchor(Operation *op) {
  if (isa<triton::LoadOp, triton::StoreOp>(op))
    return isExpensiveLoadOrStore(op);
  if (isa<triton::ReshapeOp, triton::DotOp, triton::AtomicRMWOp,
          triton::AtomicCASOp>(op))
    return true;
  return false;
}

void LayoutPropagation::initAnchorLayout() {
  funcOp.walk([&](Operation *op) {
    if (isLayoutAnchor(op)) {
      for (auto result : op->getResults()) {
        if (auto tensorType = result.getType().dyn_cast<RankedTensorType>()) {
          // Workaround, don't popagate MMA layout unless there is a convert
          // back to mma further down to avoid generating reduction with MMA
          // layout that may have lower performance.
          // This can be improved with more aggressive backward propagation.
          if (tensorType.getEncoding()
                  .isa<triton::gpu::NvidiaMmaEncodingAttr>() &&
              !hasConvertToMMATransisitiveUse(op, tensorType.getEncoding()))
            continue;
#ifdef USE_ROCM
          // Workaround to not propagate MFMA layout in case there are
          // no chained dots MFMA layout is expensive to convert, so we want
          // to convert it to something else as soon as possible.
          // It saves LDS space in some cases.
          //
          // TODO: rework this heuristic if we can store MFMA layout directly
          // into global memory.
          //
          // Also relevant for WMMA layout.
          if (tensorType.getEncoding()
                  .isa<triton::gpu::AMDMfmaEncodingAttr,
                       triton::gpu::AMDWmmaEncodingAttr>() &&
              !hasConvertToAmdMmaTransisitiveUse(op, tensorType.getEncoding()))
            continue;
#endif
          layouts.insert({result, LayoutInfo(tensorType.getEncoding())});
        }
      }
    }
  });
}

void LayoutPropagation::setEncoding(ValueRange values, LayoutInfo &info,
                                    SmallVector<Value> &changed,
                                    Operation *op) {
  for (Value value : values) {
    if (!value.getType().isa<RankedTensorType>())
      continue;
    bool hasChanged = false;
    for (auto encoding : info.encodings) {
      std::optional<Attribute> dstEncoding;
      if (isa<ConvertLayoutOp>(op)) {
        // Try to remove the convert by making the dst encoding match the source
        // encoding.
        dstEncoding = encoding;
      } else {
        dstEncoding = inferDstEncoding(op, encoding);
      }
      if (dstEncoding)
        hasChanged |= layouts[value].encodings.insert(*dstEncoding);
    }
    if (hasChanged)
      changed.push_back(value);
  }
}

SmallVector<Value> LayoutPropagation::propagateToUsers(Value value,
                                                       LayoutInfo &info) {
  SmallVector<Value> changed;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      setEncoding({arg, result}, info, changed, user);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      Value arg = whileOp.getBeforeArguments()[use.getOperandNumber()];
      setEncoding({arg}, info, changed, user);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto parent = yieldOp->getParentOp();
      SmallVector<Value> valuesToPropagate;
      if (isa<scf::ForOp, scf::IfOp>(parent))
        valuesToPropagate.push_back(parent->getResult(use.getOperandNumber()));
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        valuesToPropagate.push_back(
            forOp.getRegionIterArg(use.getOperandNumber()));
      if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
        valuesToPropagate.push_back(
            whileOp.getBeforeArguments()[use.getOperandNumber()]);
        valuesToPropagate.push_back(
            whileOp->getOperand(use.getOperandNumber()));
      }
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
        setEncoding(valuesToPropagate, info, changed, user);
      continue;
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      // Skip arg 0 as it is the condition.
      unsigned argIndex = use.getOperandNumber() - 1;
      Value afterArg = whileOp.getAfterArguments()[argIndex];
      Value result = whileOp->getResult(argIndex);
      setEncoding({afterArg, result}, info, changed, user);
      continue;
    }
      if (user->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
          user->hasTrait<mlir::OpTrait::Elementwise>() ||
          isa<triton::ReduceOp, triton::ExpandDimsOp,
              triton::gpu::ConvertLayoutOp>(user)) {
        setEncoding(user->getResults(), info, changed, user);
        continue;
      }
  }
  return changed;
}

void LayoutPropagation::propagateLayout() {
  SmallVector<Value> queue;
  for (auto it : layouts) {
    queue.push_back(it.first);
  }
  while (!queue.empty()) {
    Value currentValue = queue.back();
    LayoutInfo info = layouts[currentValue];
    queue.pop_back();
    SmallVector<Value> changed = propagateToUsers(currentValue, info);
    queue.insert(queue.end(), changed.begin(), changed.end());
  }
}

void LayoutPropagation::resolveConflicts() {
  for (auto &it : layouts) {
    Operation *op = it.first.getDefiningOp();
    LayoutInfo &info = it.second;
    if (info.encodings.size() <= 1)
      continue;
    // Hacky resolve, prefer block encoding.
    // TODO: add a proper heuristic.
    int maxSizePerThread = 1;
    Attribute encoding = *info.encodings.begin();
    bool isLoadOrStore =
        op && isa<triton::LoadOp, triton::StoreOp, triton::AtomicRMWOp,
                  triton::AtomicCASOp>(op);
    for (Attribute e : info.encodings) {
      if ((isLoadOrStore && e.isa<triton::gpu::BlockedEncodingAttr>()) ||
          (!isLoadOrStore && e.isa<triton::gpu::NvidiaMmaEncodingAttr>())) {
        encoding = e;
        break;
      }
    }
    info.encodings.clear();
    info.encodings.insert(encoding);
  }
}

void LayoutPropagation::dump() {
  for (auto it : layouts) {
    llvm::errs() << "Value: ";
    OpPrintingFlags flags;
    flags.skipRegions();
    it.first.print(llvm::errs(), flags);
    llvm::errs() << " \n encoding:\n";
    for (auto encoding : it.second.encodings) {
      encoding.print(llvm::errs());
      llvm::errs() << "\n";
    }
    llvm::errs() << "--\n";
  }
}

void LayoutPropagation::rewrite() { rewriteRegion(funcOp->getRegion(0)); }

static bool reduceToScalar(Operation *op) {
  // For reductions returning a scalar we can change the src encoding without
  // affecting the output.
  return isa<triton::ReduceOp>(op) &&
         !op->getResultTypes()[0].isa<RankedTensorType>();
}

void LayoutPropagation::rewriteRegion(Region &region) {
  SmallVector<Region *> queue = {&region};
  while (!queue.empty()) {
    Region *currentRegion = queue.back();
    queue.pop_back();
    for (Operation &op : currentRegion->getOps()) {
      bool needRewrite = false;
      SmallVector<Value> results = op.getResults();
      for (Value result : results) {
        auto it = layouts.find(result);
        // If we haven't mapped this value skip.
        if (it == layouts.end())
          continue;
        LayoutInfo &info = it->second;
        assert(info.encodings.size() == 1 &&
               "we should have resolved to a single encoding");
        auto encoding = result.getType().cast<RankedTensorType>().getEncoding();
        // If the encoding is already what we want skip.
        if (encoding == *info.encodings.begin())
          continue;
        needRewrite = true;
      }
      if (needRewrite) {
        Operation *newOp = rewriteOp(&op);
        for (Region &R : newOp->getRegions())
          queue.push_back(&R);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
        rewriteYieldOp(yieldOp);
      } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(&op)) {
        rewriteConditionOp(conditionOp);
      } else if (reduceToScalar(&op)) {
        rewriteReduceToScalar(&op);
      } else {
        // If we don't need to rewrite the op we still need to remap the
        // operands.
        for (OpOperand &operand : op.getOpOperands()) {
          auto it = layouts.find(operand.get());
          if (it == layouts.end())
            continue;
          Attribute encoding =
              operand.get().getType().cast<RankedTensorType>().getEncoding();
          Value newOperand = getValueAs(operand.get(), encoding);
          op.setOperand(operand.getOperandNumber(), newOperand);
        }
        for (Region &R : op.getRegions())
          queue.push_back(&R);
      }
    }
  }
  for (Operation *op : llvm::reverse(opToDelete))
    op->erase();
}

void LayoutPropagation::map(Value old, Value newV) {
  rewriteMapping[{old, newV.getType().cast<RankedTensorType>().getEncoding()}] =
      newV;
}

Value LayoutPropagation::getValueAs(Value value, Attribute encoding) {
  if (auto tensorType = value.getType().dyn_cast<RankedTensorType>()) {
    Value rewrittenValue;
    auto layoutIt = layouts.find(value);
    if (layoutIt == layouts.end()) {
      rewrittenValue = value;
    } else {
      assert(layoutIt->second.encodings.size() == 1 &&
             "we should have resolved to a single encoding");
      Attribute encodingPicked = *(layoutIt->second.encodings.begin());
      if (encodingPicked == tensorType.getEncoding())
        rewrittenValue = value;
      else
        rewrittenValue = rewriteMapping[{value, encodingPicked}];
    }
    assert(rewrittenValue);
    if (rewrittenValue.getType().cast<RankedTensorType>().getEncoding() ==
        encoding)
      return rewrittenValue;
    OpBuilder rewriter(value.getContext());
    rewriter.setInsertionPointAfterValue(rewrittenValue);
    auto tmpType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), encoding);
    Value converted = rewriter.create<triton::gpu::ConvertLayoutOp>(
        value.getLoc(), tmpType, rewrittenValue);
    // TODO: we could cache the conversion.
    return converted;
  }
  return value;
}

Operation *LayoutPropagation::cloneElementwise(OpBuilder &rewriter,
                                               Operation *op,
                                               Attribute encoding) {
  Operation *newOp = rewriter.clone(*op);
  for (OpOperand &operand : op->getOpOperands())
    newOp->setOperand(
        operand.getOperandNumber(),
        getValueAs(operand.get(), *inferSrcEncoding(op, encoding)));
  for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
    auto origType = op->getResult(i).getType().dyn_cast<RankedTensorType>();
    if (!origType)
      continue;
    auto newType = RankedTensorType::get(origType.getShape(),
                                         origType.getElementType(), encoding);
    newOp->getResult(i).setType(newType);
  }
  return newOp;
}

Operation *LayoutPropagation::rewriteForOp(scf::ForOp forOp) {
  SmallVector<Value> operands;
  OpBuilder rewriter(forOp);
  for (auto [operand, result] :
       llvm::zip(forOp.getInitArgs(), forOp.getResults())) {
    Value convertedOperand = operand;
    if (layouts.count(result))
      convertedOperand =
          getValueAs(operand, *layouts[result].encodings.begin());
    operands.push_back(convertedOperand);
  }
  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), operands);
  newForOp->setAttrs(forOp->getAttrs());
  newForOp.getBody()->getOperations().splice(
      newForOp.getBody()->getOperations().begin(),
      forOp.getBody()->getOperations());

  for (auto [oldResult, newResult] :
       llvm::zip(forOp.getResults(), newForOp.getResults())) {
    if (oldResult.getType() == newResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    map(oldResult, newResult);
  }

  for (auto [oldArg, newArg] : llvm::zip(forOp.getBody()->getArguments(),
                                         newForOp.getBody()->getArguments())) {
    if (oldArg.getType() == newArg.getType()) {
      oldArg.replaceAllUsesWith(newArg);
      continue;
    }
    map(oldArg, newArg);
  }
  return newForOp.getOperation();
}

Operation *LayoutPropagation::rewriteWhileOp(scf::WhileOp whileOp) {
  SmallVector<Value> operands;
  SmallVector<Type> returnTypes;
  OpBuilder rewriter(whileOp);
  for (auto [operand, arg] :
       llvm::zip(whileOp->getOperands(), whileOp.getBeforeArguments())) {
    Value convertedOperand = operand;
    if (layouts.count(arg))
      convertedOperand = getValueAs(operand, *layouts[arg].encodings.begin());
    operands.push_back(convertedOperand);
  }
  for (Value ret : whileOp.getResults()) {
    auto it = layouts.find(ret);
    if (it == layouts.end()) {
      returnTypes.push_back(ret.getType());
      continue;
    }
    auto origType = ret.getType().dyn_cast<RankedTensorType>();
    auto newType =
        RankedTensorType::get(origType.getShape(), origType.getElementType(),
                              it->second.encodings[0]);
    returnTypes.push_back(newType);
  }

  auto newWhileOp =
      rewriter.create<scf::WhileOp>(whileOp.getLoc(), returnTypes, operands);
  SmallVector<Type> argsTypesBefore;
  for (Value operand : operands)
    argsTypesBefore.push_back(operand.getType());
  SmallVector<Location> bbArgLocsBefore(argsTypesBefore.size(),
                                        whileOp.getLoc());
  SmallVector<Location> bbArgLocsAfter(returnTypes.size(), whileOp.getLoc());
  rewriter.createBlock(&newWhileOp.getBefore(), {}, argsTypesBefore,
                       bbArgLocsBefore);
  rewriter.createBlock(&newWhileOp.getAfter(), {}, returnTypes, bbArgLocsAfter);

  for (int i = 0; i < whileOp.getNumRegions(); ++i) {
    newWhileOp->getRegion(i).front().getOperations().splice(
        newWhileOp->getRegion(i).front().getOperations().begin(),
        whileOp->getRegion(i).front().getOperations());
  }

  auto remapArg = [&](Value oldVal, Value newVal) {
    if (oldVal.getType() == newVal.getType())
      oldVal.replaceAllUsesWith(newVal);
    else
      map(oldVal, newVal);
  };
  for (auto [oldResult, newResult] :
       llvm::zip(whileOp.getResults(), newWhileOp.getResults()))
    remapArg(oldResult, newResult);
  for (auto [oldArg, newArg] :
       llvm::zip(whileOp.getBeforeArguments(), newWhileOp.getBeforeArguments()))
    remapArg(oldArg, newArg);
  for (auto [oldArg, newArg] :
       llvm::zip(whileOp.getAfterArguments(), newWhileOp.getAfterArguments()))
    remapArg(oldArg, newArg);
  return newWhileOp.getOperation();
}

Operation *LayoutPropagation::rewriteIfOp(scf::IfOp ifOp) {
  SmallVector<Value> operands;
  OpBuilder rewriter(ifOp);
  SmallVector<Type> newResultTypes(ifOp->getResultTypes());
  for (unsigned i = 0, e = ifOp->getNumResults(); i < e; ++i) {
    auto it = layouts.find(ifOp->getResult(i));
    if (it == layouts.end())
      continue;
    auto origType = ifOp->getResult(i).getType().cast<RankedTensorType>();
    Attribute encoding = *(it->second.encodings.begin());
    newResultTypes[i] = RankedTensorType::get(
        origType.getShape(), origType.getElementType(), encoding);
  }
  auto newIfOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                            ifOp.getCondition(), true, true);
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  for (auto [oldResult, newResult] :
       llvm::zip(ifOp.getResults(), newIfOp.getResults())) {
    if (oldResult.getType() == newResult.getType()) {
      oldResult.replaceAllUsesWith(newResult);
      continue;
    }
    map(oldResult, newResult);
  }
  return newIfOp.getOperation();
}

void LayoutPropagation::rewriteYieldOp(scf::YieldOp yieldOp) {
  Operation *parentOp = yieldOp->getParentOp();
  for (OpOperand &operand : yieldOp->getOpOperands()) {
    Type yieldType = operand.get().getType();
    if (isa<scf::ForOp, scf::IfOp>(parentOp))
      yieldType = parentOp->getResult(operand.getOperandNumber()).getType();
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
      yieldType =
          whileOp.getBeforeArguments()[operand.getOperandNumber()].getType();
    auto tensorType = yieldType.dyn_cast<RankedTensorType>();
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    yieldOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteConditionOp(scf::ConditionOp conditionOp) {
  scf::WhileOp whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
  for (unsigned i = 1; i < conditionOp->getNumOperands(); ++i) {
    OpOperand &operand = conditionOp->getOpOperand(i);
    Type argType = whileOp->getResult(operand.getOperandNumber() - 1).getType();
    auto tensorType = argType.dyn_cast<RankedTensorType>();
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    conditionOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteReduceToScalar(Operation *reduceOp) {
  OpBuilder rewriter(reduceOp);
  Attribute srcEncoding;
  // Since all the operands need to have the same encoding pick the first one
  // and use it for all the operands.
  for (Value operand : reduceOp->getOperands()) {
    auto it = layouts.find(operand);
    if (it != layouts.end()) {
      srcEncoding = it->second.encodings[0];
      break;
    }
  }
  if (!srcEncoding)
    return;
  for (OpOperand &operand : reduceOp->getOpOperands()) {
    Value newOperand = getValueAs(operand.get(), srcEncoding);
    reduceOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

Operation *LayoutPropagation::rewriteOp(Operation *op) {
  opToDelete.insert(op);
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return rewriteForOp(forOp);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return rewriteWhileOp(whileOp);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewriteIfOp(ifOp);
  OpBuilder rewriter(op);
  Attribute encoding = *layouts[op->getResult(0)].encodings.begin();
  if (auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
    Attribute srcEncoding =
        convertOp.getSrc().getType().cast<RankedTensorType>().getEncoding();
    auto it = layouts.find(convertOp.getSrc());
    if (it != layouts.end())
      srcEncoding = *(it->second.encodings.begin());
    Value src = getValueAs(convertOp.getSrc(), srcEncoding);
    auto tensorType = op->getResult(0).getType().cast<RankedTensorType>();
    auto newType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), encoding);
    auto cvt = rewriter.create<triton::gpu::ConvertLayoutOp>(op->getLoc(),
                                                             newType, src);
    map(op->getResult(0), cvt.getResult());
    return cvt.getOperation();
  }
  if (canFoldIntoConversion(op, encoding)) {
    Operation *newOp = rewriter.clone(*op);
    auto tensorType = op->getResult(0).getType().cast<RankedTensorType>();
    auto newType = RankedTensorType::get(tensorType.getShape(),
                                         tensorType.getElementType(), encoding);
    auto cvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op->getLoc(), newType, newOp->getResult(0));
    map(op->getResult(0), cvt.getResult());
    return cvt.getOperation();
  }
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::Elementwise>() ||
      isa<triton::ReduceOp, triton::ExpandDimsOp,
          triton::gpu::ConvertLayoutOp>(op)) {
    Operation *newOp = cloneElementwise(rewriter, op, encoding);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults()))
      map(oldResult, newResult);
    return newOp;
  }
  llvm::report_fatal_error("unexpected op in rewrite");
  return nullptr;
}

static bool canBeRemat(Operation *op) {
  if (isa<triton::LoadOp, triton::StoreOp>(op))
    return !isExpensiveLoadOrStore(op);
  if (isa<triton::AtomicRMWOp,
          triton::AtomicCASOp, triton::DotOp>(op))
    return false;
  if (isa<scf::IfOp, scf::WhileOp, scf::ConditionOp>(op))
    return false;

  return true;
}

static void rewriteSlice(SetVector<Value> &slice,
                         DenseMap<Value, Attribute> &layout,
                         ConvertLayoutOp convertOp, IRMapping &mapping) {
  SetVector<Operation *> opsToRewrite;
  for (Value v : slice) {
    if (v.getDefiningOp()) {
      opsToRewrite.insert(v.getDefiningOp());
    } else {
      opsToRewrite.insert(v.cast<BlockArgument>().getOwner()->getParentOp());
      // We also need to rewrite the yield op.
      opsToRewrite.insert(v.cast<BlockArgument>().getOwner()->getTerminator());
    }
  }
  opsToRewrite = multiRootTopologicalSort(opsToRewrite);

  SmallVector<Operation *> deadLoops;
  IRRewriter builder(slice.begin()->getContext());
  for (Operation *op : opsToRewrite) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Keep a mapping of the operands index to the new operands index.
      SmallVector<std::pair<size_t, size_t>> argMapping;
      SmallVector<Value> newOperands;
      for (auto arg : forOp.getRegionIterArgs()) {
        if (slice.count(arg)) {
          OpOperand &initVal = *forOp.getTiedLoopInit(arg);
          argMapping.push_back(std::make_pair(
              forOp.getTiedLoopResult(&initVal).getResultNumber(),
              forOp.getInitArgs().size() + newOperands.size()));
          newOperands.push_back(mapping.lookup(initVal.get()));
        }
      }
      // Create a new for loop with the new operands.
      scf::ForOp newForOp =
          replaceForOpWithNewSignature(builder, forOp, newOperands);
      deadLoops.push_back(forOp.getOperation());
      Block &loopBody = *newForOp.getBody();
      for (auto m : argMapping) {
        mapping.map(newForOp.getResult(m.first), newForOp.getResult(m.second));
        int numIndVars = newForOp.getNumInductionVars();
        mapping.map(loopBody.getArgument(m.first + numIndVars),
                    loopBody.getArgument(m.second + numIndVars));
      }
      continue;
    }
    builder.setInsertionPoint(op);
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      auto yieldOperands = llvm::to_vector(yieldOp.getOperands());
      for (Value operand : yieldOp.getOperands()) {
        if (slice.count(operand) == 0)
          continue;
        yieldOperands.push_back(mapping.lookup(operand));
      }
      builder.create<scf::YieldOp>(op->getLoc(), yieldOperands);
      op->erase();
      continue;
    }
    if (isa<arith::ConstantOp>(op)) {
      Operation *newOp = builder.clone(*op);
      auto tensorType = op->getResult(0).getType().cast<RankedTensorType>();
      auto newType = RankedTensorType::get(tensorType.getShape(),
                                           tensorType.getElementType(),
                                           layout[op->getResult(0)]);
      auto cvt = builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newType, newOp->getResult(0));
      mapping.map(op->getResult(0), cvt.getResult());
      continue;
    }
    Operation *newOp = builder.clone(*op, mapping);
    for (auto [old, newV] : llvm::zip(op->getResults(), newOp->getResults())) {
      auto it = layout.find(old);
      if (it == layout.end())
        continue;
      auto newType = RankedTensorType::get(
          old.getType().cast<RankedTensorType>().getShape(),
          old.getType().cast<RankedTensorType>().getElementType(), it->second);
      newV.setType(newType);
    }
  }
  convertOp.replaceAllUsesWith(mapping.lookup(convertOp.getSrc()));
  convertOp.erase();
  for (Operation *op : deadLoops)
    op->erase();
}

static void rewriteSlice(SetVector<Value> &slice,
                         DenseMap<Value, Attribute> &layout,
                         ConvertLayoutOp convertOp) {
  IRMapping mapping;
  rewriteSlice(slice, layout, convertOp, mapping);
}

static LogicalResult getRematerializableSlice(
    Value root, Attribute rootEncoding, SetVector<Value> &slice,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation = nullptr) {
  LogicalResult result = getConvertBackwardSlice(root, slice, rootEncoding,
                                                 layout, stopPropagation);
  if (result.failed() || slice.empty())
    return failure();

  // Check if all the operations in the slice can be rematerialized.
  for (Value v : slice) {
    if (Operation *op = v.getDefiningOp()) {
      if (!canBeRemat(op))
        return failure();
    }
  }
  return success();
}

static void backwardRematerialization(ConvertLayoutOp convertOp) {
  // we don't handle conversions to DotOperandEncodingAttr
  // this is a heuristics to accommodate fused attention
  auto targetType = convertOp->getResultTypes()[0].cast<RankedTensorType>();
  if (targetType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>())
    return;

  // 1. Take a backward slice of all the tensor dependencies that can be
  // rematerialized.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrc(), targetType.getEncoding(), slice, layout);
  if (result.failed())
    return;

  // 2. Rewrite the slice.
  rewriteSlice(slice, layout, convertOp);
}

// For convert left we try to hoist them above type extension to reduce the cost
// of the convert.
static void hoistConvertOnTopOfExtOrBroadcast(ConvertLayoutOp convertOp) {
  // we don't handle conversions to DotOperandEncodingAttr
  // this is a heuristics to accommodate fused attention
  auto targetType = convertOp->getResultTypes()[0].cast<RankedTensorType>();
  if (targetType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>())
    return;

  auto isExtOrBroadcastOp = [](Operation *op) {
    return isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp,
               triton::BroadcastOp, triton::ExpandDimsOp>(op);
  };
  // 1. Take a backward slice of all the tensor dependencies.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  LogicalResult result =
      getRematerializableSlice(convertOp.getSrc(), targetType.getEncoding(),
                               slice, layout, isExtOrBroadcastOp);
  if (result.failed())
    return;

  Operation *extOrBroadcatOp = nullptr;
  unsigned sliceSize = slice.size();
  for (unsigned i = 0; i < sliceSize; i++) {
    Value v = slice[i];
    Operation *op = v.getDefiningOp();
    if (!op)
      continue;
    if (isExtOrBroadcastOp(op)) {
      SetVector<Value> tempSlice;
      DenseMap<Value, Attribute> tempLayout;
      std::optional<Attribute> srcEncoding = inferSrcEncoding(op, layout[v]);
      if (!srcEncoding)
        return;
      LogicalResult result = getRematerializableSlice(
          op->getOperand(0), *srcEncoding, tempSlice, tempLayout);
      // If we can rematerialize the rest of the ext slice we can ignore this
      // ext as it won't need a convert.
      if (result.succeeded()) {
        slice.insert(tempSlice.begin(), tempSlice.end());
        layout.insert(tempLayout.begin(), tempLayout.end());
        continue;
      }
      // Only apply it if there is a single ext op otherwise we would have to
      // duplicate the convert.
      if (extOrBroadcatOp != nullptr)
        return;
      extOrBroadcatOp = op;
    }
  }

  if (extOrBroadcatOp == nullptr)
    return;
  Attribute dstEncoding = layout[extOrBroadcatOp->getResult(0)];
  std::optional<Attribute> srcEncoding =
      inferSrcEncoding(extOrBroadcatOp, dstEncoding);
  if (!srcEncoding)
    return;
  // Move the convert before the ext op and rewrite the slice.
  OpBuilder builder(extOrBroadcatOp);
  auto tensorType =
      extOrBroadcatOp->getOperand(0).getType().cast<RankedTensorType>();
  auto newType = RankedTensorType::get(
      tensorType.getShape(), tensorType.getElementType(), *srcEncoding);
  auto newConvertOp = builder.create<ConvertLayoutOp>(
      convertOp.getLoc(), newType, extOrBroadcatOp->getOperand(0));
  Operation *newExtOrBroadcast = builder.clone(*extOrBroadcatOp);
  newExtOrBroadcast->setOperand(0, newConvertOp.getResult());
  auto oldExtOrBroadcastType =
      extOrBroadcatOp->getResult(0).getType().cast<RankedTensorType>();
  Type newExtOrBroadcasrType = RankedTensorType::get(
      oldExtOrBroadcastType.getShape(), oldExtOrBroadcastType.getElementType(),
      dstEncoding);
  newExtOrBroadcast->getResult(0).setType(newExtOrBroadcasrType);
  IRMapping mapping;
  mapping.map(extOrBroadcatOp->getResult(0), newExtOrBroadcast->getResult(0));
  slice.remove(extOrBroadcatOp->getResult(0));
  // 3. Rewrite the slice.
  rewriteSlice(slice, layout, convertOp, mapping);
}

static void backwardRematerialization(ModuleOp module) {
  SmallVector<ConvertLayoutOp> convertOps;
  module.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    backwardRematerialization(convertOp);
  }
}

static void hoistConvert(ModuleOp module) {
  SmallVector<ConvertLayoutOp> convertOps;
  module.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    hoistConvertOnTopOfExtOrBroadcast(convertOp);
  }
}

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

class TritonAMDGPURemoveLayoutConversionsPass
    : public TritonAMDGPURemoveLayoutConversionsBase<
          TritonAMDGPURemoveLayoutConversionsPass> {
public:
  TritonAMDGPURemoveLayoutConversionsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // 1. Propagate layout forward starting from "anchor" ops.
    m.walk([](triton::FuncOp funcOp) {
      LayoutPropagation layoutPropagation(funcOp);
      layoutPropagation.initAnchorLayout();
      layoutPropagation.propagateLayout();
      layoutPropagation.resolveConflicts();
      layoutPropagation.rewrite();
    });

    mlir::RewritePatternSet cleanUpPatterns(context);
    ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns, context);
    if (mlir::applyPatternsAndFoldGreedily(m, std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }

    // 2. For convert ops left try to rematerialize the slice of producer
    // operation to avoid having to convert.
    backwardRematerialization(m);
    // 3. For converts left try to hoist them above cast generating larger size
    // types in order to reduce the cost of the convert op.
    hoistConvert(m);

    mlir::RewritePatternSet decomposePatterns(context);
    decomposePatterns.add<ConvertDotConvert>(context);
    if (mlir::applyPatternsAndFoldGreedily(m, std::move(decomposePatterns))
            .failed()) {
      signalPassFailure();
    }

    // 4. Apply clean up patterns to remove remove dead convert and dead code
    // generated by the previous transformations.
    mlir::RewritePatternSet cleanUpPatterns2(context);
    populateForOpDeadArgumentElimination(cleanUpPatterns2);
    scf::ForOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns2, context);
    if (mlir::applyPatternsAndFoldGreedily(m, std::move(cleanUpPatterns2))
            .failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPURemoveLayoutConversionsPass() {
  return std::make_unique<TritonAMDGPURemoveLayoutConversionsPass>();
}
