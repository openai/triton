#ifndef TRITON_ANALYSIS_UTILITY_H
#define TRITON_ANALYSIS_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>
#include <string>

namespace mlir {

class ReduceOpHelper {
public:
  explicit ReduceOpHelper(triton::ReduceOp rop)
      : op(rop.getOperation()), axis(rop.getAxis()) {
    auto firstTy = rop.getOperands()[0].getType().cast<RankedTensorType>();
    srcShape = firstTy.getShape();
    srcEncoding = firstTy.getEncoding();
    srcElementTypes = rop.getElementTypes();

    for (const auto &t : rop.getInputTypes()) {
      if (t.getShape() != srcShape) {
        rop.emitError() << "shape mismatch";
      }
      if (t.getEncoding() != srcEncoding) {
        rop.emitError() << "encoding mismatch";
      }
    }
  }

  ArrayRef<int64_t> getSrcShape() { return srcShape; }

  Attribute getSrcLayout() { return srcEncoding; }

  bool isFastReduction();

  unsigned getInterWarpSize();

  unsigned getIntraWarpSize();

  unsigned getThreadsReductionAxis();

  SmallVector<unsigned> getScratchConfigBasic();

  SmallVector<SmallVector<unsigned>> getScratchConfigsFast();

  unsigned getScratchSizeInBytes();

  bool isSupportedLayout();

private:
  Operation *op;
  ArrayRef<int64_t> srcShape;
  Attribute srcEncoding;
  SmallVector<Type> srcElementTypes;
  int axis;
};

bool isSharedEncoding(Value value);

bool maybeSharedAllocationOp(Operation *op);

bool maybeAliasOp(Operation *op);

bool supportMMA(triton::DotOp op, int version);

bool supportMMA(Value value, int version);

Type getElementType(Value value);

std::string getValueOperandName(Value value, AsmState &state);

template <typename T_OUT, typename T_IN>
inline SmallVector<T_OUT> convertType(ArrayRef<T_IN> in) {
  SmallVector<T_OUT> out;
  for (const T_IN &i : in)
    out.push_back(T_OUT(i));
  return out;
}

template <typename Int> Int product(llvm::ArrayRef<Int> arr) {
  return std::accumulate(arr.begin(), arr.end(), 1, std::multiplies{});
}

template <typename Int> Int ceil(Int m, Int n) { return (m + n - 1) / n; }

// output[i] = input[order[i]]
template <typename T, typename RES_T = T>
SmallVector<RES_T> reorder(ArrayRef<T> input, ArrayRef<unsigned> order) {
  size_t rank = order.size();
  assert(input.size() == rank);
  SmallVector<RES_T> result(rank);
  for (auto it : llvm::enumerate(order)) {
    result[it.index()] = input[it.value()];
  }
  return result;
}

template <typename T> T highestPowOf2Divisor(T n) {
  if (n == 0) {
    return (static_cast<T>(1) << (sizeof(T) * 8 - 2));
  }
  return (n & (~(n - 1)));
}

bool isSingleValue(Value value);

bool isMmaToDotShortcut(RankedTensorType &srcTy, RankedTensorType &dstTy);

/// Multi-root DAG topological sort.
/// Performs a topological sort of the Operation in the `toSort` SetVector.
/// Returns a topologically sorted SetVector.
/// It is faster than mlir::topologicalSort because it prunes nodes that have
/// been visited before.
SetVector<Operation *>
multiRootTopologicalSort(const SetVector<Operation *> &toSort);

/// This uses the toplogicalSort above
SetVector<Operation *>
multiRootGetSlice(Operation *op, TransitiveFilter backwardFilter = nullptr,
                  TransitiveFilter forwardFilter = nullptr);

/// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();

/// This class represents a call graph for a given ModuleOp and holds
/// data of type T associated with each FunctionOpInterface.
template <typename T> class CallGraph {
public:
  using FuncDataMapT = DenseMap<FunctionOpInterface, T>;

  /// Constructor that builds the call graph for the given moduleOp.
  CallGraph(ModuleOp moduleOp) : moduleOp(moduleOp) { build(); }

  /// Walks the call graph and applies the provided update functions
  /// to the edges and nodes.
  template <WalkOrder UpdateEdgeOrder = WalkOrder::PreOrder,
            WalkOrder UpdateNodeOrder = WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void walk(UpdateEdgeFn updateEdgeFn, UpdateNodeFn updateNodeFn) {
    DenseSet<FunctionOpInterface> visited;
    for (auto root : roots) {
      doWalk<UpdateEdgeOrder, UpdateNodeOrder>(root, visited, updateEdgeFn,
                                               updateNodeFn);
    }
  }

  /// Retrieves the data associated with a function
  T *getFuncData(FunctionOpInterface funcOp) {
    if (funcMap.count(funcOp)) {
      return &funcMap[funcOp];
    }
    return nullptr;
  }

  /// Getters for moduleOp and roots.
  ModuleOp getModuleOp() const { return moduleOp; }
  SmallVector<FunctionOpInterface> getRoots() const { return roots; }

  /// Returns true if the given function is a root.
  bool isRoot(FunctionOpInterface funcOp) const {
    return llvm::is_contained(roots, funcOp);
  }

  /// Maps the data associated with a FunctionOpInterface to a targetFuncOp.
  template <typename FROM, typename TO>
  void mapFuncOp(FROM funcOp, TO targetFuncOp) {
    // Iterate over graph and replace
    for (auto &kv : graph) {
      for (auto &edge : kv.second) {
        if (edge.second == funcOp) {
          edge.second = targetFuncOp;
        }
      }
    }
    graph[targetFuncOp] = graph[funcOp];
    // Replace in roots
    for (auto it = roots.begin(); it != roots.end(); ++it) {
      if (*it == funcOp) {
        *it = targetFuncOp;
        break;
      }
    }
    // Replace in funcMap
    funcMap[targetFuncOp] = funcMap[funcOp];
  }

private:
  void build() {
    SymbolTableCollection symbolTable;
    DenseMap<Operation *, Operation *> parentMap;
    moduleOp.walk([&](Operation *op) {
      auto parent = op->getParentOfType<FunctionOpInterface>();
      if (auto callOp = dyn_cast<CallOpInterface>(op)) {
        auto *callee = callOp.resolveCallable(&symbolTable);
        auto funcOp = dyn_cast_or_null<FunctionOpInterface>(callee);
        if (funcOp)
          graph[parent].emplace_back(
              std::pair<CallOpInterface, FunctionOpInterface>(callOp, funcOp));
      }
      parentMap[op] = parent;
      if (parent == nullptr && isa<FunctionOpInterface>(op))
        roots.push_back(dyn_cast<FunctionOpInterface>(op));
    });
  }

  template <WalkOrder UpdateEdgeOrder = WalkOrder::PreOrder,
            WalkOrder UpdateNodeOrder = WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void doWalk(FunctionOpInterface funcOp,
              DenseSet<FunctionOpInterface> &visited, UpdateEdgeFn updateEdgeFn,
              UpdateNodeFn updateNodeFn) {
    if (visited.count(funcOp)) {
      llvm::report_fatal_error("Cycle detected in call graph");
    }
    if constexpr (UpdateNodeOrder == WalkOrder::PreOrder) {
      updateNodeFn(funcOp);
    }
    for (auto [callOp, callee] : graph[funcOp]) {
      if constexpr (UpdateEdgeOrder == WalkOrder::PreOrder) {
        updateEdgeFn(callOp, callee);
      }
      doWalk(callee, visited, updateEdgeFn, updateNodeFn);
      if constexpr (UpdateEdgeOrder == WalkOrder::PostOrder) {
        updateEdgeFn(callOp, callee);
      }
    }
    if constexpr (UpdateNodeOrder == WalkOrder::PostOrder) {
      updateNodeFn(funcOp);
    }
    visited.erase(funcOp);
  }

protected:
  ModuleOp moduleOp;
  DenseMap<FunctionOpInterface,
           SmallVector<std::pair<CallOpInterface, FunctionOpInterface>>>
      graph;
  FuncDataMapT funcMap;
  SmallVector<FunctionOpInterface> roots;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_UTILITY_H
