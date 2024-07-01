#ifndef TRITON_ANALYSIS_ALLOCATION_H
#define TRITON_ANALYSIS_ALLOCATION_H

#include "triton/Analysis/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <atomic>
#include <limits>

namespace mlir {

namespace triton {
class AllocationAnalysis;

SmallVector<unsigned>
getScratchConfigForCvtLayout(triton::gpu::ConvertLayoutOp op, unsigned &inVec,
                             unsigned &outVec);
SmallVector<unsigned> getScratchConfigForCvtLayout(RankedTensorType srcType,
                                                   RankedTensorType dstType,
                                                   unsigned &inVec,
                                                   unsigned &outVec);
SmallVector<unsigned> getRepShapeForCvtLayout(triton::gpu::ConvertLayoutOp op);
SmallVector<unsigned> getRepShapeForCvtLayout(RankedTensorType srcTy,
                                              RankedTensorType dstTy);

} // namespace triton

/// Modified from llvm-15.0: llvm/ADT/AddressRanges.h
/// A class that represents an interval, specified using a start and an end
/// values: [Start, End).
template <typename T> class Interval {
public:
  Interval() {}
  Interval(T S, T E) : Start(S), End(E) { assert(Start <= End); }
  T start() const { return Start; }
  T end() const { return End; }
  T size() const { return End - Start; }
  bool contains(T Addr) const { return Start <= Addr && Addr < End; }
  bool intersects(const Interval &R) const {
    return Start < R.End && R.Start < End;
  }
  bool operator==(const Interval &R) const {
    return Start == R.Start && End == R.End;
  }
  bool operator!=(const Interval &R) const { return !(*this == R); }
  bool operator<(const Interval &R) const {
    return std::make_pair(Start, End) < std::make_pair(R.Start, R.End);
  }

private:
  T Start = std::numeric_limits<T>::min();
  T End = std::numeric_limits<T>::max();
};

template <class T> Interval(T, T) -> Interval<T>;

class Allocation {
public:
  /// A unique identifier for shared memory buffers
  using BufferId = size_t;
  using BufferIdSetT = DenseSet<BufferId>;
  using FuncAllocMapT = CallGraph<Allocation>::FuncDataMapT;

  static constexpr BufferId InvalidBufferId =
      std::numeric_limits<BufferId>::max();

  Allocation() = default;
  /// Creates a new Allocation analysis that computes the shared memory
  /// information for all associated shared memory values.
  explicit Allocation(Operation *operation) : operation(operation) {}

  /// Runs allocation analysis on the given top-level operation.
  void run(FuncAllocMapT &funcAllocMap);

  /// Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return operation; }

  /// Returns the offset of the given buffer in the shared memory.
  size_t getOffset(BufferId bufferId) const {
    return bufferSet.at(bufferId).offset;
  }

  /// Returns the size of the given buffer in the shared memory.
  size_t getAllocatedSize(BufferId bufferId) const {
    return bufferSet.at(bufferId).size;
  }

  /// Returns the allocated interval of the given buffer.
  Interval<size_t> getAllocatedInterval(BufferId bufferId) const {
    auto &buffer = bufferSet.at(bufferId);
    return Interval<size_t>(buffer.offset, buffer.offset + buffer.size);
  }

  /// Returns the buffer id of the given value.
  /// This interface only returns the allocated buffer id.
  /// If you want to get all the buffer ids that are associated with the given
  /// value, including alias buffers, use getBufferIds.
  BufferId getBufferId(Value value) const {
    if (valueBuffer.count(value)) {
      return valueBuffer.lookup(value)->id;
    } else {
      return InvalidBufferId;
    }
  }

  /// Returns all the buffer ids of the given value, including alias buffers.
  BufferIdSetT getBufferIds(Value value) const {
    BufferIdSetT bufferIds;
    auto allocBufferId = getBufferId(value);
    if (allocBufferId != InvalidBufferId)
      bufferIds.insert(allocBufferId);
    for (auto *buffer : aliasBuffer.lookup(value)) {
      if (buffer->id != InvalidBufferId)
        bufferIds.insert(buffer->id);
    }
    return bufferIds;
  }

  /// Returns the scratch buffer id of the given value.
  BufferId getBufferId(Operation *operation) const {
    if (opScratch.count(operation)) {
      return opScratch.lookup(operation)->id;
    } else if (opVirtual.count(operation)) {
      return opVirtual.lookup(operation)->id;
    } else {
      return InvalidBufferId;
    }
  }

  /// Returns if the given buffer is a virtual buffer.
  bool isVirtualBuffer(BufferId bufferId) const {
    return bufferSet.at(bufferId).kind == BufferT::BufferKind::Virtual;
  }

  /// Returns the size of total shared memory allocated
  size_t getSharedMemorySize() const { return sharedMemorySize; }

  /// Returns mapping from operation to list of live LDS buffers
  std::map<Operation *, SmallVector<BufferId>> getLiveBuffers();

private:
  /// A class that represents a shared memory buffer
  struct BufferT {
    /// Explicit: triton_gpu.local_alloc
    /// Scratch: triton_gpu.convert_layout
    /// Virtual: triton.call
    enum class BufferKind { Explicit, Scratch, Virtual };

    /// MT: thread-safe
    inline static std::atomic<BufferId> nextId = 0;

    BufferKind kind;
    BufferId id;
    size_t size;
    size_t alignment;
    size_t offset;

    bool operator==(const BufferT &other) const { return id == other.id; }
    bool operator<(const BufferT &other) const { return id < other.id; }

    BufferT() : BufferT(BufferKind::Explicit, 0) {}
    BufferT(BufferKind kind, size_t size, size_t alignment = 4,
            size_t offset = 0)
        : kind(kind), id(nextId++), size(size), alignment(alignment),
          offset(offset) {}

    size_t setOffsetAligned(size_t newOffset) {
      return offset = llvm::alignTo(newOffset, alignment);
    }
  };

  /// Op -> Scratch Buffer
  using OpScratchMapT = DenseMap<Operation *, BufferT *>;
  /// Value -> Explicit Buffer
  using ValueBufferMapT = llvm::MapVector<Value, BufferT *>;
  /// Value -> Alias Buffer
  using AliasBufferMapT = llvm::MapVector<Value, llvm::SetVector<BufferT *>>;
  /// BufferId -> Buffer
  using BufferSetT = std::map<BufferId, BufferT>;

private:
  template <BufferT::BufferKind Kind, typename KeyType, typename... Args>
  void addBuffer(KeyType &key, Args &&...args) {
    auto buffer = BufferT(Kind, std::forward<Args>(args)...);
    bufferSet[buffer.id] = std::move(buffer);
    if constexpr (Kind == BufferT::BufferKind::Explicit) {
      valueBuffer[key] = &bufferSet[buffer.id];
    } else if constexpr (Kind == BufferT::BufferKind::Virtual) {
      opVirtual[key] = &bufferSet[buffer.id];
    } else {
      opScratch[key] = &bufferSet[buffer.id];
    }
  }

  void addAlias(Value value, Value alloc) {
    aliasBuffer[value].insert(valueBuffer[alloc]);
  }

private:
  Operation *operation = nullptr;
  OpScratchMapT opScratch;
  OpScratchMapT opVirtual;
  ValueBufferMapT valueBuffer;
  AliasBufferMapT aliasBuffer;
  BufferSetT bufferSet;
  size_t sharedMemorySize = 0;

  friend class triton::AllocationAnalysis;
};

/// Static analysis that computes the allocation of shared memory buffers
/// of the entire call graph.
/// The allocation is performed in a post-order walk of the call graph.
/// Each call op is treated like convert_layout that allocates a scratch buffer.
/// At each call, we compute the start offset of the scratch buffer and pass it
/// as an argument to the callee.
class ModuleAllocation : public CallGraph<Allocation> {
public:
  using FuncOffsetMapT = DenseMap<FunctionOpInterface, Value>;

  explicit ModuleAllocation(ModuleOp moduleOp)
      : CallGraph<Allocation>(moduleOp) {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order edge walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order node walk callback
        [&](FunctionOpInterface funcOp) {
          auto [iter, inserted] = funcMap.try_emplace(funcOp, funcOp);
          if (inserted)
            iter->second.run(funcMap);
        });
  }

  size_t getSharedMemorySize() {
    size_t size = 0;
    for (auto funcOp : getRoots()) {
      auto *alloc = getFuncData(funcOp);
      size = std::max(size, alloc->getSharedMemorySize());
    }
    return size;
  }

  size_t getSharedMemorySize(FunctionOpInterface funcOp) {
    return getFuncData(funcOp)->getSharedMemorySize();
  }

  void setFunctionSharedMemoryValue(FunctionOpInterface funcOp, Value value) {
    sharedMemoryValue[funcOp] = value;
  }

  Value getFunctionSharedMemoryBase(FunctionOpInterface funcOp) {
    return sharedMemoryValue[funcOp];
  }

private:
  FuncOffsetMapT sharedMemoryValue;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_ALLOCATION_H
