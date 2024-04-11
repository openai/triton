#include "triton/Analysis/Utility.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <deque>

namespace mlir {
namespace {

using namespace triton;
using namespace triton::gpu;

int getParentAxis(Attribute layout, int axis) {
  if (auto sliceEncoding = layout.dyn_cast<SliceEncodingAttr>()) {
    axis = axis < sliceEncoding.getDim() ? axis : axis + 1;
    return getParentAxis(sliceEncoding.getParent(), axis);
  }
  return axis;
}

SmallVector<unsigned> getParentOrder(Attribute layout) {
  if (auto sliceEncoding = layout.dyn_cast<SliceEncodingAttr>()) {
    return getParentOrder(sliceEncoding.getParent());
  }
  return getOrder(layout);
}

} // namespace

// TODO(jlebar): Move this class into namespace triton.
bool ReduceOpHelper::isReductionOnLayoutFastAxis() {
  return getParentAxis(getSrcLayout(), axis) ==
         getParentOrder(getSrcLayout())[0];
}

SmallVector<unsigned> ReduceOpHelper::getOrderWithAxisAtBeginning() {
  auto srcLayout = getSrcLayout();
  auto order = getOrder(srcLayout);
  auto it = std::find(order.begin(), order.end(), axis);
  // delete the axis from order
  order.erase(it);
  // insert axis at the beginning of order
  order.insert(order.begin(), axis);
  return order;
}

// Thread offset is the thread index offset of two adjacent threads on the
// reduction axis within the warp.
unsigned ReduceOpHelper::getThreadOffsetOnReductionAxis() {
  auto srcLayout = getSrcLayout();

  // If the reduction axis is the fast axis of the parent layout
  if (isReductionOnLayoutFastAxis()) {
    return 1;
  }

  unsigned threadOffset = 1;
  if (auto sliceLayout = srcLayout.dyn_cast<SliceEncodingAttr>()) {
    auto parentLayout = sliceLayout.getParent();
    auto threadsPerWarp = getThreadsPerWarp(parentLayout);
    threadOffset = threadsPerWarp[sliceLayout.getDim()];
  } else {
    auto threadsPerWarp = getThreadsPerWarp(srcLayout);
    auto order = getOrder(srcLayout);
    for (unsigned i = 0; i < order.size(); i++) {
      if (order[i] == axis)
        break;
      threadOffset *= threadsPerWarp[order[i]];
    }
  }
  return threadOffset;
}

// Cases where distributed shared memory is not required in ConvertLayout:
// (1) numCTAs == 1
// (2) numCTAs > 1 but srcCTALayout == dstCTALayout
// TODO: Case with SliceLayout as srcLayout and numCTAs > 1 is to be implemented
// in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout) {
  unsigned numCTAs = getNumCTAs(srcLayout);
  assert(numCTAs == getNumCTAs(dstLayout) &&
         "Invalid layout conversion: the numbers of CTAs of src and dst "
         "layouts are different");

  // Case (1): Never use dsmem when numCTAs == 1
  if (numCTAs == 1)
    return false;

  // Case where CTAsPerCGA of srcLayout in the sliced dim is not 1 is not
  // implemented yet
  if (auto sliceLayout = srcLayout.dyn_cast<SliceEncodingAttr>()) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] != 1)
      llvm::report_fatal_error("Layout conversion to be implemented");
  }

  // Case where CTAsPerCGA of dstLayout in the sliced dim is not 1 is supported
  if (auto sliceLayout = dstLayout.dyn_cast<SliceEncodingAttr>()) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] != 1)
      return true;
  }

  // The above two branches make sure that it is legal to call getCTALayout of
  // srcLayout and dstLayout

  // Case (2): Do not use dsmem when srcCTALayout == dstCTALayout
  auto srcCTALayout = getCTALayout(srcLayout);
  auto dstCTALayout = getCTALayout(dstLayout);
  if (srcCTALayout == dstCTALayout)
    return false;

  // Dsmem access is required when srcCTALayout != dstCTALayout
  return true;
}

unsigned ReduceOpHelper::getInterWarpSize() {
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  unsigned sizeIntraWarps = getIntraWarpSize();
  return std::min(srcReduceDimSize / sizeIntraWarps,
                  getWarpsPerCTA(getSrcLayout())[axis]);
}

unsigned ReduceOpHelper::getIntraWarpSize() {
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  return std::min(srcReduceDimSize, getThreadsPerWarp(getSrcLayout())[axis]);
}

unsigned ReduceOpHelper::getInterWarpSizeWithUniqueData() {
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  unsigned sizeIntraWarps = getIntraWarpSizeWithUniqueData();
  return std::min(
      srcReduceDimSize / sizeIntraWarps,
      getWarpsPerCTAWithUniqueData(getSrcLayout(), getSrcShape())[axis]);
}

unsigned ReduceOpHelper::getIntraWarpSizeWithUniqueData() {
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  unsigned elementPerThreads =
      getUniqueContigPerThread(getSrcLayout(), getSrcShape())[axis];
  return std::min(
      srcReduceDimSize / elementPerThreads,
      getThreadsPerWarpWithUniqueData(getSrcLayout(), getSrcShape())[axis]);
}

unsigned ReduceOpHelper::getThreadsReductionAxis() {
  auto srcLayout = getSrcLayout();
  auto srcShape = getSrcShape();
  return getThreadsPerWarpWithUniqueData(srcLayout, srcShape)[axis] *
         getWarpsPerCTAWithUniqueData(srcLayout, srcShape)[axis];
}

bool ReduceOpHelper::isWarpSynchronous() {
  auto srcLayout = getSrcLayout();
  auto srcShape = getSrcShape();
  return getWarpsPerCTAWithUniqueData(srcLayout, srcShape)[axis] == 1;
}

SmallVector<unsigned> ReduceOpHelper::getScratchConfig() {
  SmallVector<unsigned> smemShape;
  // that case doesn't need inter-warp communication
  if (isWarpSynchronous())
    return {0, 0};

  smemShape = convertType<unsigned>(getSrcShape());
  smemShape[axis] = getInterWarpSizeWithUniqueData();

  return smemShape;
}

unsigned ReduceOpHelper::getScratchSizeInBytes() {
  auto smemShape = getScratchConfig();
  auto elems = product<unsigned>(smemShape);

  unsigned bytesPerElem = 0;
  for (const auto &ty : srcElementTypes) {
    bytesPerElem += ceil<unsigned>(ty.getIntOrFloatBitWidth(), 8);
  }
  return bytesPerElem * elems;
}

bool ReduceOpHelper::isReduceWithinCTA() {
  auto axis = getAxis();
  auto srcLayout = getSrcLayout();
  auto CTASplitNum = getCTASplitNum(srcLayout);
  assert(axis < CTASplitNum.size());
  return CTASplitNum[axis] == 1;
}

bool ReduceOpHelper::isSupportedLayout() {
  // Layout optimization passes such as PlanCTAPass and
  // RemoveLayoutConversionPass should avoid cross-CTA reduction
  if (!isReduceWithinCTA()) {
    return false;
  }

  auto srcLayout = getSrcLayout();
  if (srcLayout.isa<BlockedEncodingAttr>()) {
    return true;
  }
  if (auto mmaLayout = srcLayout.dyn_cast<MmaEncodingTrait>()) {
    return mmaLayout.supportReduction();
  }
  if (auto sliceLayout = srcLayout.dyn_cast<SliceEncodingAttr>()) {
    return true;
  }
  return false;
}

unsigned ScanLoweringHelper::getAxisNumElementsPerThread() {
  return getEncoding().getSizePerThread()[getAxis()];
}

unsigned ScanLoweringHelper::getNonAxisNumElementsPerThread() {
  SmallVector<unsigned> sizePerThreads = getContigPerThread(getEncoding());
  sizePerThreads[getAxis()] = 1;
  return product<unsigned>(sizePerThreads);
}

Region &ScanLoweringHelper::getCombineOp() { return scanOp.getCombineOp(); }

unsigned ScanLoweringHelper::getAxisNumThreadsPerWarp() {
  return getThreadsPerWarp(getEncoding())[getAxis()];
}

unsigned ScanLoweringHelper::getAxisNumThreadsPerWarpWithUniqueData() {
  return getThreadsPerWarpWithUniqueData(getEncoding(), getShape())[getAxis()];
}

unsigned ScanLoweringHelper::getNonAxisNumThreadsPerWarp() {
  auto threadsPerWarp = getThreadsPerWarp(getEncoding());
  threadsPerWarp[getAxis()] = 1;
  return product<unsigned>(threadsPerWarp);
}

// Return the flat numbers of threads computing independent scan results.
unsigned ScanLoweringHelper::getNonAxisNumThreadsPerCTA() {
  unsigned numParallelThreadsPerWarp = getNonAxisNumThreadsPerWarp();
  auto warpsPerCTA = getWarpsPerCTA(getEncoding());
  warpsPerCTA[getAxis()] = 1;
  unsigned numParallelWarpsPerCTA = product<unsigned>(warpsPerCTA);
  return numParallelThreadsPerWarp * numParallelWarpsPerCTA;
}

unsigned ScanLoweringHelper::getAxisNumWarps() {
  return getWarpsPerCTA(getEncoding())[getAxis()];
}

unsigned ScanLoweringHelper::getAxisNumWarpsWithUniqueData() {
  return getWarpsPerCTAWithUniqueData(getEncoding(), getShape())[getAxis()];
}

unsigned ScanLoweringHelper::getAxisNumBlocks() {
  auto sizePerThreads = getSizePerThread(getEncoding());
  auto threadsPerWarp = getThreadsPerWarp(getEncoding());
  auto warpsPerCTA = getWarpsPerCTA(getEncoding());
  unsigned axis = getAxis();
  return ceil<unsigned>(
      getShape()[axis],
      (sizePerThreads[axis] * threadsPerWarp[axis] * warpsPerCTA[axis]));
}

unsigned ScanLoweringHelper::getNonAxisNumBlocks() {
  auto sizePerThreads = getSizePerThread(getEncoding());
  auto threadsPerWarp = getThreadsPerWarp(getEncoding());
  auto warpsPerCTA = getWarpsPerCTA(getEncoding());
  unsigned axis = getAxis();
  unsigned numBlocks = 1;
  for (unsigned i = 0; i < sizePerThreads.size(); i++) {
    if (i == axis)
      continue;
    numBlocks *=
        ceil<unsigned>(getShape()[i], (sizePerThreads[i] * threadsPerWarp[i] *
                                       warpsPerCTA[i]));
  }
  return numBlocks;
}

bool ScanLoweringHelper::isSupported() {
  // TODO: Support the following cases:
  // 1. Scan on non-blocking encodings
  if (!isa<BlockedEncodingAttr>(getEncoding()))
    return false;
  return true;
}

unsigned ScanLoweringHelper::getScratchSizeInElems() {
  auto mod = scanOp->getParentOfType<ModuleOp>();
  unsigned numWarps = TritonGPUDialect::getNumWarps(mod);
  unsigned numNonAxisElementsPerWarp =
      getNonAxisNumThreadsPerWarp() * getNonAxisNumElementsPerThread();
  unsigned numElements = numWarps * numNonAxisElementsPerWarp *
                         getAxisNumBlocks() * getNonAxisNumBlocks();
  return numElements;
}

unsigned ScanLoweringHelper::getScratchSizeInBytes() {
  unsigned axisNumWarps = getAxisNumWarpsWithUniqueData();
  if (axisNumWarps == 1)
    return 0;
  unsigned elementSizeInBytes = 0;
  for (const auto &ty : srcElementTypes) {
    elementSizeInBytes += ceil<unsigned>(ty.getIntOrFloatBitWidth(), 8);
  }
  return elementSizeInBytes * getScratchSizeInElems();
}

SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getReshapeDecomposition(ArrayRef<int64_t> srcShape,
                        ArrayRef<int64_t> dstShape) {
  SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> ret;

  if (srcShape.empty()) {
    assert(dstShape.empty());
    return ret;
  }
  ret.push_back({});

  int srcIdx = 0;
  int dstIdx = 0;
  int srcNElems = 1;
  int dstNElems = 1;
  while (srcIdx < srcShape.size() || dstIdx < dstShape.size()) {
    if (srcNElems < dstNElems || //
        (srcIdx < srcShape.size() && srcNElems == 1) ||
        (srcIdx < srcShape.size() && srcShape[srcIdx] == 1)) {
      assert(srcIdx < srcShape.size());
      srcNElems *= srcShape[srcIdx];
      ret.back().first.push_back(srcIdx);
      srcIdx++;
    } else if (dstNElems < srcNElems ||
               (dstIdx < dstShape.size() && dstShape[dstIdx] == 1)) {
      assert(dstIdx < dstShape.size());
      dstNElems *= dstShape[dstIdx];
      ret.back().second.push_back(dstIdx);
      dstIdx++;
    } else {
      ret.push_back({});
      srcNElems = 1;
      dstNElems = 1;
    }
  }
  return ret;
}

BlockedEncodingAttr ScanLoweringHelper::getEncoding() {
  return srcEncoding.cast<BlockedEncodingAttr>();
}

unsigned ScanLoweringHelper::getAxisElementStride() {
  auto order = getOrder(getEncoding());
  unsigned stride = 1;
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= getContigPerThread(getEncoding())[dim];
  }
  llvm_unreachable("Axis not found in order");
}

unsigned ScanLoweringHelper::getAxisThreadStride() {
  auto order = getOrder(getEncoding());
  unsigned stride = 1;
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= getEncoding().getThreadsPerWarp()[dim];
  }
  llvm_unreachable("Axis not found in order");
}

unsigned ScanLoweringHelper::getAxisBlockStride() {
  auto order = getOrder(getEncoding());
  unsigned stride = 1;
  auto sizePerThreads = getSizePerThread(getEncoding());
  auto threadsPerWarp = getThreadsPerWarp(getEncoding());
  auto warpsPerCTA = getWarpsPerCTA(getEncoding());
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= ceil<unsigned int>(getShape()[dim], sizePerThreads[dim] *
                                                      threadsPerWarp[dim] *
                                                      warpsPerCTA[dim]);
  }
  llvm_unreachable("Axis not found in order");
}

bool maybeSharedAllocationOp(Operation *op) {
  // TODO(Keren): This function can be replaced by adding
  // MemoryEffectOpInterface. We can then use the MemoryEffectOpInterface to
  // query the memory effects of the op.
  auto *dialect = op->getDialect();
  return dialect &&
         (dialect->getTypeID() == TypeID::get<TritonGPUDialect>() ||
          dialect->getTypeID() ==
              TypeID::get<triton::nvidia_gpu::TritonNvidiaGPUDialect>() ||
          dialect->getTypeID() == TypeID::get<triton::TritonDialect>() ||
          dialect->getTypeID() == TypeID::get<arith::ArithDialect>() ||
          dialect->getTypeID() == TypeID::get<tensor::TensorDialect>());
}

static bool supportMFMAGranularity(int m, int n, int k) {
  // these limitations are dtype dependent, in future we may relax them
  const static std::pair<int, int> mfmaTypes[2] = {{32, 8}, {16, 16}};
  for (const auto &mfmaType : mfmaTypes) {
    auto [granularityMN, granularityK] = mfmaType;
    if (m % granularityMN != 0 || n % granularityMN != 0)
      continue;
    if (k % granularityK != 0)
      continue;
    return true;
  }
  return false;
}

bool supportMFMATypes(Type a, Type b) {
  if (a.getIntOrFloatBitWidth() != b.getIntOrFloatBitWidth())
    return false;

  auto F8E4M3FNUZ = TypeID::get<Float8E4M3FNUZType>();
  auto F8E5M2FNUZ = TypeID::get<Float8E5M2FNUZType>();
  auto F16 = TypeID::get<Float16Type>();
  auto BF16 = TypeID::get<BFloat16Type>();
  auto F32 = TypeID::get<Float32Type>();
  auto Int = TypeID::get<IntegerType>();
  DenseSet<std::pair<TypeID, TypeID>> supportedTypes = {
      {F32, F32},
      {F16, F16},
      {BF16, BF16},
      {F8E4M3FNUZ, F8E4M3FNUZ},
      {F8E4M3FNUZ, F8E5M2FNUZ},
      {F8E5M2FNUZ, F8E4M3FNUZ},
      {F8E5M2FNUZ, F8E5M2FNUZ},
      {Int, Int}};

  if (!supportedTypes.contains({a.getTypeID(), b.getTypeID()}))
    return false;

  if (a.isIntOrIndex() && a.getIntOrFloatBitWidth() != 8)
    return false;
  return true;
}

bool supportMFMA(triton::DotOp op) {
  auto aTy = op.getA().getType().cast<RankedTensorType>();
  auto bTy = op.getB().getType().cast<RankedTensorType>();

  auto aElemTy = aTy.getElementType();
  auto bElemTy = bTy.getElementType();

  if (!supportMFMATypes(aElemTy, bElemTy))
    return false;

  auto aShape = aTy.getShape();
  auto bShape = bTy.getShape();

  auto rank = aShape.size();
  auto M = aShape[rank - 2];
  auto N = bShape[rank - 1];
  auto K = aShape[rank - 1];
  assert(K == bShape[rank - 2]);
  if (!supportMFMAGranularity(M, N, K))
    return false;

  return true;
}

static bool supportWMMAGranularity(int m, int n, int k) {
  return m % 16 == 0 && n % 16 == 0 && k % 16 == 0;
}

static bool supportWMMATypes(Type a, Type b, Type c, Type d) {
  if (a != b || c != d)
    return false;
  if (a.isIntOrIndex()) {
    if (!c.isIntOrIndex())
      return false;
    auto aWidth = a.getIntOrFloatBitWidth();
    auto cWidth = c.getIntOrFloatBitWidth();
    bool aValid = a.isUnsignedInteger() && aWidth <= 8;
    bool cValid = cWidth <= 32;
    return aValid && cValid;
  } else if (a.isa<FloatType>()) {
    if (a.isBF16())
      return c.isBF16() || c.isF32();
    if (a.isF16())
      return c.isF16() || c.isF32();
  }
  return false;
}

bool supportWMMA(triton::DotOp op) {
  auto aTy = op.getA().getType().cast<RankedTensorType>();
  auto bTy = op.getB().getType().cast<RankedTensorType>();
  auto cTy = op.getC().getType().cast<RankedTensorType>();
  auto dTy = op.getResult().getType().cast<RankedTensorType>();

  auto aElemTy = aTy.getElementType();
  auto bElemTy = bTy.getElementType();
  auto cElemTy = cTy.getElementType();
  auto dElemTy = dTy.getElementType();

  if (!supportWMMATypes(aElemTy, bElemTy, cElemTy, dElemTy))
    return false;

  auto aShape = aTy.getShape();
  auto bShape = bTy.getShape();

  assert(aShape[1] == bShape[0]);
  if (!supportWMMAGranularity(aShape[0], bShape[1], aShape[1]))
    return false;

  return true;
}

bool supportMMA(triton::DotOp op, int version) {
  // Refer to mma section for the data type supported by Volta and Hopper
  // Tensor Core in
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16
  auto aElemTy = op.getA().getType().getElementType();
  auto bElemTy = op.getB().getType().getElementType();
  if (version == 3) {
    if (triton::tools::getBoolEnv("DISABLE_MMA_V3"))
      return false;
    auto retType = op.getType();
    auto retShapePerCTA = getShapePerCTA(retType);
    auto rank = retShapePerCTA.size();
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = TritonGPUDialect::getNumWarps(mod);
    if (!(numWarps % 4 == 0 && retShapePerCTA[rank - 2] % 64 == 0 &&
          retShapePerCTA[rank - 1] % 8 == 0 &&
          (aElemTy.isFloat8E5M2() || aElemTy.isFloat8E4M3FNUZ() ||
           aElemTy.isInteger(8) || aElemTy.isF16() || aElemTy.isBF16() ||
           aElemTy.isF32()))) {
      return false;
    }
    // We cannot use MMA_V3 if we need to accumulate in F32 within the MMA op.
    if (op.getMaxNumImpreciseAcc() < 32 &&
        (aElemTy.isFloat8E5M2() || aElemTy.isFloat8E4M3FNUZ()) &&
        op.getType().cast<RankedTensorType>().getElementType().isF32()) {
      return false;
    }
  }
  if (aElemTy.isF32() && bElemTy.isF32()) {
    return op.getInputPrecision() == InputPrecision::TF32 && version >= 2;
  }
  return supportMMA(op.getA(), version) && supportMMA(op.getB(), version);
}

bool supportMMA(Value value, int version) {
  // Tell whether a DotOp support MMA by the operand type(either $a or $b).
  // We cannot get both the operand types(in TypeConverter), here we assume the
  // types of both the operands are identical here.
  assert((version == 1 || version == 2 || version == 3) &&
         "Unexpected MMA layout version found");
  auto elemTy = value.getType().cast<TensorOrMemDesc>().getElementType();
  // FP8 is not natively supported on all mma versions but it can always be
  // promoted to fp16 therefore we can always support it.
  bool isFP8 = elemTy.isFloat8E5M2() || elemTy.isFloat8E4M3FN() ||
               elemTy.isFloat8E5M2FNUZ() || elemTy.isFloat8E4M3FNUZ();
  return isFP8 || elemTy.isF16() || elemTy.isBF16() ||
         (elemTy.isF32() && version >= 2) ||
         (elemTy.isInteger(8) && version >= 2);
}

bool isMfmaToDotShortcut(RankedTensorType &srcTy, RankedTensorType &dstTy) {
  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  auto mfmaLayout = srcLayout.cast<AMDMfmaEncodingAttr>();
  auto dotOperandLayout = dstLayout.cast<DotOperandEncodingAttr>();
  // TODO: Remove the restriction on the warpsPerCTA once chain dot testing is
  // improved. In addition, we can enable this shortcut for regular MFMA
  // layout when opIdx == 1.
  return mfmaLayout.getWarpsPerCTA()[1] == 1 &&
         dotOperandLayout.getOpIdx() == 0 && mfmaLayout.getIsTransposed() &&
         dotOperandLayout.getKWidth() == getContigPerThread(mfmaLayout)[1] &&
         dotOperandLayout.getParent() == mfmaLayout &&
         (mfmaLayout.getMDim() == 32 || mfmaLayout.getMDim() == 16) &&
         (srcTy.getElementType().isF16() || srcTy.getElementType().isBF16());
}

static bool isMmaToMmaShortcut(Attribute srcEncoding, Attribute dstEncoding) {
  auto src = srcEncoding.dyn_cast<NvidiaMmaEncodingAttr>();
  auto dst = dstEncoding.dyn_cast<NvidiaMmaEncodingAttr>();
  if (!src || !dst)
    return false;
  // when #mma = MmaEncoding<version=3, warpsPerCTA=[..., 1]>
  return src && dst && src.getVersionMajor() == 3 &&
         src.getWarpsPerCTA()[1] == 1 && dst.getVersionMajor() == 3 &&
         dst.getWarpsPerCTA()[1] == 1;
}

bool isMmaToMmaShortcut(RankedTensorType srcTy, RankedTensorType dstTy) {
  return isMmaToMmaShortcut(srcTy.getEncoding(), dstTy.getEncoding());
}

// For MMAV3 dotOperand layout matches mma operand for f16 and bf16 cases.
bool matchMmaV3AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy) {
  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  auto mmaLayout = srcLayout.cast<NvidiaMmaEncodingAttr>();
  auto dotOperandLayout = dstLayout.cast<DotOperandEncodingAttr>();
  unsigned elementTypeSize = srcTy.getElementType().getIntOrFloatBitWidth();
  auto ans = mmaLayout.getVersionMajor() == 3 &&
             dotOperandLayout.getOpIdx() == 0 &&
             isMmaToMmaShortcut(dotOperandLayout.getParent(), srcLayout) &&
             (elementTypeSize == 16 || elementTypeSize == 8);
  return ans;
}

bool isMmaToDotShortcut(RankedTensorType srcTy, RankedTensorType dstTy) {
  if (matchMmaV3AndDotOperandLayout(srcTy, dstTy))
    return true;
  // dot_op<opIdx=0, parent=#mma> = #mma
  // when #mma = MmaEncoding<version=2, warpsPerCTA=[..., 1]>
  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  auto mmaLayout = srcLayout.cast<NvidiaMmaEncodingAttr>();
  unsigned elementTypeSize = srcTy.getElementType().getIntOrFloatBitWidth();
  auto dotOperandLayout = dstLayout.cast<DotOperandEncodingAttr>();
  return mmaLayout.getVersionMajor() == 2 &&
         mmaLayout.getWarpsPerCTA()[1] == 1 &&
         dotOperandLayout.getOpIdx() == 0 &&
         dotOperandLayout.getParent() == mmaLayout &&
         (elementTypeSize == 16 || elementTypeSize == 8);
}

namespace {

/// A data structure similar to SetVector but maintains
/// a deque instead of a vector to allow for efficient
/// push_back and pop_front operations.
/// Using SetVector doesn't suffice our needs because
/// it only pushes and pops from the back.
/// For example, if we have a queue like this:
/// 0->4 1->2->3
///    ^--------
/// where 3 depends on 4, once we pop 3, we found
/// 4 is not ready, so we check 2 and push 3 back
/// to the queue.
struct DFSSubgraphState {
  DFSSubgraphState() : set(), deque() {}
  DenseSet<Operation *> set;
  std::deque<Operation *> deque;

  bool push_back(Operation *op) {
    if (set.insert(op).second) {
      deque.push_back(op);
      return true;
    }
    return false;
  }

  Operation *pop_front() {
    Operation *op = deque.front();
    deque.pop_front();
    set.erase(op);
    return op;
  }

  bool empty() { return deque.empty(); }
};

/// DFS post-order implementation that maintains a global count to work across
/// multiple invocations, to help implement topological sort on multi-root DAGs.
/// We traverse all operations but only record the ones that appear in
/// `toSort` for the final result.
struct DFSState {
  DFSState(const SetVector<Operation *> &set) : toSort(set), seen() {}
  const SetVector<Operation *> &toSort;
  SmallVector<Operation *, 16> topologicalCounts;
  DenseSet<Operation *> seen;

  /// We mark each op as ready if all its operands and parents ops are seen. If
  /// an op is ready, we add it to the queue. Otherwise, we keep adding its
  /// operands to the ancestors set.
  /// We always want an op to be scheduled after all its parents to handle
  /// correctly cases with scf operations.
  void addToReadyQueue(Operation *op, DFSSubgraphState &subGraph,
                       SmallVector<Operation *, 4> &readyQueue) {
    bool ready = true;
    for (Value operand : op->getOperands()) {
      auto def = operand.getDefiningOp();
      if (def && !seen.count(def)) {
        subGraph.push_back(def);
        ready = false;
      }
    }
    Operation *parent = op->getParentOp();
    while (parent) {
      if (!seen.count(parent)) {
        subGraph.push_back(parent);
        ready = false;
      }
      parent = parent->getParentOp();
    }
    if (ready)
      readyQueue.push_back(op);
  }
};

void dfsPostorder(Operation *root, DFSState *state) {
  DFSSubgraphState subGraph;
  subGraph.push_back(root);
  SmallVector<Operation *> ops;
  while (!subGraph.empty()) {
    // Nodes in the ready queue are ready to be processed.
    // Meaning that either their operands are all seen or they have null
    // operands.
    SmallVector<Operation *, 4> readyQueue;
    auto *current = subGraph.pop_front();
    state->addToReadyQueue(current, subGraph, readyQueue);
    while (!readyQueue.empty()) {
      Operation *current = readyQueue.pop_back_val();
      if (!state->seen.insert(current).second)
        continue;
      ops.push_back(current);
      for (Value result : current->getResults()) {
        for (Operation *op : result.getUsers())
          state->addToReadyQueue(op, subGraph, readyQueue);
      }
      for (Region &region : current->getRegions()) {
        for (Operation &op : region.getOps())
          state->addToReadyQueue(&op, subGraph, readyQueue);
      }
    }
  }

  for (Operation *op : llvm::reverse(ops)) {
    if (state->toSort.count(op) > 0)
      state->topologicalCounts.push_back(op);
  }
}

} // namespace

SetVector<Operation *>
multiRootTopologicalSort(const SetVector<Operation *> &toSort) {
  if (toSort.empty()) {
    return toSort;
  }

  // Run from each root with global count and `seen` set.
  DFSState state(toSort);
  for (auto *s : toSort) {
    assert(toSort.count(s) == 1 && "NYI: multi-sets not supported");
    dfsPostorder(s, &state);
  }

  // Reorder and return.
  SetVector<Operation *> res;
  for (auto it = state.topologicalCounts.rbegin(),
            eit = state.topologicalCounts.rend();
       it != eit; ++it) {
    res.insert(*it);
  }
  return res;
}

SetVector<Operation *> multiRootGetSlice(Operation *op,
                                         TransitiveFilter backwardFilter,
                                         TransitiveFilter forwardFilter) {
  SetVector<Operation *> slice;
  slice.insert(op);

  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = backwardFilter;
    getBackwardSlice(currentOp, &backwardSlice, opt);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
    getForwardSlice(currentOp, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return multiRootTopologicalSort(slice);
}

namespace {
// Copied from TestDeadCodeAnalysis.cpp, because some dead code analysis
// interacts with constant propagation, but SparseConstantPropagation
// doesn't seem to be sufficient.
class ConstantAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    WalkResult result = top->walk([&](Operation *op) {
      if (failed(visit(op)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  LogicalResult visit(ProgramPoint point) override {
    Operation *op = point.get<Operation *>();
    Attribute value;
    if (matchPattern(op, m_Constant(&value))) {
      auto *constant = getOrCreate<dataflow::Lattice<dataflow::ConstantValue>>(
          op->getResult(0));
      propagateIfChanged(constant, constant->join(dataflow::ConstantValue(
                                       value, op->getDialect())));
      return success();
    }
    // Dead code analysis requires every operands has initialized ConstantValue
    // state before it is visited.
    // https://github.com/llvm/llvm-project/blob/2ec1aba2b69faa1de5f71832a48e25aa3b5d5314/mlir/lib/Analysis/DataFlow/DeadCodeAnalysis.cpp#L322
    // That's why we need to set all operands to unknown constants.
    setAllToUnknownConstants(op->getResults());
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks())
        setAllToUnknownConstants(block.getArguments());
    }
    return success();
  }

private:
  /// Set all given values as not constants.
  void setAllToUnknownConstants(ValueRange values) {
    dataflow::ConstantValue unknownConstant(nullptr, nullptr);
    for (Value value : values) {
      auto *constant =
          getOrCreate<dataflow::Lattice<dataflow::ConstantValue>>(value);
      propagateIfChanged(constant, constant->join(unknownConstant));
    }
  }
};
} // namespace

std::unique_ptr<DataFlowSolver> createDataFlowSolver() {
  auto solver = std::make_unique<DataFlowSolver>();
  solver->load<dataflow::DeadCodeAnalysis>();
  solver->load<ConstantAnalysis>();
  return solver;
}

static MakeTensorPtrOp getMakeTensorPtrOpImpl(Operation *op, Value v) {

  if (auto makeTensorPtrOp = dyn_cast<MakeTensorPtrOp>(op)) {
    return makeTensorPtrOp;
  }

  if (auto advanceOp = dyn_cast<AdvanceOp>(op)) {
    return getMakeTensorPtrOp(advanceOp.getPtr());
  }

  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    auto idx = v.cast<OpResult>().getResultNumber();
    llvm::SmallVector<scf::YieldOp> yieldOps;
    op->walk([&](Operation *op) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op))
        yieldOps.push_back(yieldOp);
    });

    // benzh@ if multi yields, all yields operand should come from same arg.
    Value newValue = yieldOps[0].getOperands()[idx];
    return getMakeTensorPtrOp(newValue);
  }

  llvm_unreachable("Unable to getMakeTensorPtr()");
}

MakeTensorPtrOp getMakeTensorPtrOp(Value v) {
  using BranchOps = llvm::SetVector<std::pair<Operation *, int>>;
  llvm::DenseMap<Block *, BranchOps> blockToCFOps;
  auto moduleOp =
      v.getParentBlock()->getParentOp()->getParentOfType<ModuleOp>();

  moduleOp.walk([&](Operation *op) {
    if (auto br = dyn_cast<cf::BranchOp>(op)) {
      Block *block = br.getDest();
      blockToCFOps[block].insert({op, -1});
    }
    if (auto condBr = dyn_cast<cf::CondBranchOp>(op)) {
      Block *blockT = condBr.getTrueDest();
      Block *blockF = condBr.getFalseDest();
      blockToCFOps[blockT].insert({condBr, 1});
      blockToCFOps[blockF].insert({condBr, 0});
    }
  });

  if (Operation *definingOp = v.getDefiningOp())
    return getMakeTensorPtrOpImpl(definingOp, v);

  // If there is no defining op, v must be a BlockArgument.
  BlockArgument arg = v.cast<BlockArgument>();
  unsigned argNum = arg.getArgNumber();
  Operation *argOwner = arg.getOwner()->getParentOp();

  if (auto forOp = dyn_cast<scf::ForOp>(argOwner))
    return getMakeTensorPtrOp(
        forOp.getOperand(argNum + forOp.getNumControlOperands() - 1));
  if (auto funcOp = dyn_cast<FuncOp>(argOwner)) {
    Block *block = arg.getOwner();
    Operation *op;
    int tOrF;
    std::tie(op, tOrF) = blockToCFOps[block][0];
    if (auto br = dyn_cast<cf::BranchOp>(op))
      return getMakeTensorPtrOp(br.getDestOperands()[argNum]);
    if (auto condBr = dyn_cast<cf::CondBranchOp>(op))
      return getMakeTensorPtrOp(tOrF ? condBr.getTrueDestOperands()[argNum]
                                     : condBr.getFalseDestOperands()[argNum]);
    return getMakeTensorPtrOp(argOwner->getOperand(argNum));
  }
  llvm_unreachable("Unable to getMakeTensorPtr()");
}

} // namespace mlir
