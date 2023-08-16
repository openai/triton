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

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <numeric>

//===----------------------------------------------------------------------===//
//
// This pass works after pipeline pass, converts the remaining tt.LoadOp taking
// ptr<tensor> as input into ttg.InsertSliceAsyncOp and emit proper barriers
//
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::getCTALayout;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

struct MaterializeLoadStorePass
    : public MaterializeLoadStoreBase<MaterializeLoadStorePass> {

public:
  MaterializeLoadStorePass() = default;
  MaterializeLoadStorePass(int numWarps, int computeCapability) {
    this->numWarps = numWarps;
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    SmallVector<mlir::triton::LoadOp> worklists;
    getOperation()->walk([&](mlir::triton::LoadOp load) -> void {
      if (isLoadFromTensorPtr(load)) {
        worklists.push_back(load);
      }
    });
    for (auto load : worklists) {
      materializeLoadTilePtr(load);
    }

    SmallVector<mlir::triton::StoreOp> storeOpWorklists;
    getOperation()->walk([&](mlir::triton::StoreOp store) -> void {
      if (isStoreToTensorPtr(store)) {
        storeOpWorklists.push_back(store);
      }
    });
    for (auto store : storeOpWorklists) {
      materializeStoreTilePtr(store);
    }
  }

private:
  void materializeLoadTilePtr(mlir::triton::LoadOp load);
  void materializeStoreTilePtr(mlir::triton::StoreOp store);
};

void MaterializeLoadStorePass::materializeLoadTilePtr(
    mlir::triton::LoadOp load) {
  if (computeCapability < 90)
    return;
  if (!::triton::tools::getBoolEnv("ENABLE_TMA"))
    return;
  auto loc = load.getLoc();
  OpBuilder b(load);
  auto loadTy = load.getType().dyn_cast<RankedTensorType>();
  auto loadShape = loadTy.getShape();
  auto CTASplitNum = ttg::getCTASplitNum(loadTy.getEncoding());
  auto shapePerSlice = ttg::getShapePerCTA(CTASplitNum, loadShape);
  auto elemTy = loadTy.getElementType();
  assert(loadTy);
  SmallVector<int64_t> bufferShape(loadShape.begin(), loadShape.end());
  bufferShape.insert(bufferShape.begin(), 1);

  auto sharedEncoding = getSharedEncoding(loadTy);
  auto bufferTy = RankedTensorType::get(bufferShape, elemTy, sharedEncoding);
  Value buffer = b.create<ttg::AllocTensorOp>(loc, bufferTy);
  unsigned elems = std::accumulate(shapePerSlice.begin(), shapePerSlice.end(),
                                   1, std::multiplies{});
  elems *= (elemTy.getIntOrFloatBitWidth() / 8);
  auto mBarrierTy = mlir::triton::PointerType::get(b.getIntegerType(64), 3);
  Value mBarrier = b.create<ttng::AllocMBarrierOp>(loc, mBarrierTy, 1);
  Value _0 = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value threadId = b.create<ttng::GetThreadIdOp>(loc);
  Value pred =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, threadId, _0);
  b.create<ttng::MBarrierArriveOp>(loc, mBarrier, pred, /*remoteCtaId*/ nullptr,
                                   /*trackAsyncOp*/ false, elems);
  Value inserted = b.create<ttng::InsertSliceAsyncV2Op>(
      loc, bufferTy, load.getPtr(), buffer,
      /*index*/ _0, mBarrier, load.getMask(), load.getOther(), load.getCache(),
      load.getEvict(), load.getIsVolatile(),
      /*axis*/ 0);
  auto extractedTy = RankedTensorType::get(loadShape, elemTy, sharedEncoding);
  Value extracted = b.create<mlir::triton::gpu::ExtractSliceOp>(
      loc, extractedTy, inserted,
      SmallVector<OpFoldResult>{b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
                                b.getI64IntegerAttr(0)},
      SmallVector<OpFoldResult>{b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(loadShape[0]),
                                b.getI64IntegerAttr(loadShape[1])},
      SmallVector<OpFoldResult>{b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(1)});

  Value phase = b.create<arith::ConstantIntOp>(loc, 0, 1);
  b.create<ttng::MBarrierWaitOp>(loc, mBarrier, phase);
  Value newValue =
      b.create<ttg::ConvertLayoutOp>(loc, load.getType(), extracted);
  load.getResult().replaceAllUsesWith(newValue);
  load->erase();
}

void MaterializeLoadStorePass::materializeStoreTilePtr(
    mlir::triton::StoreOp store) {
  if (computeCapability < 90 || !::triton::tools::getBoolEnv("ENABLE_TMA"))
    return;
  auto loc = store.getLoc();
  OpBuilder builder(store);
  auto value = store.getValue();
  auto dst = store.getPtr();

  auto cvtOp = llvm::dyn_cast_or_null<mlir::triton::gpu::ConvertLayoutOp>(
      value.getDefiningOp());
  if (cvtOp) {
    auto srcTy = cvtOp.getOperand().getType().cast<RankedTensorType>();
    auto dstTy = cvtOp.getResult().getType().cast<RankedTensorType>();
    auto elemTy = srcTy.getElementType();
    auto srcMmaLayout = srcTy.getEncoding().dyn_cast<MmaEncodingAttr>();
    auto dstSharedLayout = dstTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    auto truncFOP = llvm::dyn_cast_or_null<arith::TruncFOp>(cvtOp.getOperand().getDefiningOp());
    unsigned numElems = ttg::getTotalElemsPerThread(srcTy);
    auto inOrd = ttg::getOrder(srcTy.getEncoding());
    auto outOrd = ttg::getOrder(dstTy.getEncoding());
    if (srcMmaLayout && srcMmaLayout.isHopper() && dstSharedLayout && truncFOP && elemTy.getIntOrFloatBitWidth() == 16 && numElems >= 16 && inOrd == outOrd) {
      builder.create<ttng::StoreAsyncOp>(loc, dst, cvtOp.getOperand());
      builder.create<ttg::AsyncBulkCommitGroupOp>(loc);
      builder.create<ttg::AsyncBulkWaitOp>(loc, 0);
      store->erase();
      return;
    }
  }

  auto *ctx = store.getContext();
  auto storeTy = value.getType().dyn_cast<RankedTensorType>();
  assert(storeTy);
  auto storeElemTy = storeTy.getElementType();
  auto ctaLayout = getCTALayout(storeTy.getEncoding());
  auto storeShape = storeTy.getShape();
  SmallVector<int64_t> bufferShape(storeShape.begin(), storeShape.end());
  auto rank = storeShape.size();
  // The order of smem should be consistent with gmem.
  auto makeTensorPtrOp = getMakeTensorPtrOp(dst);
  SmallVector<unsigned> sharedOrder;
  for (auto o : makeTensorPtrOp.getOrder()) {
    sharedOrder.emplace_back(o);
  }
  auto sharedEncoding = SharedEncodingAttr::get(ctx, storeShape, sharedOrder,
                                                ctaLayout, storeElemTy);
  auto bufferTy =
      RankedTensorType::get(bufferShape, storeElemTy, sharedEncoding);
  Value cvt = builder.create<ttg::ConvertLayoutOp>(loc, bufferTy, value);
  builder.create<ttng::StoreAsyncOp>(loc, dst, cvt);
  builder.create<mlir::triton::gpu::AsyncBulkCommitGroupOp>(loc);
  builder.create<mlir::triton::gpu::AsyncBulkWaitOp>(loc, 0);
  store->erase();
}

} // anonymous namespace

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUMaterializeLoadStorePass(int numWarps,
                                                    int computeCapability) {
  return std::make_unique<MaterializeLoadStorePass>(numWarps,
                                                    computeCapability);
}
