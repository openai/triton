#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

using ValueTableFMA = std::map<std::pair<int, int>, Value>;

static ValueTableFMA
getValueTableFromStructFMA(Value val, int K, int n0, int shapePerCTATile,
                           int sizePerThread,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const LLVMTypeConverter *typeConverter, Type type) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTATile)
      for (unsigned mm = 0; mm < sizePerThread; ++mm) {
        res[{m + mm, k}] = elems[index++];
      }
  }
  return res;
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto C = op.getC();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto bTensorTy = cast<RankedTensorType>(B.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  auto aShapePerCTA = getShapePerCTA(aTensorTy);
  auto bShapePerCTA = getShapePerCTA(bTensorTy);

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  auto order = dLayout.getOrder();
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getSizePerThread(dLayout);
  auto shapePerCTATile = getShapePerCTATile(dLayout);

  int K = aShapePerCTA[1];
  int M = aShapePerCTA[0];
  int N = bShapePerCTA[1];

  int mShapePerCTATile =
      order[0] == 1 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int mSizePerThread =
      order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  int nShapePerCTATile =
      order[0] == 0 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int nSizePerThread =
      order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];

  auto has =
      getValueTableFromStructFMA(llA, K, M, mShapePerCTATile, mSizePerThread,
                                 rewriter, loc, typeConverter, aTensorTy);
  auto hbs =
      getValueTableFromStructFMA(llB, K, N, nShapePerCTATile, nSizePerThread,
                                 rewriter, loc, typeConverter, bTensorTy);

  SmallVector<Value> ret = cc;
  bool isCRow = order[0] == 1;

  for (unsigned k = 0; k < K; k++) {
    for (unsigned m = 0; m < M; m += mShapePerCTATile)
      for (unsigned n = 0; n < N; n += nShapePerCTATile)
        for (unsigned mm = 0; mm < mSizePerThread; ++mm)
          for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
            int mIdx = m / mShapePerCTATile * mSizePerThread + mm;
            int nIdx = n / nShapePerCTATile * nSizePerThread + nn;

            int z = isCRow
                        ? mIdx * N / nShapePerCTATile * mSizePerThread + nIdx
                        : nIdx * M / mShapePerCTATile * nSizePerThread + mIdx;
            Type z_type = ret[z].getType();
            auto cvt_a =
                rewriter.create<LLVM::FPExtOp>(loc, z_type, has[{m + mm, k}]);
            auto cvt_b =
                rewriter.create<LLVM::FPExtOp>(loc, z_type, hbs[{n + nn, k}]);
            ret[z] =
                rewriter.create<LLVM::FMulAddOp>(loc, cvt_a, cvt_b, ret[z]);
          }
  }

  auto res = packLLElements(loc, typeConverter, ret, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}
