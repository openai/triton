#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir::triton::gpu;

namespace mlir::triton::gpu {

Type getFunctionType(Type resultType, ValueRange operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  return LLVM::LLVMFunctionType::get(resultType, operandTypes);
}

LLVM::LLVMFuncOp appendOrGetExternFuncOp(ConversionPatternRewriter &rewriter,
                                         Operation *op, StringRef funcName,
                                         Type funcType,
                                         StringRef libname /*= ""*/,
                                         StringRef libpath /*= ""*/) {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(op->getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  auto parent = op->getParentOfType<LLVM::LLVMFuncOp>();
  OpBuilder b(parent);
  auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  ret.getOperation()->setAttr("libname",
                              StringAttr::get(op->getContext(), libname));
  ret.getOperation()->setAttr("libpath",
                              StringAttr::get(op->getContext(), libpath));
  return ret;
}

Type getElementType(Value value) {
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return tensorType.getElementType();
  return type;
}
// MMA encoding has a different order depending on the element's bit width;
// reorder if we're in this case.
SmallVector<Value> reorderValues(const SmallVector<Value> &values, Type inType,
                                 Type ouType) {
  auto inTensorTy = inType.dyn_cast<RankedTensorType>();
  auto ouTensorTy = ouType.dyn_cast<RankedTensorType>();
  if (!inTensorTy || !ouTensorTy)
    return values;
  auto inEncoding = dyn_cast<DotOperandEncodingAttr>(inTensorTy.getEncoding());
  auto ouEncoding = dyn_cast<DotOperandEncodingAttr>(ouTensorTy.getEncoding());
  assert(inEncoding == ouEncoding);
  if (!inEncoding)
    return values;
  // If the parent of the dot operand is in block encoding, we don't need to
  // reorder elements
  auto parentEncoding = dyn_cast<NvidiaMmaEncodingAttr>(ouEncoding.getParent());
  if (!parentEncoding)
    return values;
  size_t inBitWidth = inTensorTy.getElementType().getIntOrFloatBitWidth();
  size_t ouBitWidth = ouTensorTy.getElementType().getIntOrFloatBitWidth();
  auto ouEltTy = ouTensorTy.getElementType();
  if (inBitWidth == ouBitWidth)
    return values;
  if (inEncoding.getOpIdx() == 1) {
    auto inK = inTensorTy.getShape()[inTensorTy.getRank() - 2];
    auto minK = 256 / inBitWidth;
    if (inK < minK) {
      // The smaller bit-width, the larger needed k to make a whole instr-shape.
      // It's OK to have k less than minK, because after cast, result type will
      // have proper k.
      return {values.begin(), values.begin() + (values.size() * inK / minK)};
    } else {
      return values;
    }
  }
  if (inBitWidth == 16 && ouBitWidth == 32) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 8) {
      ret.push_back(values[i]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
    }
    return ret;
  }
  if (inBitWidth == 8 && ouBitWidth == 16) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 16) {
      ret.push_back(values[i + 0]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 8]);
      ret.push_back(values[i + 9]);
      ret.push_back(values[i + 10]);
      ret.push_back(values[i + 11]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
      ret.push_back(values[i + 12]);
      ret.push_back(values[i + 13]);
      ret.push_back(values[i + 14]);
      ret.push_back(values[i + 15]);
    }
    return ret;
  }
  llvm_unreachable("unimplemented code path");
}

SmallVector<Value> unpackI32(const SmallVector<Value> &inValues, Type srcTy,
                             ConversionPatternRewriter &rewriter, Location loc,
                             const LLVMTypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<NvidiaMmaEncodingAttr>()))
    return inValues;
  SmallVector<Value> outValues;
  for (auto v : inValues) {
    // cast i32 to appropriate eltType vector and extract elements
    auto eltType = typeConverter->convertType(tensorTy.getElementType());
    auto vecType = vec_ty(eltType, 32 / eltType.getIntOrFloatBitWidth());
    auto vec = bitcast(v, vecType);
    for (int i = 0; i < 32 / eltType.getIntOrFloatBitWidth(); i++) {
      outValues.push_back(extract_element(vec, i32_val(i)));
    }
  }
  return outValues;
}

SmallVector<Value> packI32(const SmallVector<Value> &inValues, Type srcTy,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const LLVMTypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<NvidiaMmaEncodingAttr>()))
    return inValues;
  SmallVector<Value> outValues;
  auto eltType = typeConverter->convertType(tensorTy.getElementType());
  int vecWidth = 32 / eltType.getIntOrFloatBitWidth();
  auto vecType = vec_ty(eltType, vecWidth);
  for (int i = 0; i < inValues.size(); i += vecWidth) {
    Value vec = undef(vecType);
    for (int j = 0; j < vecWidth; j++) {
      vec = insert_element(vec, inValues[i + j], i32_val(j));
    }
    outValues.push_back(bitcast(vec, i32_ty));
  }
  return outValues;
}
} // namespace mlir::triton::gpu

namespace {
struct AddPtrOpConversion : public ConvertOpToLLVMPattern<AddPtrOp> {
  using ConvertOpToLLVMPattern<AddPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto typeConverter = getTypeConverter();
    auto resultTensorTy = resultTy.dyn_cast<RankedTensorType>();
    if (resultTensorTy) {
      unsigned elems = getTotalElemsPerThread(resultTy);
      Type elemTy = typeConverter->convertType(
          resultTensorTy.getElementType().cast<PointerType>().getPointeeType());
      Type ptrTy = typeConverter->convertType(resultTensorTy.getElementType());
      auto ptrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
      auto offsets = unpackLLElements(loc, adaptor.getOffset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(ptrTy, elemTy, ptrs[i], offsets[i]);
      }
      Value view =
          packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(resultTy.isa<PointerType>());
      auto resultPtrTy = typeConverter->convertType(resultTy);
      auto resultElemTy = typeConverter->convertType(
          resultTy.cast<PointerType>().getPointeeType());
      Value result =
          gep(resultPtrTy, resultElemTy, adaptor.getPtr(), adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct CmpIOpConversion
    : public ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<LLVM::ICmpOp> createDestOps(arith::CmpIOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          Type elemTy,
                                          MultipleOperandsRange operands,
                                          Location loc) const {
    return {rewriter.create<LLVM::ICmpOp>(
        loc, elemTy, ArithCmpIPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
  }

  static LLVM::ICmpPredicate
  ArithCmpIPredicateToLLVM(arith::CmpIPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__)                                                    \
  case arith::CmpIPredicate::item__:                                           \
    return LLVM::ICmpPredicate::item__

      __PRED_ENUM(eq);
      __PRED_ENUM(ne);
      __PRED_ENUM(sgt);
      __PRED_ENUM(sge);
      __PRED_ENUM(slt);
      __PRED_ENUM(sle);
      __PRED_ENUM(ugt);
      __PRED_ENUM(uge);
      __PRED_ENUM(ult);
      __PRED_ENUM(ule);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpIPredicate");
  }
};

struct CmpFOpConversion
    : public ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  static SmallVector<LLVM::FCmpOp>
  createDestOps(arith::CmpFOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                MultipleOperandsRange operands, Location loc) {
    return {rewriter.create<LLVM::FCmpOp>(
        loc, elemTy, ArithCmpFPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
  }

  static LLVM::FCmpPredicate
  ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

      __PRED_ENUM(OEQ, oeq);
      __PRED_ENUM(ONE, one);
      __PRED_ENUM(OGT, ogt);
      __PRED_ENUM(OGE, oge);
      __PRED_ENUM(OLT, olt);
      __PRED_ENUM(OLE, ole);
      __PRED_ENUM(ORD, ord);
      __PRED_ENUM(UEQ, ueq);
      __PRED_ENUM(UGT, ugt);
      __PRED_ENUM(UGE, uge);
      __PRED_ENUM(ULT, ult);
      __PRED_ENUM(ULE, ule);
      __PRED_ENUM(UNE, une);
      __PRED_ENUM(UNO, uno);
      __PRED_ENUM(AlwaysTrue, _true);
      __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }
};

struct ExternElementwiseOpConversion
    : public ElementwiseOpConversionBase<ExternElementwiseOp,
                                         ExternElementwiseOpConversion> {
  using Base = ElementwiseOpConversionBase<ExternElementwiseOp,
                                           ExternElementwiseOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  SmallVector<Value> createDestOps(ExternElementwiseOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    StringRef funcName = op.getSymbol();
    if (funcName.empty())
      llvm::errs() << "ExternElementwiseOpConversion";

    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp = appendOrGetExternFuncOp(
        rewriter, op, funcName, funcType, op.getLibname(), op.getLibpath());
    return {
        rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]).getResult()};
  }
};

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<DestOp> createDestOps(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    Type elemTy, MultipleOperandsRange operands,
                                    Location loc) const {
    return {rewriter.create<DestOp>(loc, elemTy, operands[0],
                                    adaptor.getAttributes().getValue())};
  }
};

struct ElementwiseInlineAsmOpConversion
    : public ConvertOpToLLVMPattern<ElementwiseInlineAsmOp> {
  using Base = ConvertOpToLLVMPattern<ElementwiseInlineAsmOp>;

  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  // If operand size is smaller than 32 bits, pack in groups of 32 bits.
  SmallVector<Value> packOperands(ElementwiseInlineAsmOp op,
                                  MultipleOperandsRange operands,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) const {
    SmallVector<Value> packedOperands;
    unsigned numPackedElements = op.getPackedElement();
    for (int i = 0, e = op.getNumOperands(); i < e; i++) {
      Type elemTy = getElementType(op.getOperand(i));
      unsigned bitWidth =
          elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 64;
      unsigned numElementPerReg = bitWidth < 32 ? 32 / bitWidth : 1;
      numElementPerReg = std::min(numElementPerReg, numPackedElements);
      for (int j = 0; j < numPackedElements; j += numElementPerReg) {
        if (numElementPerReg == 1) {
          packedOperands.push_back(operands[j][i]);
          continue;
        }
        Type t =
            vec_ty(getTypeConverter()->convertType(elemTy), numElementPerReg);
        Value packed = undef(t);
        for (int k = 0; k < numElementPerReg; k++) {
          packed = insert_element(packed, operands[j + k][i], i32_val(k));
        }
        packedOperands.push_back(packed);
      }
    }
    return packedOperands;
  }

  SmallVector<SmallVector<Value>>
  createDestOps(ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter,
                MultipleOperandsRange operands, Location loc) const {
    auto ctx = op->getContext();

    if (operands.size() % op.getPackedElement() != 0)
      llvm::report_fatal_error("Inline asm op has more packed elements than "
                               "number of elements per thread.");

    // Pack elems smaller than 32 bits into 32-bit registers.
    SmallVector<Value> packedOperands =
        packOperands(op, operands, rewriter, loc);

    // Types returned by the LLVM asm op.  If there's more than one, they'll be
    // wrapped in a struct.
    SmallVector<Type> asmRetTypes;
    for (auto result : op.getResult()) {
      auto ty = getTypeConverter()->convertType(getElementType(result));

      // Pack return elements into 32-bits.
      unsigned bitWidth = ty.isIntOrFloat() ? ty.getIntOrFloatBitWidth() : 64;
      unsigned numElemsPerReg =
          std::min(bitWidth < 32 ? 32 / bitWidth : 1, op.getPackedElement());
      assert(op.getPackedElement() % numElemsPerReg == 0);
      if (numElemsPerReg > 1) {
        ty = vec_ty(ty, numElemsPerReg);
      }
      for (unsigned i = 0; i < op.getPackedElement() / numElemsPerReg; i++) {
        asmRetTypes.push_back(ty);
      }
    }
    Type asmRetType =
        asmRetTypes.size() > 1 ? struct_ty(asmRetTypes) : asmRetTypes[0];

    Value asmResults =
        rewriter
            .create<LLVM::InlineAsmOp>(
                loc, asmRetType,
                /*operands=*/packedOperands,
                /*asm_string=*/op.getAsmString(),
                /*constraints=*/op.getConstraints(),
                /*has_side_effects=*/!op.getPure(),
                /*is_align_stack=*/false,
                /*asm_dialect=*/
                LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                          LLVM::AsmDialect::AD_ATT),
                /*operand_attrs=*/ArrayAttr())
            ->getResult(0);

    // asmResults is a flat struct; pack its values into
    // [return_value][op.getPackedElement()].
    SmallVector<SmallVector<Value>> ret(op->getNumResults());
    for (int i = 0; i < op->getNumResults(); i++) {
      int structIdx = 0;
      for (int j = 0; j < op.getPackedElement(); j++) {
        Value val;
        if (asmRetTypes.size() > 1) {
          val =
              extract_val(asmResults, i * op.getPackedElement() + structIdx++);
        } else {
          val = asmResults;
        }
        if (auto vectorTy = val.getType().dyn_cast<VectorType>()) {
          for (int k = 0; k < vectorTy.getNumElements(); k++) {
            ret[i].push_back(extract_element(val, i32_val(k)));
          }
          j += vectorTy.getNumElements() - 1;
        } else {
          ret[i].push_back(val);
        }
      }
    }
    return ret;
  }

  LogicalResult
  matchAndRewrite(ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Layout is unpackedOperands[operand][elem].
    SmallVector<SmallVector<Value>> unpackedOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto subOperands = unpackLLElements(loc, operand, rewriter);
      unpackedOperands.push_back(
          unpackI32(subOperands, argTy, rewriter, loc, getTypeConverter()));
    }

    // Although we ensure that all operands and results to this op have the same
    // encoding, MMA layouts have a different physical ordering depending on the
    // bit width of the underlying element.
    //
    // Thus if the inputs to the inline asm op are MMA with different widths, we
    // need to reorder them so we iterate over the operands' elements in the
    // same logical order.
    for (unsigned i = 0; i < unpackedOperands.size(); ++i) {
      unpackedOperands[i] = reorderValues(
          unpackedOperands[i], /*inType=*/op->getOperand(i).getType(),
          /*ouType=*/op->getResult(0).getType());
    }

    size_t numElemsPerThread = 1;
    auto structType = typeConverter->convertType(op->getResult(0).getType())
                          .dyn_cast<LLVM::LLVMStructType>();
    if (structType) {
      numElemsPerThread = structType.getBody().size();
    }

    // These are checked by the verifier, so we don't need to raise a nice
    // error.
    assert(all_of(unpackedOperands, [&](auto &operands) {
      return operands.size() == numElemsPerThread;
    }));
    if (numElemsPerThread % op.getPackedElement() != 0) {
      // Pad with the first value of the operand for each operand to have a
      // multiple of op.getPackedElement() elements.
      int numPaddedValue =
          op.getPackedElement() - numElemsPerThread % op.getPackedElement();
      for (auto &operands : unpackedOperands) {
        for (int i = 0; i < numPaddedValue; i++) {
          operands.push_back(operands[0]);
        }
      }
    }

    // Run the inline asm op on each block of elements.
    //
    // Layout is unpackedResults[result_idx][elem].
    //
    // This loop always runs at least once, even when the asm has no input
    // elements.
    SmallVector<SmallVector<Value>> unpackedResults(op->getNumResults());
    for (unsigned i = 0; i < numElemsPerThread; i += op.getPackedElement()) {
      // Block of elements to process with one call to the inline asm.  This is
      // ordered opposite `unpackedResults`: The outer dim is
      // op.getPackedElement(), and the inner dim is the operand.
      SmallVector<SmallVector<Value>> block(op.getPackedElement());
      for (auto &os : unpackedOperands) {
        for (int j = 0; j < op.getPackedElement(); j++) {
          block[j].push_back(os[i + j]);
        }
      }
      auto cur = createDestOps(op, adaptor, rewriter, block, loc);
      assert(cur.size() == unpackedResults.size());
      for (unsigned j = 0; j < cur.size(); j++) {
        unpackedResults[j].insert(unpackedResults[j].end(), cur[j].begin(),
                                  cur[j].end());
      }
    }
    for (auto &results : unpackedResults) {
      results.resize(numElemsPerThread);
    }
    // Reorder and pack the results.
    SmallVector<Value> outs;
    for (int i = 0; i < unpackedResults.size(); i++) {
      // We reordered all the inputs so they match operand 0.  Reorder the
      // outputs accordingly.
      if (op->getNumOperands() > 0) {
        unpackedResults[i] = reorderValues(
            unpackedResults[i], /*inType=*/op->getOperand(0).getType(),
            /*ouType=*/op->getResult(i).getType());
      }
      auto packed = packI32(unpackedResults[i], op->getResult(i).getType(),
                            rewriter, loc, getTypeConverter());
      outs.push_back(packLLElements(loc, getTypeConverter(), unpackedResults[i],
                                    rewriter, op->getResult(i).getType()));
    }

    rewriter.replaceOp(op, outs);
    return success();
  }
};

struct AbsIOpConversion
    : ElementwiseOpConversionBase<math::AbsIOp, AbsIOpConversion> {
  using Base = ElementwiseOpConversionBase<math::AbsIOp, AbsIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::AbsIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    return {rewriter.create<LLVM::AbsOp>(loc, elemTy, operands[0][0],
                                         /*is_int_min_poison=*/false)};
  }
};

struct AbsFOpConversion
    : ElementwiseOpConversionBase<math::AbsFOp, AbsFOpConversion> {
  using Base = ElementwiseOpConversionBase<math::AbsFOp, AbsFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::AbsFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (llvm::isa<IntegerType>(elemTy)) {
      // Mask out the sign bit
      auto num_bits =
          getElementTypeOrSelf(op.getType()).getIntOrFloatBitWidth();
      assert(num_bits <= 16);
      auto mask = (1u << (num_bits - 1u)) - 1u;
      auto maskAttr = rewriter.getIntegerAttr(elemTy, mask);
      auto maskConst = rewriter.create<LLVM::ConstantOp>(loc, maskAttr);
      return {and_(operands[0][0], maskConst)};
    }

    return {rewriter.create<LLVM::FAbsOp>(loc, elemTy, operands[0][0])};
  }
};
/// The lowering of index_cast becomes an integer conversion since index
/// becomes an integer.  If the bit width of the source and target integer
/// types is the same, just erase the cast.  If the target type is wider,
/// sign-extend the value, otherwise truncate it.
struct IndexCastOpLowering
    : public ElementwiseOpConversionBase<arith::IndexCastOp,
                                         IndexCastOpLowering> {
  using Base =
      ElementwiseOpConversionBase<arith::IndexCastOp, IndexCastOpLowering>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::IndexCastOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy =
        this->getTypeConverter()->convertType(getElementType(op.getIn()));
    unsigned targetBits = elemTy.getIntOrFloatBitWidth();
    unsigned sourceBits = inElemTy.getIntOrFloatBitWidth();

    if (targetBits == sourceBits)
      return {operands[0][0]};
    if (targetBits < sourceBits)
      return {rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, elemTy,
                                                         operands[0][0])};
    return {
        rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, elemTy, operands[0][0])};
  }
};

struct SelectOpConversion
    : ElementwiseOpConversionBase<arith::SelectOp, SelectOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SelectOp, SelectOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::SelectOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    std::array<Value, 3> llvmOperands;
    if (operands[0].size() == 2) {
      // Case of scalar condition with tensor operands.
      assert(op.getCondition().getType().isInteger(1));
      llvmOperands = {adaptor.getCondition(), operands[0][0], operands[0][1]};
    } else {
      llvmOperands = {operands[0][0], operands[0][1], operands[0][2]};
    }
    return {rewriter.create<LLVM::SelectOp>(
        loc, llvmOperands[1].getType(), llvmOperands,
        adaptor.getAttributes().getValue())};
  }
};
template <typename OpTy>
struct MinMaxFOpConversion
    : ElementwiseOpConversionBase<OpTy, MinMaxFOpConversion<OpTy>> {
  using Base = ElementwiseOpConversionBase<OpTy, MinMaxFOpConversion<OpTy>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  static_assert(std::is_same<OpTy, arith::MinimumFOp>::value ||
                    std::is_same<OpTy, arith::MaximumFOp>::value,
                "OpTy must be arith::MinimumFOp or arith::MaximumFOp");

  // Choose the destination op based on the OpTy.
  using DestOpNanProp =
      typename std::conditional<std::is_same<OpTy, arith::MinimumFOp>::value,
                                LLVM::MinimumOp, LLVM::MaximumOp>::type;
  using DestOpNoNanProp =
      typename std::conditional<std::is_same<OpTy, arith::MinimumFOp>::value,
                                LLVM::MinNumOp, LLVM::MaxNumOp>::type;

  explicit MinMaxFOpConversion(LLVMTypeConverter &typeConverter,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               bool hwNanPropagationSupported,
                               PatternBenefit benefit = 1)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        hwNanPropagationSupported(hwNanPropagationSupported) {}

  SmallVector<Value> createDestOps(OpTy op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (hwNanPropagationSupported) {
      return {rewriter.create<DestOpNanProp>(loc, elemTy, operands[0][0],
                                             operands[0][1])};
    }
    // Handle workaround for NaN propagation, i.e. software emulation of NaN
    // propagation. If any of the operands is NaN, return NaN.
    auto lhs = operands[0][0];
    auto rhs = operands[0][1];
    auto lhsIsNan =
        rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une, lhs, lhs);
    auto rhsIsNan =
        rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une, rhs, rhs);
    auto isNan = rewriter.create<LLVM::OrOp>(loc, lhsIsNan, rhsIsNan);
    auto nonNanRes = rewriter.create<DestOpNoNanProp>(loc, elemTy, lhs, rhs);

    auto nan = LLVM::createNaNConstant(loc, rewriter, elemTy);

    // Select the result based on the isNan flag.
    return {rewriter.create<LLVM::SelectOp>(loc, isNan, nan, nonNanRes)};
  }

private:
  bool hwNanPropagationSupported;
};

struct ClampFOpConversion
    : ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion> {
  using Base = ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit ClampFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        targetInfo(targetInfo) {}

  SmallVector<Value> createDestOps(ClampFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // Clip pattern not found, use min/max.
    if (op.getPropagateNan() == PropagateNan::ALL) {
      if (targetInfo.supportMaximumMinimum()) {
        auto v = rewriter.create<LLVM::MaximumOp>(loc, elemTy, operands[0][0],
                                                  operands[0][1]);
        return {rewriter.create<LLVM::MinimumOp>(loc, v, operands[0][2])};
      }
      // On pre-80 compute capability, we need to handle NaN propagation
      // manually. We need to check only the first operand for clamp.
      auto lhs = operands[0][0];
      auto isNan = rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une,
                                                 lhs, lhs);
      auto v = rewriter.create<LLVM::MaxNumOp>(loc, elemTy, operands[0][0],
                                               operands[0][1]);
      auto nonNanRes = rewriter.create<LLVM::MinNumOp>(loc, v, operands[0][2]);
      auto nan = LLVM::createNaNConstant(loc, rewriter, elemTy);
      // Select the result based on the isNan flag.
      return {rewriter.create<LLVM::SelectOp>(loc, isNan, nan, nonNanRes)};
    }

    // No NaN propagation.
    assert(op.getPropagateNan() == PropagateNan::NONE);
    auto v = rewriter.create<LLVM::MaxNumOp>(loc, elemTy, operands[0][0],
                                             operands[0][1]);
    return {rewriter.create<LLVM::MinNumOp>(loc, v, operands[0][2])};
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateMinMaxFOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, bool hwNanPropagationSupported,
    PatternBenefit benefit) {
  patterns.add<MinMaxFOpConversion<arith::MinimumFOp>>(
      typeConverter, axisInfoAnalysis, hwNanPropagationSupported, benefit);
  patterns.add<MinMaxFOpConversion<arith::MaximumFOp>>(
      typeConverter, axisInfoAnalysis, hwNanPropagationSupported, benefit);
}

void mlir::triton::populateClampFOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit) {
  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis, targetInfo,
                                   benefit);
}

void mlir::triton::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);

  POPULATE_UNARY_OP(arith::TruncIOp, LLVM::TruncOp)
  POPULATE_UNARY_OP(arith::ExtSIOp, LLVM::SExtOp)
  POPULATE_UNARY_OP(arith::ExtUIOp, LLVM::ZExtOp)
  POPULATE_UNARY_OP(arith::FPToUIOp, LLVM::FPToUIOp)
  POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)
  POPULATE_UNARY_OP(math::FloorOp, math::FloorOp)
  POPULATE_UNARY_OP(math::LogOp, math::LogOp)
  POPULATE_UNARY_OP(math::Log2Op, math::Log2Op)
  POPULATE_UNARY_OP(math::CosOp, math::CosOp)
  POPULATE_UNARY_OP(math::SinOp, math::SinOp)
  POPULATE_UNARY_OP(math::SqrtOp, math::SqrtOp)
  POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)
  POPULATE_UNARY_OP(math::Exp2Op, math::Exp2Op)
  POPULATE_UNARY_OP(math::ErfOp, math::ErfOp)
  POPULATE_UNARY_OP(triton::BitcastOp, LLVM::BitcastOp)
  POPULATE_UNARY_OP(triton::IntToPtrOp, LLVM::IntToPtrOp)
  POPULATE_UNARY_OP(triton::PtrToIntOp, LLVM::PtrToIntOp)
#undef POPULATE_UNARY_OP

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);

  POPULATE_BINARY_OP(arith::SubIOp, LLVM::SubOp) // -
  POPULATE_BINARY_OP(arith::AddIOp, LLVM::AddOp) // +
  POPULATE_BINARY_OP(arith::MulIOp, LLVM::MulOp) // *
  POPULATE_BINARY_OP(arith::DivSIOp, LLVM::SDivOp)
  POPULATE_BINARY_OP(arith::DivUIOp, LLVM::UDivOp)
  POPULATE_BINARY_OP(arith::RemFOp, LLVM::FRemOp) // %
  POPULATE_BINARY_OP(arith::RemSIOp, LLVM::SRemOp)
  POPULATE_BINARY_OP(arith::RemUIOp, LLVM::URemOp)
  POPULATE_BINARY_OP(arith::AndIOp, LLVM::AndOp)   // &
  POPULATE_BINARY_OP(arith::OrIOp, LLVM::OrOp)     // |
  POPULATE_BINARY_OP(arith::XOrIOp, LLVM::XOrOp)   // ^
  POPULATE_BINARY_OP(arith::ShLIOp, LLVM::ShlOp)   // <<
  POPULATE_BINARY_OP(arith::ShRSIOp, LLVM::AShrOp) // >>
  POPULATE_BINARY_OP(arith::ShRUIOp, LLVM::LShrOp) // >>
  // fmin (return non-NaN if either op is non-NaN)
  POPULATE_BINARY_OP(arith::MinNumFOp, LLVM::MinNumOp)
  // fmax (return non-NaN if either op is non-NaN)
  POPULATE_BINARY_OP(arith::MaxNumFOp, LLVM::MaxNumOp)
  POPULATE_BINARY_OP(arith::MinSIOp, LLVM::SMinOp) // smin
  POPULATE_BINARY_OP(arith::MaxSIOp, LLVM::SMaxOp) // smax
  POPULATE_BINARY_OP(arith::MinUIOp, LLVM::UMinOp) // umin
  POPULATE_BINARY_OP(arith::MaxUIOp, LLVM::UMaxOp) // umax
#undef POPULATE_BINARY_OP

  patterns.add<ElementwiseOpConversion<math::FmaOp, LLVM::FMAOp>>(
      typeConverter, axisInfoAnalysis, benefit);

  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<CmpIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<CmpFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ExternElementwiseOpConversion>(typeConverter, axisInfoAnalysis,
                                              benefit);
  patterns.add<ElementwiseInlineAsmOpConversion>(typeConverter, benefit);
  patterns.add<AbsIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<AbsFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<IndexCastOpLowering>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SelectOpConversion>(typeConverter, axisInfoAnalysis, benefit);
}
