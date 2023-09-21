#ifndef TRITON_OPS
#define TRITON_OPS

include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/Interfaces/CastInterfaces.td" // CastOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/CastInterfaces.td" // CastOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface

//
// Op Base
//
class TT_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Dialect, mnemonic, !listconcat(traits, [TensorSizeTrait])> {
}

//
// Cast Ops
//
// Use cast ops in arith:
//   bitcast
//   fptoui, fptosi, uitofp, sitofp,
//   extf, tructf,
//   extui, extsi, tructi
def TT_IntToPtrOp : TT_Op<"int_to_ptr", [Elementwise,
                                         SameOperandsAndResultShape,
                                         SameOperandsAndResultEncoding,
                                         Pure,
                                         /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast int64 to pointer";

    let arguments = (ins TT_I64Like:$from);

    let results = (outs TT_PtrLike:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";
}

def TT_PtrToIntOp : TT_Op<"ptr_to_int", [Elementwise,
                                         SameOperandsAndResultShape,
                                         SameOperandsAndResultEncoding,
                                         Pure,
                                         /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast pointer to int64";

    let arguments = (ins TT_PtrLike:$from);

    let results = (outs TT_I64Like:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";
}

// arith.bitcast doesn't support pointers
def TT_BitcastOp : TT_Op<"bitcast", [Elementwise,
                                     SameOperandsAndResultShape,
                                     SameOperandsAndResultEncoding,
                                     Pure,
                                     /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast between types of the same bitwidth";

    let arguments = (ins TT_Type:$from);

    let results = (outs TT_Type:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";

    // TODO: Add verifier
}

// FIXME: Not elementwise because scalars are not supported
def TT_FpToFpOp : TT_Op<"fp_to_fp", [SameOperandsAndResultShape,
                                     SameOperandsAndResultEncoding,
                                     Pure,
                                     /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Floating point casting for custom types";

    let description = [{
        Floating point casting for custom types (F8).

        F8 <-> FP16, BF16, FP32, FP64
    }];

    let arguments = (ins TT_FloatLike:$from);

    let results = (outs TT_FloatLike:$result);

    let assemblyFormat = "$from attr-dict `:` type($from) `->` type($result)";

    // TODO: We need a verifier here.
}

//
// Pointer Arith Ops
//
def TT_AddPtrOp : TT_Op<"addptr",
                     [Pure,
                      Elementwise,
                      SameOperandsAndResultShape,
                      SameOperandsAndResultEncoding,
                      TypesMatchWith<"result type matches ptr type",
                                     "result", "ptr", "$_self">]> {
    let arguments = (ins TT_PtrLike:$ptr, TT_IntLike:$offset);

    let results = (outs TT_PtrLike:$result);

    let assemblyFormat = "$ptr `,` $offset attr-dict `:` type($result) `,` type($offset)";
}

def TT_AdvanceOp : TT_Op<"advance",
                         [Pure,
                          TypesMatchWith<"result type matches ptr type",
                                         "result", "ptr", "$_self">]> {
    let summary = "Advance a tensor pointer by offsets";

    let arguments = (ins TT_TensorPtr:$ptr, Variadic<I32>:$offsets);

    let results = (outs TT_TensorPtr:$result);

    let assemblyFormat = "$ptr `,` `[` $offsets `]` attr-dict `:` type($result)";
}

//
// Load/Store Ops
//
def TT_LoadOp : TT_Op<"load",
                      [SameLoadStoreOperandsAndResultShape,
                       SameLoadStoreOperandsAndResultEncoding,
                       AttrSizedOperandSegments,
                       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                       TypesMatchWith<"infer ptr type from result type",
                                      "result", "ptr", "$_self",
                                      "mlir::OpTrait::impl::verifyLoadStorePointerAndValueType">,
                       TypesMatchWith<"infer mask type from result type or none",
                                      "result", "mask", "getI1SameShape($_self)",
                                      "($_op.getOperands().size() <= 1) || std::equal_to<>()">,
                       TypesMatchWith<"infer other type from result type or none",
                                      "result", "other", "$_self",
                                      "($_op.getOperands().size() <= 2) || std::equal_to<>()">]> {
    let summary = "Load from a tensor of pointers or from a tensor pointer";

    let arguments = (ins AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr, Optional<TT_BoolLike>:$mask,
                         Optional<TT_Type>:$other, OptionalAttr<DenseI32ArrayAttr>:$boundaryCheck,
                         OptionalAttr<TT_PaddingOptionAttr>:$padding, TT_CacheModifierAttr:$cache,
                         TT_EvictionPolicyAttr:$evict, BoolAttr:$isVolatile);

    let results = (outs TT_Type:$result);

    let builders = [
        // A tensor of pointers or a pointer to a scalar
        OpBuilder<(ins "Value":$ptr, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A tensor pointer with boundary check and padding
        OpBuilder<(ins "Value":$ptr, "ArrayRef<int32_t>":$boundaryCheck,
                       "std::optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A tensor of pointers or a pointer to a scalar with mask
        OpBuilder<(ins "Value":$ptr, "Value":$mask, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A tensor of pointers or a pointer to a scalar with mask and other
        OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
        // A utility function to build the operation with all attributes
        OpBuilder<(ins "Value":$ptr, "Value":$mask, "Value":$other,
                       "std::optional<ArrayRef<int32_t>>":$boundaryCheck,
                       "std::optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>
    ];

    // Format: `tt.load operands attrs : optional(type(ptr)) -> type(result)`
    // We need an extra `optional(type(ptr))` for inferring the tensor pointer type with back compatibility
    let hasCustomAssemblyFormat = 1;

    let hasCanonicalizer = 1;
}

def TT_StoreOp : TT_Op<"store",
                       [SameLoadStoreOperandsShape,
                        SameLoadStoreOperandsEncoding,
                        MemoryEffects<[MemWrite]>,
                        TypesMatchWith<"infer ptr type from value type",
                                       "value", "ptr", "$_self",
                                       "mlir::OpTrait::impl::verifyLoadStorePointerAndValueType">,
                        TypesMatchWith<"infer mask type from value type",
                                       "value", "mask", "getI1SameShape($_self)",
                                       "($_op.getOperands().size() <= 2) || std::equal_to<>()">]> {
    let summary = "Store by a tensor of pointers or by a tensor pointer";

    let arguments = (ins AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr, TT_Type:$value, Optional<TT_BoolLike>:$mask,
                         OptionalAttr<DenseI32ArrayAttr>:$boundaryCheck,
                         DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
                         DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict);

    let builders = [
        // A tensor of pointers or a pointer to a scalar
        OpBuilder<(ins "Value":$ptr, "Value":$value, "triton::CacheModifier":$cache, "triton::EvictionPolicy":$evict)>,
        // A tensor of pointers or a pointer to a scalar with mask
        OpBuilder<(ins "Value":$ptr, "Value":$value, "Value":$mask, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict)>,
        // A tensor pointer with boundary check
        OpBuilder<(ins "Value":$ptr, "Value":$value, "ArrayRef<int32_t>":$boundaryCheck, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict)>
    ];

    // Format: `tt.store operands attrs : optional(type(ptr)), type(val)
    // We need an extra `optional(type(ptr))` for inferring the tensor pointer type with back compatibility
    let hasCustomAssemblyFormat = 1;

    let hasCanonicalizer = 1;
}

//
// Atomic Ops
//
def TT_AtomicRMWOp : TT_Op<"atomic_rmw", [SameOperandsAndResultShape,
                                          SameOperandsAndResultEncoding,
                                          MemoryEffects<[MemRead]>,
                                          MemoryEffects<[MemWrite]>,
                                          TypesMatchWith<"infer ptr type from value type",
                                                         "val", "ptr",
                                                         "getPointerTypeSameShape($_self)">,
                                          TypesMatchWith<"infer mask type from value type",
                                                         "val", "mask", "getI1SameShape($_self)",
                                                       "($_op.getOperands().size() <= 2) || std::equal_to<>()">]> {
    let summary = "atomic rmw";

    let description = [{
        load data at $ptr, do $rmw_op with $val, and store result to $ptr.

        return old value at $ptr
    }];

    let arguments = (ins TT_AtomicRMWAttr:$atomic_rmw_op, TT_PtrLike:$ptr,
                         TT_Type:$val, Optional<TT_BoolLike>:$mask,
                         TT_MemSemanticAttr:$sem);

    let results = (outs TT_Type:$result);
}

def TT_AtomicCASOp : TT_Op<"atomic_cas", [MemoryEffects<[MemRead]>,
                                          MemoryEffects<[MemWrite]>,
                                          SameOperandsAndResultShape,
                                          SameOperandsAndResultEncoding]> {
    let summary = "atomic cas";

    let description = [{
        compare $cmp with data $old at location $ptr,

        if $old == $cmp, store $val to $ptr,

        else store $old to $ptr,

        return $old
    }];

    let arguments = (ins TT_PtrLike:$ptr, TT_Type:$cmp, TT_Type:$val,
                     TT_MemSemanticAttr:$sem);

    let results = (outs TT_Type:$result);
}

def TT_AtomicLoadOp : TT_Op<"atomic_load",
                            [MemoryEffects<[MemRead, MemWrite]>,
                             TypesMatchWith<"infer ptr type from result type",
                                            "result", "ptr", "$_self",
                                            "mlir::OpTrait::impl::verifyLoadStorePointerAndValueType">]> {
    let summary = "atomic load";

    let description = [{
        Load a scalar from global memory, synchronized with any other atomic accesses to that value.
    }];
    let arguments = (ins TT_Ptr:$ptr, TT_MemSemanticAttr:$sem);
    let results = (outs TT_Scalar:$result);
}

def TT_AtomicStoreOp : TT_Op<"atomic_store",
                             [MemoryEffects<[MemWrite]>,
                              TypesMatchWith<"infer ptr type from value type",
                                             "value", "ptr", "$_self",
                                             "mlir::OpTrait::impl::verifyLoadStorePointerAndValueType">]> {
    let summary = "atomic store";

    let description = [{
        Store a scalar to global memory, synchronized with any other atomic accesses to that value.
    }];
    let arguments = (ins TT_Ptr:$ptr, TT_Scalar:$value, TT_MemSemanticAttr:$sem);
}

//
// Shape Manipulation Ops
//
def TT_SplatOp : TT_Op<"splat", [Pure,
                                 SameOperandsAndResultElementType,
                                 SameOperandsAndResultEncoding]> {
    let summary = "splat";

    let arguments = (ins TT_Type:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasFolder = 1;
}

def TT_ExpandDimsOp : TT_Op<"expand_dims", [Pure,
                                            DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                            SameOperandsAndResultElementType]> {
    let summary = "expand_dims";

    let arguments = (ins TT_Tensor:$src, I32Attr:$axis);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

// view is not `pure` because it may reorder elements
def TT_ViewOp : TT_Op<"view", [NoMemoryEffect,
                               SameOperandsAndResultElementType]> {
    let summary = "view";

    let arguments = (ins TT_Tensor:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

def TT_BroadcastOp : TT_Op<"broadcast", [Pure,
                                         SameOperandsAndResultElementType,
                                         SameOperandsAndResultEncoding]> {
    let summary = "broadcast. No left-padding as of now.";

    let arguments = (ins TT_Type:$src);

    let results = (outs TT_Type:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";

    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

// cat is not `pure` because it may reorder elements
def TT_CatOp : TT_Op<"cat", [NoMemoryEffect,
                             SameOperandsAndResultElementType]> {
    let summary = "concatenate 2 tensors";

    let arguments = (ins TT_Tensor:$lhs, TT_Tensor:$rhs);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def TT_TransOp : TT_Op<"trans", [Pure,
                                 DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                 SameOperandsAndResultElementType]> {

    let summary = "transpose a tensor";

    let arguments = (ins TT_Tensor:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

//
// SPMD Ops
//
def TT_GetProgramIdOp : TT_Op<"get_program_id", [Pure]> {
    let arguments = (ins TT_ProgramDim:$axis);

    let results = (outs I32:$result);

    let assemblyFormat = "$axis attr-dict `:` type($result)";

    let extraClassDeclaration = [{
      int32_t getAxisAsInt() {
        return static_cast<int32_t>(getAxis());
      }
    }];
}

def TT_GetNumProgramsOp : TT_Op<"get_num_programs", [Pure]> {
    let arguments = (ins I32Attr:$axis);

    let results = (outs I32:$result);

    let assemblyFormat = "attr-dict `:` type($result)";
}

//
// Dot Op
//
def TT_DotOp : TT_Op<"dot", [Pure,
                             DeclareOpInterfaceMethods<InferTypeOpInterface>,
                             TypesMatchWith<"result's type matches accumulator's type",
                                            "d", "c", "$_self">]> {
    let summary = "dot";

    let description = [{
        $d = matrix_multiply($a, $b) + $c
    }];

    let arguments = (ins
      TT_FpIntTensor:$a,
      TT_FpIntTensor:$b,
      TT_FpIntTensor:$c,
      BoolAttr:$allowTF32,
      I32Attr:$maxNumImpreciseAcc);

    let results = (outs TT_FpIntTensor:$d);

    let assemblyFormat = "$a`,` $b`,` $c attr-dict `:` type($a) `*` type($b) `->` type($d)";
    let hasVerifier = 1;
}

//
// Reduce Op
//
def TT_ReduceOp: TT_Op<"reduce",
                       [Pure,
                        SameOperandsEncoding,
                        SingleBlock,
                        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Reduction using generic combination algorithm";
    let arguments = (ins Variadic<TT_Tensor>:$operands, I32Attr:$axis);
    let results = (outs Variadic<TT_Type>:$result);
    let regions = (region SizedRegion<1>:$combineOp);
    let builders = [
        OpBuilder<(ins "ValueRange":$operands, "int":$axis)>,
    ];
    let hasVerifier = 1;
    let hasRegionVerifier = 1;
    let extraClassDeclaration = [{
      llvm::SmallVector<RankedTensorType> getInputTypes();
      llvm::SmallVector<Type> getElementTypes();
      unsigned getNumOperands();
    }];
}

def TT_ReduceReturnOp: TT_Op<"reduce.return",
                             [HasParent<"ReduceOp">, Pure, Terminator, ReturnLike]> {
    let summary = "terminator for reduce operator";
    let arguments = (ins Variadic<AnyType>:$result);
    let assemblyFormat = "$result attr-dict `:` type($result)";
}

//
// Scan Op
//
def TT_ScanOp: TT_Op<"scan",
                       [Pure,
                        SameOperandsAndResultEncoding,
                        SameOperandsAndResultElementType,
                        SingleBlock,
                        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Reduction using generic combination algorithm";
    let arguments = (ins Variadic<TT_Tensor>:$operands, I32Attr:$axis);
    let results = (outs Variadic<TT_Tensor>:$result);
    let regions = (region SizedRegion<1>:$combineOp);
    let builders = [
        OpBuilder<(ins "ValueRange":$operands, "int":$axis)>,
    ];
    let hasVerifier = 1;
}

def TT_ScanReturnOp: TT_Op<"scan.return",
                             [HasParent<"ScanOp">, Pure, Terminator, ReturnLike]> {
    let summary = "terminator for scan operator";
    let arguments = (ins Variadic<AnyType>:$result);
    let assemblyFormat = "$result attr-dict `:` type($result)";
}


//
// External Elementwise op
//
def TT_ExternElementwiseOp : TT_Op<"extern_elementwise", [Elementwise,
                                                            SameOperandsAndResultEncoding,
                                                            SameVariadicOperandSize,
                                                            DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

    let description = [{
        call an external function $symbol implemented in $libpath/$libname with $args
        return $libpath/$libname:$symbol($args...)
    }];

    let arguments = (ins Variadic<TT_Type>:$args, StrAttr:$libname, StrAttr:$libpath, StrAttr:$symbol, BoolAttr:$pure);

    let results = (outs TT_Type:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, $result)";
}

//
// Make Range Op
//
// TODO: should have ConstantLike as Trait
def TT_MakeRangeOp : TT_Op<"make_range", [Pure]> {
    let summary = "make range";

    let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

    let arguments = (ins I32Attr:$start, I32Attr:$end);

    let results = (outs TT_IntTensor:$result);

    let assemblyFormat = "attr-dict `:` type($result)";

    let hasFolder = 1;
}

//
// ElementwiseInlineAsm Op
//
def TT_ElementwiseInlineAsmOp : TT_Op<"elementwise_inline_asm", [Elementwise,
                                                                 SameOperandsAndResultEncoding,
                                                                 DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "inline assembly applying elementwise operation to a group of packed element.";
  let description = [{
   This will apply the given in inline assembly to `packed_element` number of
   elements of the inputs. The elements packed together is unknown and will
   depend on the backend implementation.
  }];

  let arguments = (ins StrAttr:$asm_string, StrAttr:$constraints, BoolAttr:$pure, I32Attr:$packed_element, Variadic<AnyTypeOf<[TT_Type]>>:$args);
  let results = (outs TT_Type:$result);


  let assemblyFormat = [{
    $asm_string attr-dict ($args^ `:` type($args))? `->` type($result)
  }];
}

//
// Print Op
//
def TT_PrintOp : TT_Op<"print", [MemoryEffects<[MemWrite]>]>,
  Arguments<(ins StrAttr:$prefix, Variadic<AnyTypeOf<[TT_Type]>>:$args)> {
  let summary = "Device-side print, as in CUDA for debugging";
  let description = [{
    `tt.print` takes a literal string prefix and an arbitrary number of scalar or tensor arguments that should be printed.
    format are generated automatically from the arguments.
  }];
  let assemblyFormat = [{
    $prefix attr-dict (`:` $args^ `:` type($args))?
  }];
}

//
// Assert Op
//
def TT_AssertOp : TT_Op<"assert", [MemoryEffects<[MemWrite]>]> {
  let summary = "Device-side assert, as in CUDA for correctness checking";
  let description = [{
    `tt.assert` takes a condition tensor, a message string, a file string, a function string, and a line number.
    If the condition is false, the message is printed, and the program is aborted.
  }];
  let arguments = (ins TT_Tensor:$condition, StrAttr:$message, StrAttr:$file, StrAttr:$func, I32Attr:$line);
  let assemblyFormat = "$condition `,` $message `,` $file `,` $func `,` $line attr-dict `:` type($condition)";
}

//
// Make Tensor Pointer Op
//
def TT_MakeTensorPtrOp : TT_Op<"make_tensor_ptr",
                               [Pure,
                                SameVariadicOperandSize,
                                TypesMatchWith<"infer pointer type from the result type",
                                               "result", "base",
                                               "getPointerType(getElementTypeOfTensorPointerType($_self))">]> {
  let summary = "Make a tensor pointer type with meta information of the parent tensor and the block specified";

  let description = [{
      `tt.make_tensor_ptr` takes both meta information of the parent tensor and the block tensor, then it returns a
      pointer to the block tensor, e.g. returns a type of `tt.ptr<tensor<8x8xf16>>`.
  }];

  // TODO(Chenggang): unify the integer types. Currently we cannot do that due to hardware constraints.
  let arguments = (ins
    TT_Ptr:$base,
    Variadic<I64>:$shape,
    Variadic<I64>:$strides,
    Variadic<I32>:$offsets,
    DenseI32ArrayAttr:$order
  );

  let results = (outs TT_TensorPtr:$result);

  // TODO(Keren): define a custom assembly format for this op because the result type cannot be printed correctly
  // Add additional `[]` to increase readability and split variadic lists
  let assemblyFormat = "$base `,` `[` $shape `]` `,` `[` $strides `]` `,` `[` $offsets `]` attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins
        "Value":$base,
        "ValueRange":$shape,
        "ValueRange":$strides,
        "ValueRange":$offsets,
        "ArrayRef<int32_t>":$tensorShape,
        "ArrayRef<int32_t>":$order
    )>
  ];
}

// The following ops, including `call`, `func`, and `return` are copied and modified from
// https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Func/IR/FuncOps.td
// We could revert it back once MLIR has a better inliner interface.
//
// Function Ops
//
def CallOp : TT_Op<"call", [CallOpInterface, /*MemRefsNormalizable, */DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `tt.call` operation represents a direct call to a function that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = tt.call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType() {
      return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def FuncOp : TT_Op<"func", [AffineScope, AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external function declaration (used when
    referring to a function declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for function arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

    Example:

    ```mlir
    // External function definitions.
    tt.func @abort()
    tt.func @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

    // A function that returns its argument twice:
    tt.func @count(%x: i64) -> (i64, i64)
      attributes {fruit: "banana"} {
      return %x, %x: i64, i64
    }

    // A function with an argument attribute
    tt.func @example_fn_arg(%x: i32 {swift.self = unit})

    // A function with a result attribute
    tt.func @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

    // A function with an attribute
    tt.func @example_fn_attr() attributes {dialectName.attrName = false}
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

def ReturnOp : TT_Op<"return", [Pure, HasParent<"FuncOp">, /*MemRefsNormalizable, */ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `tt.return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    tt.func @foo() : (i32, f8) {
      ...
      tt.return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

#endif // Triton_OPS
