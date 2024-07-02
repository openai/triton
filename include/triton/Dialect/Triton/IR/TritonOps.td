#ifndef TRITON_OPS
#define TRITON_OPS

include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/Interfaces/CastInterfaces.td" // CastOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // BranchOpInterface
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/CastInterfaces.td" // CastOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "triton/Dialect/Triton/IR/TritonTypeInterfaces.td"


//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

//
// Op Base
//
class TT_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Dialect, mnemonic,
       !listconcat(traits, [TensorSizeTrait, VerifyTensorLayoutsTrait])> {
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

    let arguments = (ins TT_I64Like:$src);

    let results = (outs TT_PtrLike:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TT_PtrToIntOp : TT_Op<"ptr_to_int", [Elementwise,
                                         SameOperandsAndResultShape,
                                         SameOperandsAndResultEncoding,
                                         Pure,
                                         /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast pointer to int64";

    let arguments = (ins TT_PtrLike:$src);

    let results = (outs TT_I64Like:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

// arith.bitcast doesn't support pointers
def TT_BitcastOp : TT_Op<"bitcast", [Elementwise,
                                     SameOperandsAndResultShape,
                                     SameOperandsAndResultEncoding,
                                     Pure,
                                     /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Cast between types of the same bitwidth";

    let arguments = (ins TT_Type:$src);

    let results = (outs TT_Type:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

    // TODO: Add verifier
}

def TT_FpToFpOp : TT_Op<"fp_to_fp", [SameOperandsAndResultShape,
                                     SameOperandsAndResultEncoding,
                                     Pure,
                                     /*DeclareOpInterfaceMethods<CastOpInterface>*/]> {
    let summary = "Floating point casting for custom types";

    let description = [{
        Floating point casting for custom types (F8), and non-default rounding modes.

        F8 <-> FP16, BF16, FP32, FP64
    }];

    let arguments = (
      ins TT_FloatTensor:$src,
      OptionalAttr<TT_RoundingModeAttr>:$rounding
    );

    let results = (outs TT_FloatTensor:$result);

    let assemblyFormat = "$src attr-dict  (`,` `rounding` `=` $rounding^)? `:` type($src) `->` type($result)";

    let hasVerifier = 1;
}

//
// Arithmetic Ops
//

def TT_ClampFOp : TT_Op<"clampf", [Elementwise,
                                  SameOperandsAndResultType,
                                  Pure]> {
    let summary = "Clamp operation for floating point types";

    let description = [{
        Clamp operation for floating point types.

        The operation takes three arguments: x, min, and max. It returns a tensor of the same shape as x with its values clamped to the range [min, max].
    }];

    let arguments = (
      ins
      TT_FloatLike:$x,
      TT_FloatLike:$min,
      TT_FloatLike:$max,
      TT_PropagateNanAttr:$propagateNan
    );

    let results = (outs TT_FloatLike:$result);

    // List $propagateNan explicitly rather than relying on attr-dict to pick it
    // up, because if it's inside attr-dict, its value will be printed as a
    // number rather than as a meaningful string.
    let assemblyFormat = "$x `,` $min `,` $max `,` `propagateNan` `=` $propagateNan attr-dict `:` type($result)";
}

//
// Math Ops
//

def TT_PreciseSqrtOp : TT_Op<"precise_sqrt", [Elementwise,
                                  SameOperandsAndResultType,
                                  Pure]> {
    let summary = "Precise sqrt for floating point types";

    let description = [{
        Precise sqrt for floating point types.
    }];

    let arguments = (ins TT_FloatLike:$x);

    let results = (outs TT_FloatLike:$result);

    let assemblyFormat = "$x attr-dict `:` type($x)";
}

def TT_PreciseDivFOp : TT_Op<"precise_divf", [Elementwise,
                                  SameOperandsAndResultType,
                                  Pure]> {
    let summary = "Precise div for floating point types";

    let description = [{
        Precise div for floating point types.
    }];

    let arguments = (ins TT_FloatLike:$x, TT_FloatLike:$y);

    let results = (outs TT_FloatLike:$result);

    let assemblyFormat = "$x `,` $y attr-dict `:` type($x)";
}

def TT_MulhiUIOp : TT_Op<"mulhiui", [Elementwise,
                                  SameOperandsAndResultType,
                                  Pure]> {
    let summary = "Most significant N bits of the 2N-bit product of two integers";

    let description = [{
        Most significant N bits of the 2N-bit product of two integers.
    }];

    let arguments = (ins TT_IntLike:$x, TT_IntLike:$y);

    let results = (outs TT_IntLike:$result);

    let assemblyFormat = "$x `,` $y attr-dict `:` type($x)";
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

    let hasFolder = 1;
}

//
// Load/Store Ops
//
def TT_LoadOp : TT_Op<"load", [
  SameLoadStoreOperandsAndResultShape,
  SameLoadStoreOperandsAndResultEncoding,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  TypesMatchWith<"result matches ptr type", "ptr", "result", "getPointeeType($_self)">,
  TypesMatchWith<"mask type matches ptr type", "ptr", "mask", "getI1SameShape(getPointeeType($_self))",
                 "($_op.getOperands().size() <= 1) || std::equal_to<>()">,
  TypesMatchWith<"other matches ptr type", "ptr", "other", "getPointeeType($_self)",
                 "($_op.getOperands().size() <= 2) || std::equal_to<>()">
]> {
    let summary = "Load from a tensor of pointers or from a tensor pointer";

    let arguments = (
      ins
      AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr,
      Optional<TT_BoolLike>:$mask,
      Optional<TT_Type>:$other,

      DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
      OptionalAttr<TT_PaddingOptionAttr>:$padding,
      DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
      DefaultValuedAttr<TT_EvictionPolicyAttr, "::mlir::triton::EvictionPolicy::NORMAL">:$evict,
      DefaultValuedAttr<BoolAttr, "false">:$isVolatile
    );

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
                       "ArrayRef<int32_t>":$boundaryCheck,
                       "std::optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>
    ];

    // Specify `cacheModifier` and `evictionPolicy` explicitly in the
    // assemblyFormat instead of as part of attr-dict so that they get printed
    // as strings rather than opaque integers.
    //
    // Note there's no comma between `other` and `cacheModifier` and between
    // `cacheModifier` and `evictionPolicy`.  This is due to an apparent
    // limitation in the MLIR custom-format parser.  In oilist, the initial
    // keywords of each clause have to be unique, so they can't be `,`.
    //
    // Even if we gave up on order-independence and used vanilla optional
    // clauses, the format (`,` `foo` `=` $foo^)? (`,` `bar` `=` $bar^)?  will
    // not match the string ", bar = 0" because after the initial comma (first
    // token of the first optional clause) we expect to see "foo".
    let assemblyFormat = [{
      $ptr (`,` $mask^)? (`,` $other^)?
      oilist(
        `cacheModifier` `=` $cache |
        `evictionPolicy` `=` $evict
      )
      attr-dict `:` type($ptr)
    }];

    let hasCanonicalizer = 1;
}

def TT_StoreOp : TT_Op<"store", [
  SameLoadStoreOperandsShape,
  SameLoadStoreOperandsEncoding,
  MemoryEffects<[MemWrite<GlobalMemory>]>,
  TypesMatchWith<"value type matches ptr type", "ptr", "value",
                 "getPointeeType($_self)">,
  TypesMatchWith<"mask type matches ptr type", "ptr", "mask",
                 "getI1SameShape(getPointeeType($_self))",
                 "($_op.getOperands().size() <= 2) || std::equal_to<>()">
]> {
    let summary = "Store by a tensor of pointers or by a tensor pointer";

    let arguments = (
      ins
      AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr,
      TT_Type:$value,
      Optional<TT_BoolLike>:$mask,
      DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
      DefaultValuedAttr<TT_CacheModifierAttr, "triton::CacheModifier::NONE">:$cache,
      DefaultValuedAttr<TT_EvictionPolicyAttr, "triton::EvictionPolicy::NORMAL">:$evict
    );

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

    // Specify cacheModifier and evictionPolicy explicitly, instead of leaving
    // them in attr-dict, because this way their values get printed as strings,
    // rather than as opaque integers.
    //
    // Note there are no commas between mask, cacheModifier, and evictionPolicy,
    // due to limitations in MLIR's asm parser.
    let assemblyFormat = [{
      $ptr `,` $value (`,` $mask^)?
      oilist(`cacheModifier` `=` $cache | `evictionPolicy` `=` $evict)
      attr-dict `:` type($ptr)
    }];

    let hasCanonicalizer = 1;
}

//
// Atomic Ops
//
def TT_AtomicRMWOp : TT_Op<"atomic_rmw", [
  SameOperandsAndResultShape,
  SameOperandsAndResultEncoding,
  MemoryEffects<[MemRead<GlobalMemory>]>,
  MemoryEffects<[MemWrite<GlobalMemory>]>,
  TypesMatchWith<"ptr type matches value type", "val", "ptr",
                 "getPointerTypeSameShape($_self)">,
  TypesMatchWith<"mask type matches value type",
                 "val", "mask", "getI1SameShape($_self)",
                 "($_op.getOperands().size() <= 2) || std::equal_to<>()">
]> {
    let summary = "atomic rmw";

    let description = [{
        load data at $ptr, do $rmw_op with $val, and store result to $ptr.

        return old value at $ptr
    }];

    let arguments = (ins TT_AtomicRMWAttr:$atomic_rmw_op, TT_PtrLike:$ptr,
                         TT_Type:$val, Optional<TT_BoolLike>:$mask,
                         TT_MemSemanticAttr:$sem, TT_MemSyncScopeAttr:$scope);

    let results = (outs TT_Type:$result);

    // Explicitly list $atomic_rmw_op, $sem, and $scope rather than relying on
    // attr-dict so they're printed as strings rather than opaque integers.
    let assemblyFormat = [{
      $atomic_rmw_op `,` $sem `,` $scope `,` $ptr `,` $val (`,` $mask^)?  attr-dict `:`
      functional-type(operands, $result)
    }];
}

def TT_AtomicCASOp : TT_Op<"atomic_cas", [MemoryEffects<[MemRead<GlobalMemory>]>,
                                          MemoryEffects<[MemWrite<GlobalMemory>]>,
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
                     TT_MemSemanticAttr:$sem, TT_MemSyncScopeAttr:$scope);

    let results = (outs TT_Type:$result);

    // Explicitly list $sem and $scope rather than relying on attr-dict so
    // they're printed as strings rather than opaque integers.
    let assemblyFormat = [{
      $sem `,` $scope `,` $ptr `,` $cmp `,` $val attr-dict `:`
      functional-type(operands, $result)
     }];
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

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

    let hasFolder = 1;
}

def TT_ExpandDimsOp : TT_Op<"expand_dims", [Pure,
                                            DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                            SameOperandsAndResultElementType]> {
    let summary = "expand_dims";

    let arguments = (ins TT_Tensor:$src, I32Attr:$axis);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

def TT_ReshapeOp : TT_Op<"reshape", [Pure,
                                     SameOperandsAndResultElementType]> {
    let summary = "reinterpret a tensor to a different shape. It may change elements order if the attribute is set.";
    let description = [{
        reinterpret a tensor to a different shape.

        If allow_reorder is set the compiler is free to change the order of
        elements to generate more efficient code.

        If efficient_layout is set, this is a hint that the destination layout should be kept for performance reason.
        The compiler is still free to change it for better performance.
    }];
    let arguments = (ins TT_Tensor:$src, BoolAttr:$allow_reorder, OptionalAttr<UnitAttr>:$efficient_layout);
    let results = (outs TT_Tensor:$result);
    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
    let builders = [
      OpBuilder<(ins "Type":$type, "Value":$src, "bool":$allow_reorder),
        [{
        build($_builder, $_state, type, src, allow_reorder, /*efficient_layout=*/UnitAttr());
        }]>];
}

def TT_BroadcastOp : TT_Op<"broadcast", [Pure,
                                         SameOperandsAndResultElementType,
                                         SameOperandsAndResultEncoding]> {
    let summary = "broadcast a tensor";

    let description = [{
      For a given tensor, broadcast changes one or more dimensions with size 1
      to a new size, e.g. tensor<1x32x1xf32> -> tensor<2x32x4xf32>.  You cannot
      change the size of a non-1 dimension.
    }];

    let arguments = (ins TT_Tensor:$src);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

    let hasCanonicalizeMethod = 1;
    let hasFolder = 1;
}

// cat is not `pure` because it may reorder elements
def TT_CatOp : TT_Op<"cat", [NoMemoryEffect,
                             SameTypeOperands,
                             SameOperandsAndResultElementType]> {
    let summary = "concatenate 2 tensors";

    let arguments = (ins TT_Tensor:$lhs, TT_Tensor:$rhs);

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)";
}

def TT_JoinOp : TT_Op<"join", [
    NoMemoryEffect, SameTypeOperands,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
    let summary = "join two tensors along a new, minor dimension";
    let description = [{
        For example, if the two input tensors are 4x8xf32, returns a tensor of
        shape 4x8x2xf32.

        Because Triton tensors always have a power-of-two number of elements,
        the two input tensors must have the same shape.
    }];

    let arguments = (ins TT_Tensor:$lhs, TT_Tensor:$rhs);
    let results = (outs TT_Tensor:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)";
}

def TT_SplitOp : TT_Op<"split", [
  NoMemoryEffect,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  TypesMatchWith<"outLHS and outRHS types match",
                  "outLHS", "outRHS", "$_self">,
]> {
    let summary = "splits a tensor into two, along its last dimension";
    let description = [{
        The input must be a tensor whose last dimension has size 2.  Returns two
        tensors, src[..., 0] and src[..., 1].

        For example, if the input shape is 4x8x2xf32, returns two tensors of
        shape 4x8xf32.
    }];

    let arguments = (ins TT_Tensor:$src);
    let results = (outs TT_Tensor:$outLHS, TT_Tensor:$outRHS);
    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($outLHS)";
}

def TT_TransOp : TT_Op<"trans", [Pure,
                                 DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                 SameOperandsAndResultElementType]> {

    let summary = "rearrange the dimensions of a tensor";
    let description = [{
      For example, given a tensor x with shape [1,2,4], transpose(x) with
      order=[2,0,1] rearranges the tensor to have shape [4,1,2].

      Although this op is called "trans", it implements both tl.trans() and
      tl.permute().  ("permute" might be a better name, but it's called "trans"
      because originally it only supported 2D tensors.)

      ## Implementation note on encodings:

      In the TritonGPU dialect (and probably others), an encoding is chosen for
      this op's output so it's a nop from the perspective of code generation.

      For example, suppose tensor x has an encoding such that GPU thread [i,j,k]
      has a register containing element [i,j,k] of the tensor.  Now we transpose
      x with order [2,1,0], i.e. we reverse the order of its dimensions.  In
      TritonGPU, we will choose a layout for the output of the transpose so that
      GPU thread [i,j,k] has element [k,j,i] of transpose(x).  But this is the
      same element it had before!  All we've done is "rename" the element that
      thread [i,j,k] has.

      The "real" transpose -- i.e. moving data between GPU threads -- occurs in
      convertLayout ops that appear before and/or after the operation.

      We do this so that you can chain multiple data-movement ops (e.g.
      transpose+reshape+concat) without going to shared memory after each one.
    }];

    let arguments = (
      ins TT_TensorOrMemDesc:$src,
      DenseI32ArrayAttr:$order
    );

    let results = (outs TT_TensorOrMemDesc:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

    let hasFolder = 1;
    let hasVerifier = 1;
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
    let arguments = (ins TT_ProgramDim:$axis);

    let results = (outs I32:$result);

    let assemblyFormat = "$axis attr-dict `:` type($result)";

    let extraClassDeclaration = [{
      int32_t getAxisAsInt() {
        return static_cast<int32_t>(getAxis());
      }
    }];
}

//
// Dot Op
//
def TT_DotOp : TT_Op<"dot", [Pure,
                             DeclareOpInterfaceMethods<InferTypeOpInterface>,
                             DotLike,
                             TypesMatchWith<"result's type matches accumulator's type",
                                            "d", "c", "$_self">]> {
    let summary = "dot";

    let description = [{
        $d = matrix_multiply($a, $b) + $c. $inputPrecision describes how to exercise the TC
        when the inputs are f32. It can be one of: tf32, tf32x3, ieee.
        tf32: use TC with tf32 ops.
        tf32x3: implement the 3xTF32 trick. For more info see the pass in F32DotTC.cpp
        ieee: don't use TC, implement dot in software.
        If the GPU does not have Tensor cores or the inputs are not f32, this flag is ignored.
    }];

    let arguments = (
      ins
      TT_FpIntTensor:$a,
      TT_FpIntTensor:$b,
      TT_FpIntTensor:$c,
      DefaultValuedAttr<TT_InputPrecisionAttr, "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
      DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc
    );

    let results = (outs TT_FpIntTensor:$d);

    // attr-dict prints enums as integers.  To get inputPrecision printed as a
    // string, we need to specify it explicitly.
    let assemblyFormat = [{
      $a`,` $b`,` $c (`,` `inputPrecision` `=` $inputPrecision^)? attr-dict `:`
      type($a) `*` type($b) `->` type($d)
    }];
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
    let arguments = (ins Variadic<TT_Tensor>:$srcs, I32Attr:$axis);
    let results = (outs Variadic<TT_Type>:$result);
    let regions = (region SizedRegion<1>:$combineOp);
    let builders = [
        OpBuilder<(ins "ValueRange":$srcs, "int":$axis)>,
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
                        SameOperandsAndResultShape,
                        SingleBlock,
                        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Associative scan using generic combination algorithm";
    let arguments = (ins Variadic<TT_Tensor>:$srcs, I32Attr:$axis, BoolAttr:$reverse);
    let results = (outs Variadic<TT_Tensor>:$result);
    let regions = (region SizedRegion<1>:$combineOp);
    let builders = [
        OpBuilder<(ins "ValueRange":$srcs, "int":$axis, "bool":$reverse)>,
    ];
    let hasVerifier = 1;
    let hasRegionVerifier = 1;
    let extraClassDeclaration = [{
      llvm::SmallVector<RankedTensorType> getInputTypes();
      llvm::SmallVector<Type> getElementTypes();
      unsigned getNumOperands();
    }];
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

    let arguments = (ins Variadic<TT_Type>:$srcs, StrAttr:$libname, StrAttr:$libpath, StrAttr:$symbol, BoolAttr:$pure);

    let results = (outs TT_Type:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, $result)";
}

//
// Make Range Op
//
def TT_MakeRangeOp : TT_Op<"make_range", [Pure]> {
    let summary = "make range";

    let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

    // WARNING: MLIR generates getStart()/getEnd() functions which return
    // uint32_t, even though these arguments are to be interpreted as *signed*
    // int32 values.  If this matters, use get{Start,End}Attr().getInt(), which
    // return int64_t.
    let arguments = (ins I32Attr:$start, I32Attr:$end);

    let results = (outs TT_IntTensor:$result);

    let assemblyFormat = "attr-dict `:` type($result)";

    let hasFolder = 1;
    let hasVerifier = 1;
}

//
// ElementwiseInlineAsm Op
//
def TT_ElementwiseInlineAsmOp : TT_Op<"elementwise_inline_asm", [
  Elementwise,
  SameOperandsAndResultEncoding,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "inline assembly applying an elementwise operation to a group of packed elements.";
  let description = [{
    Runs an inline asm block to generate one or more tensors.

    The asm block is given `packed_element` elements at a time.  Exactly which
    elems it receives is unspecified.
  }];

  let arguments = (ins StrAttr:$asm_string, StrAttr:$constraints, BoolAttr:$pure, I32Attr:$packed_element, Variadic<AnyTypeOf<[TT_Type]>>:$args);
  let results = (outs Variadic<TT_Type>:$result);

  let assemblyFormat = [{
    $asm_string attr-dict ($args^ `:` type($args))? `->` type($result)
  }];

  let hasVerifier = 1;
}

//
// Histogram Op
//
def TT_HistogramOp : TT_Op<"histogram", [Pure]> {
  let summary = "return a histgram of the inputs.";
  let description = [{
    Return the histogram of the input tensor. The number of bins is equal to
    the dimension of the output tensor. Each bins has a width of 1 and bins
    start at 0.
  }];

  let arguments = (ins TT_IntTensor:$src);
  let results = (outs TT_IntTensor:$result);

  let assemblyFormat = [{
    $src attr-dict `:` type($src) `->` type($result)
  }];
}

//
// Print Op
//
def TT_PrintOp : TT_Op<"print", [MemoryEffects<[MemWrite<GlobalMemory>]>]>,
  Arguments<(ins StrAttr:$prefix, BoolAttr:$hex, Variadic<AnyTypeOf<[TT_Type]>>:$args)> {
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
def TT_AssertOp : TT_Op<"assert", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
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

    // Required by CallOpInterface.
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
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

  let arguments = (ins Variadic<AnyType>:$srcs);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($srcs^ `:` type($srcs))?";
  let hasVerifier = 1;
}


def TT_ExperimentalDescriptorLoadOp : TT_Op<"experimental_descriptor_load", [
  MemoryEffects<[MemRead<GlobalMemory>]>]> {
    let summary = "Load from descriptor";
    let description = [{
      This operation will be lowered to Nvidia TMA load operation on targets supporting it.
      `desc_ptr` is a pointer to the TMA descriptor allocated in global memory.
      The destination tensor type and shape must match the descriptor otherwise the result is undefined.

      This is an escape hatch and is only there for testing/experimenting.
      This op will be removed in the future.
    }];
    let arguments = (
      ins
      TT_PtrType:$desc_ptr,
      Variadic<I32>:$indices,
      DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
      DefaultValuedAttr<TT_EvictionPolicyAttr, "::mlir::triton::EvictionPolicy::NORMAL">:$evict
    );

    let results = (outs TT_Tensor:$result);

    let assemblyFormat = [{
      $desc_ptr `[` $indices `]`
      oilist(
        `cacheModifier` `=` $cache |
        `evictionPolicy` `=` $evict
      )
      attr-dict `:` qualified(type($desc_ptr)) `->` type($result)
    }];
}

def TT_ExperimentalDescriptorStoreOp : TT_Op<"experimental_descriptor_store", [
  MemoryEffects<[MemWrite<GlobalMemory>]>]> {
    let summary = "store value based on descriptor";
    let description = [{
      This operation will be lowered to Nvidia TMA store operation on targets supporting it.
      `desc_ptr` is a pointer to the TMA descriptor allocated in global memory.
      The shape and types of `src` must match the descriptor otherwise the result is undefined.

      This is an escape hatch and is only there for testing/experimenting.
      This op will be removed in the future.
    }];
    let arguments = (
      ins
      TT_PtrType:$desc_ptr,
      TT_Tensor:$src,
      Variadic<I32>:$indices
    );

    let assemblyFormat = [{
      $desc_ptr `[` $indices `]` `,` $src
      attr-dict `:` qualified(type($desc_ptr)) `,` type($src)
    }];
}

#endif // Triton_OPS
