// Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRITONNVIDIAGPU_OPS
#define TRITONNVIDIAGPU_OPS

include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUTypes.td"
include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonTypeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class TTNG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonNvidiaGPU_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def TTNG_FenceAsyncSharedOp : TTNG_Op<"fence_async_shared"> {
  let arguments = (ins BoolAttr:$bCluster);

  let summary = "fence proxy async";

  let assemblyFormat = "attr-dict";

  let extraClassDeclaration = [{
    static bool isSupported(int computeCapability) {
      return computeCapability >= 90;
    }
  }];
}

def TTNG_GetClusterCTAIdOp : TTNG_Op<"get_cluster_cta_id", [Pure]> {
  let description = [{
    Returns the one dimensional cluster_cta_id.
  }];

  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTNG_ClusterArriveOp : TTNG_Op<"cluster_arrive", []> {
  let arguments = (ins I1Attr:$relaxed);
  let assemblyFormat = "attr-dict";
}

def TTNG_ClusterWaitOp : TTNG_Op<"cluster_wait", []> {
  let assemblyFormat = "attr-dict";
}

//
// DotAsync Op
//
def TTNG_DotAsyncOp : TTNG_Op<"dot_async", [Pure,
                             DeclareOpInterfaceMethods<InferTypeOpInterface>,
                             TypesMatchWith<"result's type matches accumulator's type",
                                            "d", "c", "$_self">]> {
    let summary = "dot async";

    let description = [{
        $d = matrix_multiply($a, $b) + $c. For docs on InputPrecisionAttr, see TT_DotOp
    }];

    let arguments = (ins TT_TensorOrMemDesc:$a,
                         TT_TensorOrMemDesc:$b,
                         TT_FpIntTensor:$c,
                         TT_InputPrecisionAttr:$inputPrecision,
                         I32Attr:$maxNumImpreciseAcc);

    let results = (outs TT_FpIntTensor:$d);

    let assemblyFormat = "$a`,` $b`,` $c attr-dict `:` type($a) `*` type($b) `->` type($d)";
}

def TTNG_DotWaitOp : TTNG_Op<"dot_wait", [DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                          AllTypesMatch<["inputs", "outputs"]>]> {
  let summary = "dot wait";
  let arguments = (ins Variadic<TT_TensorOrMemDesc>:$inputs, I32Attr:$pendings);
  let results = (outs Variadic<TT_TensorOrMemDesc>:$outputs);
  let description = [{
    Waits until there are $pendings or fewer outstanding async dot operations.

    $inputs must be the tensors corresponding to the async dot ops that we're
    waiting on.  For example, if there are N pending async dot ops and we call
    `dot_wait 1`, then $inputs must be the result of the first dot op.
  }];

  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

#endif
