// RUN: triton-opt %s -split-input-file -verify-diagnostics

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf32, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{element types of operands A and B must have same bit width}}
    %D = tt.dot %A, %B, %C {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} :
        tensor<16x16xf32, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=1}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{mismatching encoding between A and B operands}}
    %D = tt.dot %A, %B, %C {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} :
        tensor<16x16xf16> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=1}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{mismatching kWidth between A and B operands}}
    %D = tt.dot %A, %B, %C {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} :
        tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}
