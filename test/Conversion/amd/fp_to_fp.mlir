// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s
// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s

//  CHECK-LABEL: f16_to_f32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @f16_to_f32(%arg0: tensor<8x8xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) {
    // CHECK-COUNT-8: llvm.inline_asm asm_dialect {{.*}}v_cvt_f32_f16 {{.*}}: (f16) -> f32
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<8x8xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: bf16_to_f32
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @bf16_to_f32(%arg0: tensor<8x8xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>>) {
    // CHECK-COUNT-8: llvm.bitcast
    %0 = tt.fp_to_fp %arg0 : tensor<8x8xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<8x8xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>>
    tt.return
  }
}
