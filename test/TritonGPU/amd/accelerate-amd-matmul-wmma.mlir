// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx1100 matrix-instruction-size=0' | FileCheck %s

// CHECK: #[[DOT_OP_PARENT:.+]] = #triton_gpu.blocked<{{.*}}>
// CHECK: #[[WMAA_0:.+]] = #triton_gpu.amd_wmma<{warpsPerCTA = [1, 4]}>
// CHECK: #[[WMAA_1:.+]] = #triton_gpu.amd_wmma<{warpsPerCTA = [2, 2]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_cf32(
   // CHECK: %[[DOT0_ARG_A:.+]]: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DOT_OP_PARENT]]}>>
   %0: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
   // CHECK-SAME: %[[DOT0_ARG_B:.+]]: tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DOT_OP_PARENT]]}>>
   %1: tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
   %2: tensor<128x256x!tt.ptr<f32, 1>, #blocked>) {
    // CHECK: %[[DOT0_ARG_C:.+]] = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #[[DOT_OP_PARENT]]>
    // CHECK: %[[DOT0_OP_C:.+]] = triton_gpu.convert_layout %[[DOT0_ARG_C]]
    // CHECK-SAME: -> tensor<128x256xf32, #[[WMAA_0]]
    %3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // CHECK: %[[DOT0_OP_A:.+]] = triton_gpu.convert_layout %[[DOT0_ARG_A]]
    // CHECK-SAME: -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[WMAA_0]]
    // CHECK: %[[DOT0_OP_B:.+]] = triton_gpu.convert_layout %[[DOT0_ARG_B]]
    // CHECK-SAME: -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[WMAA_0]]
    // CHECK: %[[DOT0_WMMA_RES:.+]] = tt.dot %[[DOT0_OP_A]], %[[DOT0_OP_B]], %[[DOT0_OP_C]]
    // CHECK-SAME: -> tensor<128x256xf32, #[[WMAA_0]]
    %4 = tt.dot %0, %1, %3 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    // CHECK: triton_gpu.convert_layout %[[DOT0_WMMA_RES]]
    // CHECK-SAME: -> tensor<128x256xf32, #[[DOT_OP_PARENT]]>
    tt.store %2, %4 {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xf32, #blocked>
    tt.return
  }
  tt.func public @wmma_dot_cf16(
   // CHECK: %[[DOT1_ARG_A:.+]]: tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[DOT_OP_PARENT]]}>>
   %0: tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
   // CHECK-SAME: %[[DOT1_ARG_B:.+]]: tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[DOT_OP_PARENT]]}>>
   %1: tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
   %2: tensor<32x32x!tt.ptr<f16, 1>, #blocked>) {
    // CHECK: %[[DOT1_ARG_C:.+]] = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #[[DOT_OP_PARENT]]>
    // CHECK: %[[DOT1_OP_C:.+]] = triton_gpu.convert_layout %[[DOT1_ARG_C]]
    // CHECK-SAME: -> tensor<32x32xf16, #[[WMAA_1]]
    %3 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    // CHECK: %[[DOT1_OP_A:.+]] = triton_gpu.convert_layout %[[DOT1_ARG_A]]
    // CHECK-SAME: -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[WMAA_1]]
    // CHECK: %[[DOT1_OP_B:.+]] = triton_gpu.convert_layout %[[DOT1_ARG_B]]
    // CHECK-SAME: -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[WMAA_1]]
    // CHECK: %[[DOT1_WMMA_RES:.+]] = tt.dot %[[DOT1_OP_A]], %[[DOT1_OP_B]], %[[DOT1_OP_C]]
    // CHECK-SAME: -> tensor<32x32xf16, #[[WMAA_1]]
    %4 = tt.dot %0, %1, %3 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf16, #blocked>
    // CHECK: triton_gpu.convert_layout %[[DOT1_WMMA_RES]]
    // CHECK-SAME: -> tensor<32x32xf16, #[[DOT_OP_PARENT]]>
    tt.store %2, %4 {cache = 1 : i32, evict = 1 : i32} : tensor<32x32xf16, #blocked>
    tt.return
  }
}
