// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul=compute-capability=90 | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul=compute-capability=89 | FileCheck %s --check-prefix=CHECK-89
// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul=compute-capability=80 | FileCheck %s --check-prefix=CHECK-80

// CHECK: #[[MMA:.+]] = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
// CHECK: #[[MMA1:.+]] = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
// CHECK: #[[MMA2:.+]] = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK: mma_chain_loop
  tt.func public @mma_chain_loop(
   %170: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
   %171: tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
   %179: tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>,
   %164: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>>,
   %165: tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>>,
   %173: tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>,
   %153: tensor<128x64x!tt.ptr<f16, 1>, #blocked1>) {
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf16, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked2>
    // CHECK: scf.for
    // CHECK:   tt.dot {{.*}} -> tensor<128x16xf16, #[[MMA]]>
    // CHECK:   tt.dot {{.*}} -> tensor<128x64xf16, #[[MMA1]]>
    %115 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %cst_0) -> (tensor<128x64xf16, #blocked1>) : i32 {
      %172 = tt.dot %170, %171, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x16xf16, #blocked>
      %178 = triton_gpu.convert_layout %172 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
      %180 = tt.dot %178, %179, %arg16 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %180 : tensor<128x64xf16, #blocked1>
    }
    // CHECK: scf.for
    // CHECK:   tt.dot {{.*}} -> tensor<128x32xf16, #[[MMA2]]>
    // CHECK:   tt.dot {{.*}} -> tensor<128x64xf16, #[[MMA1]]>
    %149 = scf.for %arg15 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg16 = %115) -> (tensor<128x64xf16, #blocked1>) : i32 {
      %166 = tt.dot %164, %165, %cst_2 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x32xf16, #blocked2>
      %172 = triton_gpu.convert_layout %166 : tensor<128x32xf16, #blocked2> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
      %174 = tt.dot %172, %173, %arg16 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x64xf16, #blocked1>
      scf.yield %174 : tensor<128x64xf16, #blocked1>
    }
    tt.store %153, %149 {cache = 1 : i32, evict = 1 : i32} : tensor<128x64xf16, #blocked1>
    tt.return
  }
}

// -----

// CHECK-80: #[[$MMA:.+]] = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-80-LABEL: chained_dot
  tt.func public @chained_dot(
    %arg0: tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
  // CHECK-80: tt.dot {{.*}} -> tensor<64x64xf32, #[[$MMA]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %c = triton_gpu.convert_layout %t : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
  // CHECK-80: tt.dot {{.*}} -> tensor<64x128xf32, #[[$MMA]]>
    %r = tt.dot %c, %arg2, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// CHECK-89: #[[$MMA:.+]] = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 8]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 89 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-89-LABEL: fp8_dot
  tt.func public @fp8_dot(
    %arg0: tensor<64x128xf8E4M3FNUZ, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x64xf32, #blocked> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
  // CHECK-89: tt.dot {{.*}} : tensor<64x128xf8E4M3FNUZ, #triton_gpu.dot_op<{opIdx = 0, parent = #[[$MMA]], kWidth = 4}>> * tensor<128x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 1, parent = #[[$MMA]], kWidth = 4}>> -> tensor<64x64xf32, #[[$MMA]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x128xf8E4M3FNUZ, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %d : tensor<64x64xf32, #blocked>
  }
}

// -----

// CHECK-DAG: #[[MMA:.+]] = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
// CHECK-DAG: #[[MMA1:.+]] = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1, 1], instrShape = [1, 16, 8]}>
// CHECK-80-DAG: #[[MMA:.+]] = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
// CHECK-80-DAG: #[[MMA1:.+]] = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1, 1], instrShape = [1, 16, 8]}>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [0, 1, 2]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 2, 2], threadsPerWarp = [1, 4, 8], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_() attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x16x16xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked1>
    %0 = triton_gpu.convert_layout %cst_0 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>>
    %1 = triton_gpu.convert_layout %cst_0 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>>
    %2 = triton_gpu.convert_layout %cst_0 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<16x16xf32, #[[MMA]]>
    // CHECK-80: tt.dot {{.*}} -> tensor<16x16xf32, #[[MMA]]>
    %3 = tt.dot %0, %1, %2 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<16x16xf32, #blocked1>
    %4 = triton_gpu.convert_layout %3 : tensor<16x16xf32, #blocked1> -> tensor<16x16xf32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16x16xf32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x16x16xf32, #blocked2>
    %6 = triton_gpu.convert_layout %5 : tensor<1x16x16xf32, #blocked2> -> tensor<1x16x16xf32, #blocked>
    %7 = tt.broadcast %6 : tensor<1x16x16xf32, #blocked> -> tensor<2x16x16xf32, #blocked>
    %8 = triton_gpu.convert_layout %7 : tensor<2x16x16xf32, #blocked> -> tensor<2x16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked3}>>
    %9 = triton_gpu.convert_layout %cst : tensor<2x16x16xf32, #blocked> -> tensor<2x16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked3}>>
    %10 = triton_gpu.convert_layout %cst : tensor<2x16x16xf32, #blocked> -> tensor<2x16x16xf32, #blocked3>
    // CHECK: tt.dot {{.*}} -> tensor<2x16x16xf32, #[[MMA1]]>
    // CHECK-80: tt.dot {{.*}} -> tensor<2x16x16xf32, #[[MMA1]]>
    %11 = tt.dot %8, %9, %10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<2x16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked3}>> * tensor<2x16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked3}>> -> tensor<2x16x16xf32, #blocked3>
    %12 = triton_gpu.convert_layout %11 : tensor<2x16x16xf32, #blocked3> -> tensor<2x16x16xf32, #blocked>
    tt.print ": " {hex = false} : %12 : tensor<2x16x16xf32, #blocked>
    tt.return
  }
}
