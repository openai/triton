// RUN: triton-opt %s -split-input-file -optimize-amd-lds-usage | FileCheck %s

// Check that optimization detects overflow of LDS and decomposes layout convert so kernel fits into LDS
// CHECK-LABEL: alloc_convert_load
// CHECK: %0 = triton_gpu.local_alloc %arg0 : {{.*}}#blocked{{.*}}#shared
// CHECK: %1 = triton_gpu.convert_layout %arg1 : {{.*}}#blocked{{.*}}#blocked1
// CHECK: %2 = triton_gpu.convert_layout %1 : {{.*}}#blocked1{{.*}}#mma
// CHECK: %3 = triton_gpu.local_load %0 : {{.*}}#shared{{.*}}#triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [32, 32], isTransposed = false}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @alloc_convert_load(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x128xf32, #blocked>) attributes {noinline = false} {
    %1 = triton_gpu.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !tt.memdesc<128x128xf16, #shared>
    %2 = triton_gpu.convert_layout %arg1 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #mma>
    %3 = triton_gpu.local_load %1 : !tt.memdesc<128x128xf16, #shared> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
    tt.return
  }
}
