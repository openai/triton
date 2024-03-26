// RUN: triton-opt %s -split-input-file --mlir-disable-threading -test-print-allocation 2>&1 | FileCheck %s

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#sliceAd0 = #triton_gpu.slice<{dim = 0, parent = #AL}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#A_SHARED_T = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#B_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A_DOT = #triton_gpu.dot_op<{opIdx = 0, parent = #C}>
#B_DOT = #triton_gpu.dot_op<{opIdx = 1, parent = #C}>

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {

// CHECK-LABEL: matmul_loop
tt.func @matmul_loop(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_ptr_init = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    // CHECK: offset = 0, size = 4608
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A_DOT>
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    // CHECK-NEXT: offset = 0, size = 4224
    %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B_DOT>

    %c = tt.dot %a, %b, %prev_c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return
  // CHECK-NEXT: size = 4608
}

// Shared memory is available after a tensor's liveness range ends
// CHECK-LABEL: reusable
tt.func @reusable(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %cst3 = arith.constant dense<true> : tensor<32x128xi1, #AL>
  %cst4 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #AL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %b_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #AL>
  %a1_ = tt.load %a_ptr, %cst1, %cst2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 4608
  %a1 = triton_gpu.convert_layout %a1_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A_DOT>
  %a2_ = tt.load %b_ptr, %cst3, %cst4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 1152
  %a2 = triton_gpu.convert_layout %a2_ : tensor<32x128xf16, #AL> -> tensor<32x128xf16, #B_DOT>
  %a3_ = tt.load %a_ptr, %cst1, %cst2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 4608
  %a3 = triton_gpu.convert_layout %a3_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A_DOT>
  %c = tt.dot %a1, %a2, %c_init {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>
  %a4_ = tt.load %b_ptr, %cst3, %cst4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 1152
  %a4 = triton_gpu.convert_layout %a4_ : tensor<32x128xf16, #AL> -> tensor<32x128xf16, #B_DOT>
  %c1 = tt.dot %a3, %a4, %c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>
  tt.return
  // CHECK-NEXT: size = 4608
}

// A tensor's shared memory offset is larger than it needs to accommodate further tensors
// %cst0->%c
// %cst1->%cst4
// %cst3->%g->%h->%i
// CHECK-LABEL: preallocate
tt.func @preallocate(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst1 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 512
  %cst2 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 3072, size = 1024
  %a = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 4096, size = 1024
  %b = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>

  triton_gpu.local_dealloc %cst0 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 0, size = 1024
  %c = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>

  triton_gpu.local_dealloc %cst1 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst2 : !tt.memdesc<1x16x16xf16, #A_SHARED>

  // CHECK-NEXT: offset = 1024, size = 1024
  %cst4 = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 6144, size = 2048
  %e = triton_gpu.local_alloc : () -> !tt.memdesc<64x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %a : !tt.memdesc<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 8192, size = 2048
  %d = triton_gpu.local_alloc : () -> !tt.memdesc<64x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %b : !tt.memdesc<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 10240, size = 2048
  %f = triton_gpu.local_alloc : () -> !tt.memdesc<64x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst4 : !tt.memdesc<32x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %c : !tt.memdesc<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 0, size = 2048
  %cst5 = triton_gpu.local_alloc : () -> !tt.memdesc<64x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 4096
  %g = triton_gpu.local_alloc : () -> !tt.memdesc<128x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %e : !tt.memdesc<64x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 4096
  %h = triton_gpu.local_alloc : () -> !tt.memdesc<128x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %d : !tt.memdesc<64x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 4096
  %i = triton_gpu.local_alloc : () -> !tt.memdesc<128x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %f : !tt.memdesc<64x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst5 : !tt.memdesc<64x16xf16, #A_SHARED>
  tt.return
  // CHECK-NEXT: size = 12288
}

// Unused tensors are immediately released
// CHECK-LABEL: unused
tt.func @unused(%A : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<32x16xf16, #AL>
  // CHECK: offset = 0, size = 1024
  %cst0 = triton_gpu.local_alloc %cst : (tensor<32x16xf16, #AL>) -> !tt.memdesc<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 0, size = 512
  %cst1 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 0, size = 512
  %cst2 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  tt.return
  // CHECK: size = 1024
}

// cst0 is alive through the entire function, it cannot be released before the end of the function
// CHECK-LABEL: longlive
tt.func @longlive(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst1 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 512
  %cst2 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 3072, size = 1024
  %a = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst1 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst2 : !tt.memdesc<1x16x16xf16, #A_SHARED>

  // CHECK-NEXT: offset = 1024, size = 512
  %cst3 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 512
  %cst4 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 3072, size = 1024
  %b = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 3072, size = 512
  %cst5 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 3072, size = 512
  %cst6 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 3072, size = 1024
  %c = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst3 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst4 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 1024
  %d = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst0 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  tt.return
  // CHECK-NEXT: size = 4096
}

// This example triggers graph coloring with > 1 colors.
// CHECK-LABEL: multi_color
tt.func @multi_color(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 64
  %cst = triton_gpu.local_alloc : () -> !tt.memdesc<4x8xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1536, size = 32
  %cst_0 = triton_gpu.local_alloc : () -> !tt.memdesc<4x4xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1664, size = 128
  %cst_1 = triton_gpu.local_alloc : () -> !tt.memdesc<16x4xf16, #A_SHARED>
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  // CHECK-NEXT: scratch offset = 128, size = 1152
  %0 = triton_gpu.convert_layout %cst_2 : tensor<16x32xf16, #AL> -> tensor<16x32xf16, #AL>
  %1 = triton_gpu.local_load %cst : !tt.memdesc<4x8xf16, #A_SHARED> -> tensor<4x8xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 128
  %cst_3 = triton_gpu.local_alloc : () -> !tt.memdesc<4x16xf16, #A_SHARED>
  %2 = triton_gpu.local_load %cst_0 : !tt.memdesc<4x4xf16, #A_SHARED> -> tensor<4x4xf16, #AL>
  // CHECK-NEXT: scratch offset = 0, size = 1152
  %3 = triton_gpu.convert_layout %cst_2 : tensor<16x32xf16, #AL> -> tensor<16x32xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 256
  %cst_4 = triton_gpu.local_alloc : () -> !tt.memdesc<4x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 256, size = 64
  %cst_5 = triton_gpu.local_alloc : () -> !tt.memdesc<4x8xf16, #A_SHARED>
  %4 = triton_gpu.local_load %cst_5 : !tt.memdesc<4x8xf16, #A_SHARED> -> tensor<4x8xf16, #AL>
  %5 = triton_gpu.local_load %cst_5 : !tt.memdesc<4x8xf16, #A_SHARED> -> tensor<4x8xf16, #AL>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst_6 = triton_gpu.local_alloc : () -> !tt.memdesc<8x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 3104, size = 128
  %cst_7 = triton_gpu.local_alloc : () -> !tt.memdesc<2x32xf16, #A_SHARED>
  %6 = triton_gpu.local_load %cst_0 : !tt.memdesc<4x4xf16, #A_SHARED> -> tensor<4x4xf16, #AL>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst_8 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 256, size = 32
  %cst_9 = triton_gpu.local_alloc : () -> !tt.memdesc<4x4xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst_10 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  %7 = triton_gpu.local_load %cst_1 : !tt.memdesc<16x4xf16, #A_SHARED> -> tensor<16x4xf16, #AL>
  %8 = triton_gpu.local_load %cst_4 : !tt.memdesc<4x32xf16, #A_SHARED> -> tensor<4x32xf16, #AL>
  // CHECK-NEXT: scratch offset = 0, size = 1152
  %9 = triton_gpu.convert_layout %cst_2 : tensor<16x32xf16, #AL> -> tensor<16x32xf16, #AL>
  %cst_11 = arith.constant dense<0.000000e+00> : tensor<4x4xf16, #AL>
  %10 = triton_gpu.local_load %cst_7 : !tt.memdesc<2x32xf16, #A_SHARED> -> tensor<2x32xf16, #AL>
  %cst_12 = arith.constant dense<0.000000e+00> : tensor<4x16xf16, #AL>
  %cst_13 = arith.constant dense<0.000000e+00> : tensor<8x32xf16, #AL>
  // CHECK-NEXT: size = 3232
  tt.return
}

// This example triggers graph coloring with multiple rounds
// CHECK-LABEL: multi_color_multi_rounds
tt.func @multi_color_multi_rounds(%arg0: !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 32
  %cst = triton_gpu.local_alloc : () -> !tt.memdesc<4x4xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1280, size = 128
  %cst_0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x4xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 8192
  %cst_1 = triton_gpu.local_alloc : () -> !tt.memdesc<1024x4xf16, #A_SHARED>
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  // CHECK-NEXT: scratch offset = 128, size = 1152
  %0 = triton_gpu.convert_layout %cst_2 : tensor<16x32xf16, #AL> -> tensor<16x32xf16, #AL>
  %1 = triton_gpu.local_load %cst : !tt.memdesc<4x4xf16, #A_SHARED> -> tensor<4x4xf16, #AL>
  // CHECK-NEXT: offset = 1152, size = 128
  %cst_3 = triton_gpu.local_alloc : () -> !tt.memdesc<2x32xf16, #A_SHARED>
  %2 = triton_gpu.local_load %cst : !tt.memdesc<4x4xf16, #A_SHARED> -> tensor<4x4xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 512
  %cst_4 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  %3 = triton_gpu.local_load %cst_0 : !tt.memdesc<16x4xf16, #A_SHARED> -> tensor<16x4xf16, #AL>
  %4 = triton_gpu.local_load %cst_1 : !tt.memdesc<1024x4xf16, #A_SHARED> -> tensor<1024x4xf16, #AL>
  // CHECK-NEXT: scratch offset = 0, size = 1152
  %5 = triton_gpu.convert_layout %cst_2 : tensor<16x32xf16, #AL> -> tensor<16x32xf16, #AL>
  %6 = triton_gpu.local_load %cst_3 : !tt.memdesc<2x32xf16, #A_SHARED> -> tensor<2x32xf16, #AL>
  // CHECK-NEXT: size = 10240
  tt.return
}


// CHECK-LABEL: alloc
tt.func @alloc(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  // CHECK-NEXT: offset = 0, size = 512
  %cst2 = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  tt.return
  // CHECK-NEXT: size = 512
}


// CHECK-LABEL: dealloc
tt.func @dealloc(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 1024
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  // CHECK: offset = 1024, size = 1024
  %cst1 = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst0 : !tt.memdesc<32x16xf16, #A_SHARED>
  tt.return
  // CHECK-NEXT: size = 2048
}

// CHECK-LABEL: scratch
tt.func @scratch() {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  // CHECK: scratch offset = 0, size = 128
  %b = "tt.reduce" (%cst0) ({
  ^bb0(%arg0: f16, %arg1: f16):
    %add = arith.addf %arg0, %arg1 : f16
    tt.reduce.return %add : f16
  }) {axis = 0 : i32} : (tensor<16x16xf16, #AL>) -> tensor<16xf16, #sliceAd0>
  tt.return
  // CHECK-NEXT: size = 128
}

// CHECK-LABEL: trans
tt.func @trans(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 1024
  %tensor = triton_gpu.local_alloc : () -> !tt.memdesc<16x32xf16, #A_SHARED>
  %b = tt.trans %tensor {order=array<i32: 1,0>} : !tt.memdesc<16x32xf16, #A_SHARED> -> !tt.memdesc<32x16xf16, #A_SHARED_T>
  tt.return
}


// CHECK-LABEL: extract_slice
tt.func @extract_slice(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  %index = arith.constant 0 : i32
  %cst1 = triton_gpu.memdesc_subview %cst0[%index, %index, %index] : !tt.memdesc<1x16x16xf16, #A_SHARED> -> !tt.memdesc<16x16xf16, #A_SHARED>
  tt.return
  // CHECK-NEXT: size = 512
}

// B0 -> (B1) -> B0
// Memory used by B1 can be reused by B0.
// CHECK-LABEL: if
tt.func @if(%i1 : i1) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst1 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK-NEXT: offset = 2048, size = 1024
    %a = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
    // CHECK-NEXT: offset = 2048, size = 1024
    %b = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
    triton_gpu.local_dealloc %cst0 : !tt.memdesc<1x16x16xf16, #A_SHARED>
    triton_gpu.local_dealloc %cst1 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  }
  // CHECK-NEXT: offset = 0, size = 512
  %cst2 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst3 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 2048, size = 1024
  %a = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst2 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst3 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  tt.return
  // CHECK-NEXT: size = 3072
}

// B0 -> (B1) -> (B2) -> B0
// Memory used by B0 cannot be reused by B1 or B2.
// CHECK-LABEL: if_else
tt.func @if_else(%i1 : i1) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: offset = 1024, size = 512
  %cst1 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK-NEXT: offset = 2048, size = 1024
    %a = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
    // CHECK-NEXT: offset = 2048, size = 1024
    %b = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  } else {
    // CHECK-NEXT: offset = 2048, size = 512
    %cst2 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
    // CHECK-NEXT: offset = 3072, size = 512
    %cst3 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
    // CHECK-NEXT: offset = 4096, size = 1024
    %a = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
    triton_gpu.local_dealloc %cst2 : !tt.memdesc<1x16x16xf16, #A_SHARED>
    triton_gpu.local_dealloc %cst3 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  }
  // CHECK-NEXT: offset = 2048, size = 1024
  %a = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst0 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  triton_gpu.local_dealloc %cst1 : !tt.memdesc<1x16x16xf16, #A_SHARED>
  tt.return
  // CHECK-NEXT: size = 5120
}

// Block arguments and yields are memory aliases that do not trigger a new
// allocation.
// CHECK-LABEL: for
tt.func @for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 8192
  %a_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 8192, size = 8192
  %b_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 16384, size = 8192
  %c_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    scf.yield %b_shared, %a_shared, %a_shared : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  tt.return
  // CHECK-NEXT: size = 24576
}

// CHECK-LABEL: for_if_slice
tt.func @for_if_slice(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: offset = 0, size = 8192
  %a_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 8192, size = 8192
  %b_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 16384, size = 8192
  %c_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    scf.if %i1 {
      %index = arith.constant 8 : i32
      %cst0 = triton_gpu.memdesc_subview %a_shared[%index, %index] : !tt.memdesc<128x32xf16, #A_SHARED> -> !tt.memdesc<32xf16, #A_SHARED>
      scf.yield
    }
    scf.yield %b_shared, %a_shared, %a_shared : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  tt.return
  // CHECK-NEXT: size = 24576
}

// c0 cannot be released in the loop
// CHECK-LABEL: for_use_ancestor
tt.func @for_use_ancestor(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: offset = 0, size = 8192
  %a_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 8192, size = 8192
  %b_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 16384, size = 8192
  %c_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  %a_shared, %b_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    %c0 = tt.trans %c_shared_init {order=array<i32: 1,0>} : !tt.memdesc<128x32xf16, #A_SHARED> -> !tt.memdesc<32x128xf16, #A_SHARED_T>
    // CHECK-NEXT: offset = 24576, size = 8192
    %c1 = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
    scf.yield %b_shared, %a_shared: !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  tt.return
  // CHECK-NEXT: size = 32768
}

// a_shared_init, b_shared_init, and c_shared_init's liveness ranges are span over the entire function before cst2.
// So they cannot be reused by cst0 and cst1, but can be reused by cst2.
// CHECK-LABEL: for_for_if
tt.func @for_for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: offset = 0, size = 8192
  %a_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 8192, size = 8192
  %b_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: offset = 16384, size = 8192
  %c_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    %c_shared_next = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (!tt.memdesc<128x32xf16, #A_SHARED>) {
      %c_shared_next_next = scf.if %i1 -> !tt.memdesc<128x32xf16, #A_SHARED> {
        // CHECK-NEXT: offset = 24576, size = 8192
        %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
        scf.yield %cst0 : !tt.memdesc<128x32xf16, #A_SHARED>
      } else {
        // CHECK-NEXT: offset = 32768, size = 8192
        %cst1 = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
        scf.yield %cst1 : !tt.memdesc<128x32xf16, #A_SHARED>
      }
      scf.yield %c_shared_next_next : !tt.memdesc<128x32xf16, #A_SHARED>
    }
    scf.yield %a_shared, %b_shared, %c_shared_next : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  // CHECK-NEXT: offset = 0, size = 8192
  %cst2 = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  tt.return
  // CHECK-NEXT: size = 40960
}

}

module attributes {"triton_gpu.num-warps" = 4 : i32} {

// CHECK-LABEL: alloc1
tt.func @alloc1(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  tt.return
  // CHECK-NEXT: size = 512
}

// CHECK-LABEL: alloc2
tt.func @alloc2(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 1024
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<32x16xf16, #A_SHARED>
  tt.return
  // CHECK-NEXT: size = 1024
}

// CHECK-LABEL: alloc3
tt.func @alloc3(%cond : i1) {
  scf.if %cond {
    // CHECK: offset = 0, size = 512
    %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  } else {
    // CHECK-NEXT: offset = 0, size = 1024
    %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x32xf16, #A_SHARED>
  }
  tt.return
  // CHECK-NEXT: size = 1024
}

// CHECK-LABEL: alloc4
tt.func @alloc4(%A : !tt.ptr<f16>, %cond : i1) {
  scf.if %cond {
    // CHECK: virtual offset = 0, size = 1024
    tt.call @alloc3(%cond) : (i1) -> ()
  } else {
    // CHECK-NEXT: virtual offset = 0, size = 512
    tt.call @alloc1(%A) : (!tt.ptr<f16>) -> ()
  }
  tt.return
  // CHECK-NEXT: size = 1024
}

// CHECK-LABEL: single_call
tt.func @single_call(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  // CHECK-NEXT: virtual offset = 0, size = 512
  tt.call @alloc1(%A) : (!tt.ptr<f16>) -> ()
  tt.return
  // CHECK-NEXT: size = 512
}

// CHECK-LABEL: multiple_calls
tt.func @multiple_calls(%A : !tt.ptr<f16>) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: virtual offset = 0, size = 512
  tt.call @alloc1(%A) : (!tt.ptr<f16>) -> ()
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  // CHECK-NEXT: virtual offset = 0, size = 1024
  tt.call @alloc2(%A) : (!tt.ptr<f16>) -> ()
  tt.return
  // CHECK-NEXT: size = 1024
}

// CHECK-LABEL: if_else_calls
tt.func @if_else_calls(%A : !tt.ptr<f16>, %cond : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  scf.if %cond {
    // CHECK: offset = 0, size = 512
    %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
    // CHECK-NEXT: offset = 0, size = 1024
    %cst1 = triton_gpu.local_alloc %cst : (tensor<16x32xf16, #AL>) -> !tt.memdesc<16x32xf16, #A_SHARED>
    // CHECK-NEXT: virtual offset = 0, size = 512
    tt.call @alloc1(%A) : (!tt.ptr<f16>) -> ()
  } else {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
    // CHECK-NEXT: virtual offset = 0, size = 1024
    tt.call @alloc2(%A) : (!tt.ptr<f16>) -> ()
  }
  tt.return
  // CHECK-NEXT: size = 1024
}

// CHECK-LABEL: for_calls
tt.func @for_calls(%A : !tt.ptr<f16>, %cond : i1) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index
  scf.for %iv = %lb to %ub step %step {
    // CHECK-NEXT: virtual offset = 0, size = 512
    tt.call @alloc1(%A) : (!tt.ptr<f16>) -> ()
  }
  tt.return
  // CHECK-NEXT: size = 512
}

// CHECK-LABEL: call_graph_1
tt.func @call_graph_1(%A : !tt.ptr<f16>, %cond : i1) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: virtual offset = 0, size = 1024
  tt.call @alloc3(%cond) : (i1) -> ()
  tt.return
  // CHECK-NEXT: size = 1024
}

// CHECK-LABEL: call_graph_2
tt.func @call_graph_2(%A : !tt.ptr<f16>, %cond : i1) {
  // CHECK: offset = 0, size = 512
  %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: virtual offset = 0, size = 1024
  tt.call @alloc4(%A, %cond) : (!tt.ptr<f16>, i1) -> ()
  tt.return
  // CHECK-NEXT: size = 1024
}

}
