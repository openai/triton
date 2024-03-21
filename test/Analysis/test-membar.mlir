// RUN: triton-opt %s -split-input-file --mlir-disable-threading --convert-scf-to-cf -test-print-membar 2>&1 | FileCheck %s

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
// There shouldn't be any membar with the dot op encoding.
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
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A_DOT>
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B_DOT>
    %c = tt.dot %a, %b, %prev_c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return
}

// CHECK-LABEL: raw_single_block
tt.func @raw_single_block(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %0 = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %1 = tt.load %0, %cst1, %cst2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  %2 = triton_gpu.local_alloc %1 : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %3 = triton_gpu.local_load %2 : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  tt.return
}

// CHECK-LABEL: war_single_block
tt.func @war_single_block(%A : !tt.ptr<f16>) {
  %cst1 = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %cst2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %0 = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %1 = tt.load %0, %cst1, %cst2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  %2 = triton_gpu.local_alloc %1 : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK: triton_gpu.local_alloc
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %3 = triton_gpu.local_load %2 : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: %4 = triton_gpu.local_alloc
  %4 = triton_gpu.local_alloc %1 : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  tt.return
}

// CHECK-LABEL: scratch
tt.func @scratch(%arg: tensor<16x16xf16, #AL>) {
  %cst0 = triton_gpu.local_alloc %arg : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  // CHECK: gpu.barrier
  // CHECK: tt.reduce
  %1 = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  %2 = "tt.reduce" (%1) ({
  ^bb0(%arg1: f16, %arg2: f16):
    %add = arith.addf %arg1, %arg2 : f16
    tt.reduce.return %add : f16
  }) {axis = 0 : i32} : (tensor<16x16xf16, #AL>) -> tensor<16xf16, #sliceAd0>
  tt.return
}

// CHECK-LABEL: async_wait
tt.func @async_wait(%arg: tensor<32x16xf16, #AL>) {
  %cst0 = triton_gpu.local_alloc %arg : (tensor<32x16xf16, #AL>) -> !tt.memdesc<32x16xf16, #A_SHARED>
  // CHECK: triton_gpu.async_wait
  triton_gpu.async_wait {num = 4 : i32}
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %1 = triton_gpu.local_load %cst0 : !tt.memdesc<32x16xf16, #A_SHARED> -> tensor<32x16xf16, #AL>
  tt.return
}

// CHECK-LABEL: subview
tt.func @subview() {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<32x16xf16, #AL>
  %a = triton_gpu.local_alloc %cst0 : (tensor<32x16xf16, #AL>) -> !tt.memdesc<32x16xf16, #A_SHARED>
  %index = arith.constant 0 : i32
  %0 = triton_gpu.memdesc_subview %a[%index, %index, %index] : !tt.memdesc<32x16xf16, #A_SHARED> -> !tt.memdesc<16x16xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %1 = triton_gpu.local_load %0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_alloc
  %2 = triton_gpu.local_alloc %1 : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  tt.return
}

// CHECK-LABEL: trans
tt.func @trans(%a: !tt.memdesc<16x32xf16, #A_SHARED>) {
  // CHECK-NOT: gpu.barrier
  %b = tt.trans %a {order=array<i32: 1,0>} : !tt.memdesc<16x32xf16, #A_SHARED> -> !tt.memdesc<32x16xf16, #A_SHARED_T>
  tt.return
}

// CHECK-LABEL: async_copy_global_to_local
tt.func @async_copy_global_to_local(%A : !tt.ptr<f16>, %i1 : i1) {
  %index = arith.constant 0 : i32
  %a_ptr = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #AL>
  %mask = tt.splat %i1 : i1 -> tensor<16x16xi1, #AL>
  %other = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %alloc = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED, mutable>
  %subview = triton_gpu.memdesc_subview %alloc[%index, %index, %index] : !tt.memdesc<1x16x16xf16, #A_SHARED, mutable> -> !tt.memdesc<16x16xf16, #A_SHARED, mutable>
  %1 = triton_gpu.async_copy_global_to_local %a_ptr, %subview {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16x!tt.ptr<f16>, #AL> -> !tt.memdesc<16x16xf16, #A_SHARED, mutable>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %4 = triton_gpu.local_load %subview : !tt.memdesc<16x16xf16, #A_SHARED, mutable> -> tensor<16x16xf16, #AL>
  tt.return
}
// If branch inserted a barrier for %cst0, but else didn't, then the barrier should be inserted in the parent region
// CHECK-LABEL: multi_blocks
tt.func @multi_blocks(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_load
    %0 = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
    scf.yield
  } else {
    %cst1 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
    scf.yield
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %2 = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
}

// Both branches inserted a barrier for %cst0 and %cst1, then the barrier doesn't need to be inserted in the parent region
// CHECK-LABEL: multi_blocks_join_barrier
tt.func @multi_blocks_join_barrier(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_load
    %0 = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
    scf.yield
  } else {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_load
    %1 = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
    scf.yield
  }
  // CHECK-NOT: gpu.barrier
  // CHECK: tt.return
  %a_ = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
}

// Read yielded tensor requires a barrier
// CHECK-LABEL: multi_blocks_yield
tt.func @multi_blocks_yield(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  %a = scf.if %i1 -> (!tt.memdesc<16x16xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_load
    %0 = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
    %1 = triton_gpu.local_alloc %0 : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
    scf.yield %1 : !tt.memdesc<16x16xf16, #A_SHARED>
  } else {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_load
    %2 = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
    %3 = triton_gpu.local_alloc %2 : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
    scf.yield %3 : !tt.memdesc<16x16xf16, #A_SHARED>
  }
  %a_ = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  // CHECK: triton_gpu.local_load
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %4 = triton_gpu.local_load %a : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
}

// Even though the entry block doesn't have a barrier, the successors should have barriers
// CHECK-LABEL: multi_blocks_entry_no_shared
tt.func @multi_blocks_entry_no_shared(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  %a = scf.if %i1 -> (!tt.memdesc<16x16xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_alloc
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_load
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_alloc
    %cst1 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
    %0 = triton_gpu.local_load %cst1 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
    %1 = triton_gpu.local_alloc %0 : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
    scf.yield %1 : !tt.memdesc<16x16xf16, #A_SHARED>
  } else {
    // CHECK-NOT: gpu.barrier
    // CHECK: triton_gpu.local_alloc
    %cst1 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
    scf.yield %cst1 : !tt.memdesc<16x16xf16, #A_SHARED>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %2 = triton_gpu.local_load %a : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
}

// Conservatively add a barrier as if the branch (%i1) is never taken
// CHECK-LABEL: multi_blocks_noelse
tt.func @multi_blocks_noelse(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_load
    %0 = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
    scf.yield
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %1 = triton_gpu.local_load %cst0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
}

// Conservatively add a barrier as if the branch (%i2) is never taken
// CHECK-LABEL: multi_blocks_nested_scf
tt.func @multi_blocks_nested_scf(%i1 : i1, %i2 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %cst0 = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  scf.if %i1 {
    scf.if %i2 {
      // CHECK: gpu.barrier
      // CHECK-NEXT: triton_gpu.local_load
      %0 = triton_gpu.local_load %cst0 : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
      scf.yield
    }
    scf.yield
  } else {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_load
    %1 = triton_gpu.local_load %cst0 : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    scf.yield
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %2 = triton_gpu.local_load %cst0 : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  tt.return
}

// CHECK-LABEL: for
tt.func @for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %b_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %c_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_load
    %a0 = triton_gpu.local_load %a_shared : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    %b0 = triton_gpu.local_load %b_shared : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    scf.yield %b_shared, %a_shared, %a_shared : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  tt.return
}

// Although a_shared and b_shared are synced before entering the loop,
// they are reassociated with aliases (c_shared) and thus require a barrier.
// CHECK-LABEL: for_alias
tt.func @for_alias(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %b_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %a0 = triton_gpu.local_load %a_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  %b0 = triton_gpu.local_load %b_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  %0 = triton_gpu.local_alloc %a0 : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %c_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_load
    %a1 = triton_gpu.local_load %a_shared : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    %b1 = triton_gpu.local_load %b_shared : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    scf.yield %c_shared, %a_shared, %b_shared : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %r = triton_gpu.local_load %0 : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  tt.return
}

// Although cst2 is not an argument of scf.yield, its memory is reused by cst1.
// So we need a barrier both before and after cst1
// CHECK-LABEL: for_reuse
tt.func @for_reuse(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %b_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %a0 = triton_gpu.local_load %a_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  %b0 = triton_gpu.local_load %b_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  %0 = triton_gpu.local_alloc %a0 : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %c_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_alloc
    %a1 = triton_gpu.local_load %a_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    %b1 = triton_gpu.local_load %b_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    %1 = triton_gpu.local_alloc %a1 : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_alloc
    %a2 = triton_gpu.local_load %a_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    %b2 = triton_gpu.local_load %b_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    %2 = triton_gpu.local_alloc %a1 : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
    scf.yield %c_shared, %a_shared, %b_shared : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %r = triton_gpu.local_load %0 : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  tt.return
}

// CHECK-LABEL: for_reuse_nested
tt.func @for_reuse_nested(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %b_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %a0 = triton_gpu.local_load %a_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  %b0 = triton_gpu.local_load %b_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  %0 = triton_gpu.local_alloc %a0 : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %c_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    // CHECK: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_alloc
    %a1 = triton_gpu.local_load %a_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    %b1 = triton_gpu.local_load %b_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    %1 = triton_gpu.local_alloc %a1 : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
    %a_shared_next, %b_shared_next, %c_shared_next = scf.for %ivv = %lb to %ub step %step iter_args(%a_shared_nested = %a_shared_init, %b_shared_nested = %b_shared_init, %c_shared_nested = %c_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
      // CHECK: gpu.barrier
      // CHECK-NEXT:  triton_gpu.local_alloc
      %a2 = triton_gpu.local_load %a_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
      %b2 = triton_gpu.local_load %b_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
      %2 = triton_gpu.local_alloc %a2 : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
      scf.yield %c_shared_nested, %a_shared_nested, %b_shared_nested : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
    }
    scf.yield %c_shared, %a_shared, %b_shared : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  // CHECK: gpu.barrier
  // CHECK-NEXT:  triton_gpu.local_load
  %r = triton_gpu.local_load %0 : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
  tt.return
}

// repeatedly write to the same shared memory addresses
// CHECK-LABEL: for_for_if
tt.func @for_for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %b_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %c_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    %c_shared_next = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (!tt.memdesc<128x32xf16, #A_SHARED>) {
      %c_shared_next_next = scf.if %i1 -> !tt.memdesc<128x32xf16, #A_SHARED> {
        // CHECK: gpu.barrier
        // CHECK-NEXT: triton_gpu.local_alloc
        %cst0 = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
        scf.yield %cst0 : !tt.memdesc<128x32xf16, #A_SHARED>
      } else {
        // CHECK: gpu.barrier
        // CHECK-NEXT: triton_gpu.local_alloc
        %cst0 = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
        scf.yield %cst0 : !tt.memdesc<128x32xf16, #A_SHARED>
      }
      scf.yield %c_shared_next_next : !tt.memdesc<128x32xf16, #A_SHARED>
    }
    scf.yield %a_shared, %b_shared, %c_shared_next : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  tt.return
}

// c_block_next can either be converted from c_shared_init or c_shared_next_next
// CHECK-LABEL: for_if_for
tt.func @for_if_for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #AL>
  %a_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %b_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  %c_shared_init = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK: gpu.barrier
  %c_blocked = triton_gpu.local_load %c_shared_init : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>

  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) -> (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    %c_shared_next_next = scf.if %i1 -> !tt.memdesc<128x32xf16, #A_SHARED> {
      // CHECK: gpu.barrier
      // CHECK-NEXT: triton_gpu.local_alloc
      %cst0 = triton_gpu.local_alloc %cst : (tensor<128x32xf16, #AL>) -> !tt.memdesc<128x32xf16, #A_SHARED>
      scf.yield %cst0 : !tt.memdesc<128x32xf16, #A_SHARED>
    } else {
      %c_shared_ = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (!tt.memdesc<128x32xf16, #A_SHARED>) {
        // CHECK: gpu.barrier
        // CHECK-NEXT: triton_gpu.local_load
        %c_blocked_next = triton_gpu.local_load %c_shared_next : !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
        scf.yield %c_shared : !tt.memdesc<128x32xf16, #A_SHARED>
      }
      scf.yield %c_shared_ : !tt.memdesc<128x32xf16, #A_SHARED>
    }
    // CHECK-NOT: gpu.barrier
    %b_blocked_next = triton_gpu.local_load %b_shared: !tt.memdesc<128x32xf16, #A_SHARED> -> tensor<128x32xf16, #AL>
    scf.yield %a_shared, %b_shared, %c_shared_next_next : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  tt.return
}

// CHECK-LABEL: cf_if
tt.func @cf_if(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %a = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  cf.cond_br %i1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %0 = triton_gpu.local_load %a : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  cf.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %1 = triton_gpu.local_load %a : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
}

tt.func @cf_if_else(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %a = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  cf.cond_br %i1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %0 = triton_gpu.local_load %a : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  %1 = triton_gpu.local_alloc %0 : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  cf.br ^bb3(%1 : !tt.memdesc<16x16xf16, #A_SHARED>)
^bb2:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %2 = triton_gpu.local_load %a : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  %3 = triton_gpu.local_alloc %2 : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  cf.br ^bb3(%3 : !tt.memdesc<16x16xf16, #A_SHARED>)
^bb3(%arg: !tt.memdesc<16x16xf16, #A_SHARED>):  // 2 preds: ^bb1, ^bb2
  cf.br ^bb4
^bb4:  // pred: ^bb3
  // CHECK: triton_gpu.local_load
  %4 = triton_gpu.local_load %a : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %5 = triton_gpu.local_load %arg : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
}

tt.func @cf_if_else_return(%i1 : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %a = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  %b = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  cf.cond_br %i1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %0 = triton_gpu.local_load %a : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  %1 = triton_gpu.local_load %b : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
^bb2:  // pred: ^bb0
  // CHECK: gpu.barrier
  // CHECK-NEXT: triton_gpu.local_load
  %2 = triton_gpu.local_load %a : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  %3 = triton_gpu.local_load %b : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
}

}

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {

// CHECK-LABEL: convert_layout1
tt.func @convert_layout1(%A : !tt.ptr<f16>) {
  // CHECK-NOT: gpu.barrier
  %0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  %1 = triton_gpu.local_load %0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
}

// CHECK-LABEL: convert_layout2
tt.func @convert_layout2(%A : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  %1 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  // CHECK: triton_gpu.local_load
  // CHECK-NEXT: gpu.barrier
  // CHECK: triton_gpu.local_load
  %3 = triton_gpu.local_load %0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  %4 = triton_gpu.local_load %1 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
  tt.return
}

// CHECK-LABEL: convert_layout3
tt.func @convert_layout3(%cond : i1) {
  scf.if %cond {
    %0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x64xf16, #A_SHARED>
    // CHECK: triton_gpu.local_load
    // CHECK-NOT: gpu.barrier
    %1 = triton_gpu.local_load %0 : !tt.memdesc<16x64xf16, #A_SHARED> -> tensor<16x64xf16, #AL>
  } else {
    %0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
    // CHECK: triton_gpu.local_load
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: triton_gpu.local_alloc
    %1 = triton_gpu.local_load %0 : !tt.memdesc<16x16xf16, #A_SHARED> -> tensor<16x16xf16, #AL>
    %2 = triton_gpu.local_alloc %1 : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  }
  tt.return
}

// CHEKC-LABEL: convert_layout4
tt.func @convert_layout4(%A : !tt.ptr<f16>, %cond : i1) {
  // CHECK-NOT: gpu.barrier
  scf.if %cond {
    tt.call @convert_layout3(%cond) : (i1) -> ()
  } else {
    tt.call @convert_layout2(%A) : (!tt.ptr<f16>) -> ()
  }
  tt.return
}

// CHECK-LABEL: single_call_sync
tt.func @single_call_sync(%A : !tt.ptr<f16>) {
  %0 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  // CHECK: tt.call
  // CHECK-NEXT: gpu.barrier
  tt.call @convert_layout1(%A) : (!tt.ptr<f16>) -> ()
  %1 = triton_gpu.convert_layout %0 : tensor<16x32xf16, #AL> -> tensor<16x32xf16, #BL>
  tt.return
}

// CHECK-LABEL: single_call_no_sync
// %1 can reuse %0 in convert_layout2, which has been synced
tt.func @single_call_no_sync(%A : !tt.ptr<f16>) {
  // CHECK-NOT: gpu.barrier
  %0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  tt.call @convert_layout2(%A) : (!tt.ptr<f16>) -> ()
  %1 = triton_gpu.convert_layout %0 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #BL>
  tt.return
}

// CHECK-LABEL: multiple_calls
tt.func @multiple_calls(%A : !tt.ptr<f16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  tt.call @convert_layout1(%A) : (!tt.ptr<f16>) -> ()
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  tt.call @convert_layout2(%A) : (!tt.ptr<f16>) -> ()
  tt.return
}

// CHECK-LABEL: if_else_calls
tt.func @if_else_calls(%A : !tt.ptr<f16>, %cond : i1) {
  scf.if %cond {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
    %cst_ = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
    %cst0 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.call
    // CHECK-NEXT: gpu.barrier
    tt.call @convert_layout1(%A) : (!tt.ptr<f16>) -> ()
    %cst1 = triton_gpu.local_alloc %cst_ : (tensor<16x32xf16, #AL>) -> !tt.memdesc<16x32xf16, #A_SHARED>
  } else {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
    // CHECK: tt.call
    // CHECK-NOT: gpu.barrier
    tt.call @convert_layout2(%A) : (!tt.ptr<f16>) -> ()
  }
  tt.return
}

// CHECK-LABEL: for_calls
tt.func @for_calls(%A : !tt.ptr<f16>, %cond : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x32xf16, #AL>
  %lb = arith.constant 0 : index
  %ub = arith.constant 10 : index
  %step = arith.constant 1 : index
  scf.for %iv = %lb to %ub step %step {
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.call
    tt.call @convert_layout1(%A) : (!tt.ptr<f16>) -> ()
  }
  tt.return
}

// CHECK-LABEL: call_graph_1
tt.func @call_graph_1(%A : !tt.ptr<f16>, %cond : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  %cst0 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.call
  tt.call @convert_layout3(%cond) : (i1) -> ()
  tt.return
}

// CHECK-LABEL: call_graph_2
tt.func @call_graph_2(%A : !tt.ptr<f16>, %cond : i1) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  tt.call @convert_layout4(%A, %cond) : (!tt.ptr<f16>, i1) -> ()
  // CHECK: tt.call
  // CHECK-NEXT: gpu.barrier
  %cst0 = triton_gpu.local_alloc %cst : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  tt.return
}

}
