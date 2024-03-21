// RUN: triton-opt %s --mlir-disable-threading -test-print-alias -split-input-file 2>&1 | FileCheck %s

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#A_SHARED_T = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#B_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A_DOT = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B_DOT = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.compute-capability" = 80} {

// CHECK-LABEL: matmul_loop
// CHECK-NOT: ->
// There shouldn't be any aliasing with the dot op encoding.
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
    %c = tt.dot %a, %b, %prev_c {transA = false, transB = false, inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return
}

// CHECK-LABEL: alloc
tt.func @alloc(%A : !tt.ptr<f16>) {
  // CHECK: %0 -> %0
  %cst2 = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED, mutable>
  tt.return
}

// CHECK-LABEL: alloc_init
tt.func @alloc_init(%A : !tt.ptr<f16>) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  // CHECK: %0 -> %0
  %cst1 = triton_gpu.local_alloc %cst0 : (tensor<16x16xf16, #AL>) -> !tt.memdesc<16x16xf16, #A_SHARED>
  tt.return
}

// CHECK-LABEL: trans
tt.func @trans(%A : !tt.ptr<f16>) {
  // CHECK: %0 -> %0
  %tensor = triton_gpu.local_alloc : () -> !tt.memdesc<16x32xf16, #A_SHARED>
  // CHECK: %1 -> %0
  %b = tt.trans %tensor {order=array<i32: 1,0>} : !tt.memdesc<16x32xf16, #A_SHARED> -> !tt.memdesc<32x16xf16, #A_SHARED_T>
  tt.return
}

// CHECK-LABEL: subview
tt.func @subview(%A : !tt.memdesc<1x16x16xf16, #A_SHARED>) {
  %index = arith.constant 0 : i32
  // CHECK: %0 -> %0
  %a = triton_gpu.local_alloc : () -> !tt.memdesc<1x16x16xf16, #A_SHARED>
  // CHECK-NEXT: %1 -> %0
  %cst1 = triton_gpu.memdesc_subview %a[%index, %index, %index] : !tt.memdesc<1x16x16xf16, #A_SHARED> -> !tt.memdesc<16x16xf16, #A_SHARED>
  tt.return
}

// CHECK-LABEL: if_alias
tt.func @if_alias(%i1 : i1) {
  // CHECK: %0 -> %0
  %a = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  // CHECK: %1 -> %1
  %b = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  // CHECK-NEXT: %2 -> %0,%1
  %cst2 = scf.if %i1 -> !tt.memdesc<16x16xf16, #A_SHARED> {
    scf.yield %a : !tt.memdesc<16x16xf16, #A_SHARED>
  } else {
    scf.yield %b : !tt.memdesc<16x16xf16, #A_SHARED>
  }
  tt.return
}

// CHECK-LABEL: for
tt.func @for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  // CHECK: %0 -> %0
  %a = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  // CHECK: %1 -> %1
  %b = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  // CHECK: %2 -> %2
  %c = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #A_SHARED>
  // CHECK-NEXT: %arg6 -> %0
  // CHECK-NEXT: %arg7 -> %1
  // CHECK-NEXT: %arg8 -> %2
  // CHECK-NEXT: %3#0 -> %0,%1
  // CHECK-NEXT: %3#1 -> %0,%1
  // CHECK-NEXT: %3#2 -> %0,%1,%2
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a, %b_shared = %b, %c_shared = %c) ->
  (!tt.memdesc<16x16xf16, #A_SHARED>, !tt.memdesc<16x16xf16, #A_SHARED>, !tt.memdesc<16x16xf16, #A_SHARED>) {
    scf.yield %b_shared, %a_shared, %a_shared : !tt.memdesc<16x16xf16, #A_SHARED>, !tt.memdesc<16x16xf16, #A_SHARED>, !tt.memdesc<16x16xf16, #A_SHARED>
  }
  tt.return
}

// CHECK-LABEL: for_if
tt.func @for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: %0 -> %0
  %a_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %1 -> %1
  %b_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %2 -> %2
  %c_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %arg7 -> %0
  // CHECK-NEXT: %arg8 -> %1
  // CHECK-NEXT: %arg9 -> %2
  // CHECK-NEXT: %3#0 -> %0,%1
  // CHECK-NEXT: %3#1 -> %0,%1
  // CHECK-NEXT: %3#2 -> %0,%1,%2
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) ->
  (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    scf.if %i1 {
      %index = arith.constant 8 : i32
      // CHECK-NEXT: %4 -> %0,%1
      %cst0 = triton_gpu.memdesc_subview %a_shared[%index, %index] : !tt.memdesc<128x32xf16, #A_SHARED> -> !tt.memdesc<32xf16, #A_SHARED>
      scf.yield
    }
    scf.yield %b_shared, %a_shared, %a_shared : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  tt.return
}

// CHECK-LABEL: for_for_if
tt.func @for_for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // CHECK: %0 -> %0
  %a_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %1 -> %1
  %b_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %2 -> %2
  %c_shared_init = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %arg7 -> %0
  // CHECK-NEXT: %arg8 -> %1
  // CHECK-NEXT: %arg9 -> %2
  // CHECK-NEXT: %3#0 -> %0
  // CHECK-NEXT: %3#1 -> %1
  // CHECK-NEXT: %3#2 -> %2,%6,%6
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) ->
  (!tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>) {
    // CHECK-NEXT: %arg11 -> %2,%6,%6
    // CHECK-NEXT: %4 -> %2,%6,%6
    %c_shared_next = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (!tt.memdesc<128x32xf16, #A_SHARED>) {
      // CHECK-NEXT: %5 -> %6,%6
      %c_shared_next_next = scf.if %i1 -> !tt.memdesc<128x32xf16, #A_SHARED> {
        // CHECK-NEXT: %6 -> %6
        %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
        scf.yield %cst0 : !tt.memdesc<128x32xf16, #A_SHARED>
      } else {
        // CHECK-NEXT: %6 -> %6
        %cst0 = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
        scf.yield %cst0 : !tt.memdesc<128x32xf16, #A_SHARED>
      }
      scf.yield %c_shared_next_next : !tt.memdesc<128x32xf16, #A_SHARED>
    }
    scf.yield %a_shared, %b_shared, %c_shared_next : !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>
  }
  tt.return
}

// CHECK-LABEL: cf_for
tt.func @cf_for(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) {
  %idx = arith.constant 0 : i32
  // CHECK: %0 -> %0
  %cst = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %1 -> %1
  %cst_0 = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %2 -> %0
  %0 = triton_gpu.memdesc_subview %cst[%idx, %idx] : !tt.memdesc<128x32xf16, #A_SHARED> -> !tt.memdesc<128x32xf16, #A_SHARED>
  gpu.barrier
  // CHECK-NEXT: %3 -> %3
  %cst_1 = triton_gpu.local_alloc : () -> !tt.memdesc<128x32xf16, #A_SHARED>
  // CHECK-NEXT: %5 -> %0,%1,%3
  // CHECK-NEXT: %6 -> %0,%1,%3
  // CHECK-NEXT: %7 -> %0,%1,%3
  cf.br ^bb1(%arg0, %cst, %cst_0, %cst_1 : index, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>)
^bb1(%1: index, %2: !tt.memdesc<128x32xf16, #A_SHARED>, %3: !tt.memdesc<128x32xf16, #A_SHARED>, %4: !tt.memdesc<128x32xf16, #A_SHARED>):  // 2 preds: ^bb0, ^bb2
  %5 = arith.cmpi slt, %1, %arg1 : index
  cf.cond_br %5, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  gpu.barrier
  %8 = arith.addi %1, %arg2 : index
  cf.br ^bb1(%8, %4, %2, %3 : index, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>, !tt.memdesc<128x32xf16, #A_SHARED>)
^bb3:  // pred: ^bb1
  gpu.barrier
  // CHECK-NEXT: %10 -> %0
  %9 = triton_gpu.memdesc_subview %0[%idx, %idx] : !tt.memdesc<128x32xf16, #A_SHARED> -> !tt.memdesc<128x32xf16, #A_SHARED>
  tt.return
}

}  // module
