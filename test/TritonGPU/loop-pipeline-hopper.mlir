// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=compute-capability=90 -canonicalize | FileCheck --dump-input-context=50 %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

// CHECK-LABEL: tt.func @matmul_loop
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK: %[[ABUFFER:.*]] = triton_gpu.local_alloc
// CHECK: %[[BBUFFER:.*]] = triton_gpu.local_alloc
// CHECK-DAG: %[[LOOP_COND_0:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK-DAG: %[[ASUB:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]] : !tt.memdesc<2x128x32xf16, #shared, mutable> -> !tt.memdesc<128x32xf16, #shared, mutable>
// CHECK: %[[T_A0:.*]] = triton_gpu.async_copy_global_to_local %{{.*}}, %[[ASUB]] mask %[[LOOP_COND_0_SPLAT_A]] {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1> -> <128x32xf16, #shared, mutable>
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK-DAG: %[[BSUB:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: %[[T_B0:.*]] = triton_gpu.async_copy_global_to_local %{{.*}}, %[[BSUB]] mask %[[LOOP_COND_0_SPLAT_B]] other %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128x!tt.ptr<f16, 1>, #blocked> -> <32x128xf16, #shared1, mutable>
// CHECK-DAG: %[[IV_1:.*]] = arith.addi %[[LB]], %[[STEP:.*]]
// CHECK-DAG: %[[LOOP_COND_1:.*]] = arith.cmpi slt, %[[IV_1]], %[[UB]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK-DAG: %[[ASUB1:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: %[[T_A1:.*]] = triton_gpu.async_copy_global_to_local %{{.*}}, %[[ASUB1]] mask %[[LOOP_COND_1_SPLAT_A]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK-DAG: %[[BSUB1:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: %[[T_B1:.*]] = triton_gpu.async_copy_global_to_local %{{.*}}, %[[BSUB1]] mask %[[LOOP_COND_1_SPLAT_B]]
// CHECK-DAG: %[[A0:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK-DAG: %[[B0:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK-DAG:   triton_gpu.async_wait {{.*}} {num = 2 : i32}
// CHECK: scf.for {{.*}} iter_args({{.*}}, %[[INS_IDX:.*]] = %[[CONSTANT_1]], %[[EXT_IDX:.*]] = %[[CONSTANT_0]]{{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]]
// CHECK:   %[[arg_a0_dot_op:.*]] = triton_gpu.local_load %[[arg_a0]]
// CHECK:   %[[arg_b0_dot_op_0:.*]] = triton_gpu.local_load %[[arg_b0]]
// CHECK:   tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op_0]], {{.*}}
// CHECK-DAG: %[[INS_IDX_2:.*]] = arith.addi %[[INS_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_INS:.*]] = arith.cmpi slt, %[[INS_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[INS_IDX_3:.*]] = arith.select %[[CMP_INS]], %[[INS_IDX_2]], %[[CONSTANT_0]]
// CHECK:   %[[ASUB3:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   %[[NEXT_A_BUFFER:.*]] = triton_gpu.async_copy_global_to_local {{.*}}, %[[ASUB3]]
// CHECK:   %[[BSUB3:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   %[[NEXT_B_BUFFER:.*]] = triton_gpu.async_copy_global_to_local {{.*}}, %[[BSUB3]]
// CHECK-DAG: %[[EXT_IDX_2:.*]] = arith.addi %[[EXT_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_EXT:.*]] = arith.cmpi slt, %[[EXT_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[EXT_IDX_3:.*]] = arith.select %[[CMP_EXT]], %[[EXT_IDX_2]], %[[CONSTANT_0]]
// CHECK-DAG: %[[NEXT_A:.*]] = triton_gpu.memdesc_subview %{{.+}}[%[[EXT_IDX_3]],
// CHECK-DAG: %[[NEXT_B:.*]] = triton_gpu.memdesc_subview %{{.+}}[%[[EXT_IDX_3]],
// CHECK-DAG: triton_gpu.async_wait {{.*}} {num = 2 : i32}
// CHECK:   scf.yield {{.*}}, %[[INS_IDX_3]], %[[EXT_IDX_3]], %[[NEXT_A]], %[[NEXT_B]]
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
tt.func @matmul_loop(%lb : index, %ub : index, %step : index,
                       %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                       %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  // A ptrs
  %a_ptr_splat = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32, #ALs0> -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32, #AL> -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<128xi32, #BLs0> -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : tensor<1x128xi32, #BL> -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>


  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return
}
}

// -----

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

// CHECK-LABEL: tt.func @matmul_loop_nested
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK:   %[[ABUFFER:.*]] = triton_gpu.local_alloc
// CHECK:   %[[BBUFFER:.*]] = triton_gpu.local_alloc
// CHECK:   triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK: scf.for
// CHECK-DAG:   %[[A0:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]],
// CHECK-DAG:   %[[B0:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]],
// CHECK-DAG:   triton_gpu.async_wait {{.*}} {num = 2 : i32}
// CHECK:   scf.for {{.*}} iter_args({{.*}}, %[[INS_IDX:.*]] = %[[CONSTANT_1]], %[[EXT_IDX:.*]] = %[[CONSTANT_0]]{{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]]
// CHECK:   %[[arg_a0_dot_op:.*]] = triton_gpu.local_load %[[arg_a0]]
// CHECK:   %[[arg_b0_dot_op_0:.*]] = triton_gpu.local_load %[[arg_b0]]
// CHECK:   tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op_0]], {{.*}}
// CHECK-DAG: %[[INS_IDX_2:.*]] = arith.addi %[[INS_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_INS:.*]] = arith.cmpi slt, %[[INS_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[INS_IDX_3:.*]] = arith.select %[[CMP_INS]], %[[INS_IDX_2]], %[[CONSTANT_0]]
// CHECK:   triton_gpu.memdesc_subview %[[ABUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[BBUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK-DAG: %[[EXT_IDX_2:.*]] = arith.addi %[[EXT_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_EXT:.*]] = arith.cmpi slt, %[[EXT_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[EXT_IDX_3:.*]] = arith.select %[[CMP_EXT]], %[[EXT_IDX_2]], %[[CONSTANT_0]]
// CHECK-DAG: %[[NEXT_A:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[EXT_IDX_3]]
// CHECK-DAG: %[[NEXT_B:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[EXT_IDX_3]]
// CHECK-DAG: triton_gpu.async_wait {{.*}} {num = 2 : i32}
// CHECK:   scf.yield {{.*}}, %[[INS_IDX_3]], %[[EXT_IDX_3]], %[[NEXT_A]], %[[NEXT_B]]
// CHECK:   triton_gpu.async_wait {num = 0 : i32}
// CHECK:   triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK    scf.yield
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
tt.func @matmul_loop_nested(%lb : index, %ub : index, %step : index,
                              %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                              %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  scf.for %iv0 = %lb to %ub step %step {
    // A ptrs
    %a_ptr_splat = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
    %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
    %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32, #ALs0> -> tensor<1x32xi32, #AL>
    %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32, #AL> -> tensor<128x32xi32, #AL>
    %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    // B ptrs
    %b_ptr_splat = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
    %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
    %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<128xi32, #BLs0> -> tensor<1x128xi32, #BL>
    %b_offs = tt.broadcast %b_tmp1 : tensor<1x128xi32, #BL> -> tensor<32x128xi32, #BL>
    %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

    %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
    %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
    %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
    %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

    %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

    scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
      %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
      %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
      %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
      %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

      %c = tt.dot %a, %b, %prev_c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

      %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
      %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
      scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
    }
  }
  tt.return
}
}

// -----

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
// CHECK-LABEL: tt.func @matmul_loop_single_pipeline
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK: %[[BBUFFER:.*]] = triton_gpu.local_alloc
// CHECK: triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK-DAG: %[[B0:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]]
// CHECK-DAG: triton_gpu.async_wait {{.*}} {num = 1 : i32}
// CHECK:   scf.for {{.*}} iter_args({{.*}}, %[[INS_IDX:.*]] = %[[CONSTANT_1]], %[[EXT_IDX:.*]] = %[[CONSTANT_0]]{{.*}}, %[[arg_b0:.*]] = %[[B0]]
// CHECK:   %[[arg_b0_dot_op:.*]] = triton_gpu.local_load %[[arg_b0]]
// CHECK:   tt.dot {{.*}}, %[[arg_b0_dot_op]], {{.*}}
// CHECK-DAG: %[[INS_IDX_2:.*]] = arith.addi %[[INS_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_INS:.*]] = arith.cmpi slt, %[[INS_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[INS_IDX_3:.*]] = arith.select %[[CMP_INS]], %[[INS_IDX_2]], %[[CONSTANT_0]]
// CHECK:     triton_gpu.memdesc_subview %[[BBUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:     triton_gpu.async_copy_global_to_local
// CHECK-DAG: %[[EXT_IDX_2:.*]] = arith.addi %[[EXT_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_EXT:.*]] = arith.cmpi slt, %[[EXT_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[EXT_IDX_3:.*]] = arith.select %[[CMP_EXT]], %[[EXT_IDX_2]], %[[CONSTANT_0]]
// CHECK-DAG: %[[NEXT_B:.*]] = triton_gpu.memdesc_subview %{{.+}}[%[[EXT_IDX_3]]
// CHECK-DAG: triton_gpu.async_wait {{.*}} {num = 1 : i32}
// CHECK:   scf.yield {{.*}}, %[[INS_IDX_3]], %[[EXT_IDX_3]], %[[NEXT_B]]
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
tt.func @matmul_loop_single_pipeline(%lb : index, %ub : index, %step : index,
                                       %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                       %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  // A ptrs
  %a_ptr_splat = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32, #ALs0> -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32, #AL> -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<128xi32, #BLs0> -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : tensor<1x128xi32, #BL> -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  scf.for %iv = %lb to %ub step %step iter_args(%b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %c = tt.dot %a, %b, %prev_c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_b_ptr, %c : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return
}
}

// -----

// TODO: MCast is not supported yet
//// 4 warps, TMA Load
//// matmul: 128x32 @ 32x128 -> 128x128
//#C = #triton_gpu.nvidia_mma<{versionMajor = 3, warpsPerCTA = [4, 1]}>
//#SA = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset=true}>
//#SB = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], hasLeadingOffset=true}>
//#BA = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
//#BB = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
//// C-HECK: func @matmul_loop
//// C-HECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
//// C-HECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
//// C-HECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
//// C-HECK-DAG: %[[CONSTANT_3:.*]] = arith.constant 3 : i32
//// C-HECK: %[[MBARRIER_AB:.*]] = triton_nvidia_gpu.alloc_mbarrier {count = 1 : i32}
//// C-HECK: %[[EMPTY_BARRIER_B:.*]] = triton_nvidia_gpu.alloc_mbarrier {count = 2 : i32}
//// C-HECK: %[[ABUFFER:.*]] = triton_gpu.alloc
//// C-HECK: %[[MBARRIER_AB0:.*]] = triton_nvidia_gpu.extract_mbarrier %[[MBARRIER_AB]][%c0_i32]
//// C-HECK: triton_nvidia_gpu.mbarrier_arrive %[[MBARRIER_AB0]]
//// C-HECK: %[[A0BUFFER:.*]] = triton_nvidia_gpu.insert_slice_tma {{.*}}, {{.*}}, %[[CONSTANT_0]], %[[MBARRIER_AB0]]
//// C-HECK: %[[BBUFFER:.*]] = triton_gpu.alloc
//// C-HECK: %[[EMPTY_BARRIER_B0:.*]] = triton_nvidia_gpu.extract_mbarrier %[[EMPTY_BARRIER_B]][%c0_i32]
//// C-HECK: triton_nvidia_gpu.mbarrier_wait %[[EMPTY_BARRIER_B0]], %true
//// C-HECK: %[[B0BUFFER:.*]] = triton_nvidia_gpu.insert_slice_tma {{.*}}, {{.*}}, %[[CONSTANT_0]], %[[MBARRIER_AB0]]
//// C-HECK: %[[MBARRIER_AB1:.*]] = triton_nvidia_gpu.extract_mbarrier %[[MBARRIER_AB]][%c1_i32]
//// C-HECK: triton_nvidia_gpu.mbarrier_arrive %[[MBARRIER_AB1]]
//// C-HECK: %[[A1BUFFER:.*]] = triton_nvidia_gpu.insert_slice_tma {{.*}}, {{.*}}, %[[CONSTANT_1]], %[[MBARRIER_AB1]]
//// C-HECK: %[[B1BUFFER:.*]] = triton_nvidia_gpu.insert_slice_tma {{.*}}, {{.*}}, %[[CONSTANT_1]], %[[MBARRIER_AB1]]
//// C-HECK: %[[A0:.*]] = triton_gpu.extract_slice %[[A1BUFFER]][0, 0, 0]
//// C-HECK: %[[B0:.*]] = triton_gpu.extract_slice %[[B1BUFFER]][0, 0, 0]
//// C-HECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, {{.*}}, {{.*}}, %[[PIPELINE_IDX:.*]] = %[[CONSTANT_2]], %[[LOOP_IDX:.*]] = %[[CONSTANT_0]]
//  // C-HECK: %[[MBARRIER_AB_ITER:.*]] = triton_nvidia_gpu.extract_mbarrier %[[MBARRIER_AB]][{{.*}}]
//  // C-HECK: triton_nvidia_gpu.mbarrier_wait %[[MBARRIER_AB_ITER]], {{.*}}
//  // C-HECK: triton_nvidia_gpu.dot_async %[[arg_a0]], %[[arg_b0]], {{.*}}
//  // C-HECK: triton_nvidia_gpu.dot_wait {{.*}}
//  // C-HECK: %[[EMPTY_BARRIER_B_ITER_ARRIVE:.*]] = triton_nvidia_gpu.extract_mbarrier %[[EMPTY_BARRIER_B]][{{.*}}]
//  // C-HECK: triton_nvidia_gpu.mbarrier_arrive %[[EMPTY_BARRIER_B_ITER_ARRIVE]]
//  // C-HECK: %[[MBARRIER_AB_NEXT_ITER:.*]] = triton_nvidia_gpu.extract_mbarrier %[[MBARRIER_AB]][{{.*}}]
//  // C-HECK: %[[NEXT_A_BUFFER:.*]] = triton_nvidia_gpu.insert_slice_tma {{.*}}, {{.*}}, {{.*}}, %[[MBARRIER_AB_NEXT_ITER]]
//  // C-HECK: %[[NEXT_A:.*]] = triton_gpu.extract_slice %[[NEXT_A_BUFFER]][{{.*}}, 0, 0]
//  // C-HECK: %[[EMPTY_BARRIER_B_ITER_WAIT:.*]] = triton_nvidia_gpu.extract_mbarrier %[[EMPTY_BARRIER_B]][{{.*}}]
//  // C-HECK: triton_nvidia_gpu.mbarrier_wait %[[EMPTY_BARRIER_B_ITER_WAIT]], {{.*}}
//  // C-HECK: %[[NEXT_B_BUFFER:.*]] = triton_nvidia_gpu.insert_slice_tma {{.*}}, {{.*}}, {{.*}}, %[[MBARRIER_AB_NEXT_ITER]]
//  // C-HECK: %[[NEXT_B:.*]] = triton_gpu.extract_slice %[[NEXT_B_BUFFER]][{{.*}}, 0, 0]
//  // C-HECK: scf.yield {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_BUFFER]], %[[NEXT_B_BUFFER]], %[[NEXT_A]], %[[NEXT_B]], {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}
//module attributes {"triton_gpu.num-ctas" = 2 : i32, "triton_gpu.num-warps" = 4 : i32} {
//  tt.func @matmul_loop(%lb : index, %ub : index, %step : index,
//                         %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
//                         %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (!tt.ptr<tensor<128x32xf16>, 1>, !tt.ptr<tensor<32x128xf16>, 1>, tensor<128x128xf32, #C>) {
//    %c0 = arith.constant 0 : i32
//    %c32_i32 = arith.constant 32 : i32
//    %c1 = arith.constant 1 : i64
//    %c32 = arith.constant 32 : i64
//    %c128 = arith.constant 128 : i64
//    %a_tileptr_init = tt.make_tensor_ptr %A, [%c128, %c32], [%c32, %c1], [%c0, %c0] { order = array<i32: 1, 0> } : !tt.ptr<tensor<128x32xf16>, 1>
//    %b_tileptr_init = tt.make_tensor_ptr %B, [%c32, %c128], [%c1, %c32], [%c0, %c0] { order = array<i32: 0, 1> } : !tt.ptr<tensor<32x128xf16>, 1>
//    %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
//
//    %res:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_tileptr = %a_tileptr_init, %b_tileptr = %b_tileptr_init, %prev_c = %c_init) -> (!tt.ptr<tensor<128x32xf16>, 1>, !tt.ptr<tensor<32x128xf16>, 1>, tensor<128x128xf32, #C>) {
//      %a = tt.load %a_tileptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x32xf16>, 1> -> tensor<128x32xf16, #BA>
//      %b = tt.load %b_tileptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x128xf16>, 1> -> tensor<32x128xf16, #BB>
//
//      %sa = triton_gpu.local_alloc %a : (tensor<128x32xf16, #BA>) -> !tt.memdesc<128x32xf16, #SA>
//      %sb = triton_gpu.local_alloc %b : (tensor<32x128xf16, #BB>) -> !tt.memdesc<32x128xf16, #SB>
//      %c = triton_gpu_nvidia.dot_async %sa, %sb, %prev_c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #SA> * tensor<32x128xf16, #SB> -> tensor<128x128xf32, #C>
//
//      %a_tileptr_next = tt.advance %a_tileptr, [%c0, %c32_i32] : !tt.ptr<tensor<128x32xf16>, 1>
//      %b_tileptr_next = tt.advance %b_tileptr, [%c32_i32, %c0] : !tt.ptr<tensor<32x128xf16>, 1>
//
//      scf.yield %a_tileptr_next, %b_tileptr_next, %c : !tt.ptr<tensor<128x32xf16>, 1>, !tt.ptr<tensor<32x128xf16>, 1>, tensor<128x128xf32, #C>
//    }
//    tt.return %res#0, %res#1, %res#2 : !tt.ptr<tensor<128x32xf16>, 1>, !tt.ptr<tensor<32x128xf16>, 1>, tensor<128x128xf32, #C>
//  }
//}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: dot_chained_single_load
  tt.func @dot_chained_single_load(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}) -> tensor<128x64xf32, #mma> {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16, 1>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16, 1>, i64
    %2 = tt.splat %1 : !tt.ptr<f16, 1> -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %3 = tt.addptr %2, %cst_1 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1> -> tensor<128x64x!tt.ptr<f16, 1>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16, 1>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16, 1> -> tensor<1x16x!tt.ptr<f16, 1>, #blocked>
    %11 = tt.addptr %10, %cst_0 : tensor<1x16x!tt.ptr<f16, 1>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16, 1>, #blocked> -> tensor<64x16x!tt.ptr<f16, 1>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: scf.for
    // CHECK:   triton_gpu.async_wait {{.*}} {num = 1 : i32}
    // CHECK:   triton_nvidia_gpu.dot_async
    // CHECK-NEXT: triton_nvidia_gpu.dot_wait {{.*}} {pendings = 0 : i32}
    // CHECK:   triton_nvidia_gpu.dot_async
    // CHECK:   triton_gpu.async_copy_global_to_local
    // CHECK:   triton_gpu.async_commit_group
    // CHECK:   scf.yield
    %17:2 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_3, %arg5 = %16) -> (tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16, 1>, #blocked>)  : i32 {
      %18 = tt.load %arg5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x16xf16, #blocked>
      %19 = triton_gpu.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !tt.memdesc<128x64xf16, #shared>
      %20 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared1>
      %21 = tt.dot %19, %20, %cst_2 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : !tt.memdesc<128x64xf16, #shared> * !tt.memdesc<64x16xf16, #shared1> -> tensor<128x16xf32, #mma1>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma1> to tensor<128x16xf16, #mma1>
      %23 = tt.trans %20 {order=array<i32: 1,0>} : !tt.memdesc<64x16xf16, #shared1> -> !tt.memdesc<16x64xf16, #shared>
      %24 = triton_gpu.convert_layout %22 : tensor<128x16xf16, #mma1> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
      %25 = tt.dot %24, %23, %arg4 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<16x64xf16, #shared> -> tensor<128x64xf32, #mma>
      %26 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<64x16xi32, #blocked>
      scf.yield %25, %26 : tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16, 1>, #blocked>
    }
    tt.return %17#0 : tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: two_accumulator_escape
  tt.func @two_accumulator_escape(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}) -> (tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>) {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16, 1>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16, 1>, i64
    %2 = tt.splat %1 : !tt.ptr<f16, 1> -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %3 = tt.addptr %2, %cst_1 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1> -> tensor<128x64x!tt.ptr<f16, 1>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16, 1>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16, 1> -> tensor<1x16x!tt.ptr<f16, 1>, #blocked>
    %11 = tt.addptr %10, %cst_0 : tensor<1x16x!tt.ptr<f16, 1>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16, 1>, #blocked> -> tensor<64x16x!tt.ptr<f16, 1>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<64x16xi32, #blocked>
    %18 = tt.load %16 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x16xf16, #blocked>
    %19 = triton_gpu.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !tt.memdesc<128x64xf16, #shared>
    %20 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared1>
    // CHECK: %[[ALLOC1:.+]] = triton_gpu.local_alloc
    // CHECK: %[[ALLOC2:.+]] = triton_gpu.local_alloc
    // CHECK: %[[R:.+]]:{{.+}} = scf.for
    // CHECK:   %[[DOT1:.+]] = triton_nvidia_gpu.dot_async{{.*}}
    // CHECK:   triton_gpu.async_wait {{.*}} {num = 1 : i32}
    // CHECK:   %[[TRANS:.+]] = tt.trans{{.*}} : !tt.memdesc
    // CHECK:   %[[DOT2:.+]] = triton_nvidia_gpu.dot_async{{.*}} %[[TRANS]]
    // CHECK:   triton_nvidia_gpu.dot_wait %[[DOT1]], %[[DOT2]], %[[ALLOC1]], %[[ALLOC2]], %[[TRANS]] {pendings = 2 : i32}
    // CHECK:   scf.yield
    // CHECK: %{{.*}}:2 = triton_nvidia_gpu.dot_wait %[[R]]#{{.+}}, %[[R]]#{{.+}} {pendings = 0 : i32} : tensor<128x16xf32, #{{.*}}>, tensor<128x64xf32, #{{.*}}>
    %17:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_3, %arg5 = %16, %arg6 = %cst_2) -> (tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<128x16xf32, #mma1>)  : i32 {
      %21 = tt.dot %19, %20, %arg6 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : !tt.memdesc<128x64xf16, #shared> * !tt.memdesc<64x16xf16, #shared1> -> tensor<128x16xf32, #mma1>
      %l = tt.load %arg5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x16xf16, #blocked>
      %c = triton_gpu.local_alloc %l : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared1>
      %23 = tt.trans %c {order=array<i32: 1,0>} : !tt.memdesc<64x16xf16, #shared1> -> !tt.memdesc<16x64xf16, #shared>
      %25 = tt.dot %cst_4, %23, %arg4 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<16x64xf16, #shared> -> tensor<128x64xf32, #mma>
      %26 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<64x16xi32, #blocked>
      scf.yield %25, %26, %21 : tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<128x16xf32, #mma1>
    }
    tt.return %17#0, %17#2 : tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 2, maxPhase = 4, order = [0, 1], hasLeadingOffset = true}>

// Make sure that if one of the load dot operand is not pipelined (and therefore not double buffered) we won't use
// async dot.
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: no_wgmma_pipeline
  tt.func public @no_wgmma_pipeline(%arg0: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant dense<512> : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<512> : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %cst_2 = arith.constant dense<512> : tensor<128x1xi32, #blocked>
    %cst_3 = arith.constant dense<512> : tensor<128x1xi32, #blocked1>
    %cst_4 = arith.constant dense<512> : tensor<64x1xi32, #blocked1>
    %cst_5 = arith.constant dense<32768> : tensor<64x256xi32, #blocked1>
    %cst_6 = arith.constant dense<64> : tensor<128x64xi32, #blocked>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = arith.remsi %0, %cst_0 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %3 = arith.remsi %2, %cst_1 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %4 = tt.expand_dims %1 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %5 = arith.muli %4, %cst_2 : tensor<128x1xi32, #blocked>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %8 = tt.broadcast %5 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %9 = tt.broadcast %7 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %10 = arith.addi %8, %9 : tensor<128x64xi32, #blocked>
    %11 = tt.splat %arg0 : !tt.ptr<f8E5M2, 1> -> tensor<128x64x!tt.ptr<f8E5M2, 1>, #blocked>
    %12 = tt.addptr %11, %10 : tensor<128x64x!tt.ptr<f8E5M2, 1>, #blocked>, tensor<128x64xi32, #blocked>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %15 = arith.muli %14, %cst_4 : tensor<64x1xi32, #blocked1>
    %16 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %17 = tt.broadcast %15 : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %18 = tt.broadcast %16 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %19 = arith.addi %17, %18 : tensor<64x256xi32, #blocked1>
    %20 = tt.splat %arg1 : !tt.ptr<f8E5M2, 1> -> tensor<64x256x!tt.ptr<f8E5M2, 1>, #blocked1>
    %21 = tt.addptr %20, %19 : tensor<64x256x!tt.ptr<f8E5M2, 1>, #blocked1>, tensor<64x256xi32, #blocked1>
    %22:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %12, %arg6 = %21) -> (tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f8E5M2, 1>, #blocked>, tensor<64x256x!tt.ptr<f8E5M2, 1>, #blocked1>)  : i32 {
      %35 = tt.load %arg5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf8E5M2, #blocked>
      %36 = tt.load %arg6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x256xf8E5M2, #blocked1>
      %37 = triton_gpu.local_alloc %35 : (tensor<128x64xf8E5M2, #blocked>) -> !tt.memdesc<128x64xf8E5M2, #shared>
      %38 = triton_gpu.local_alloc %36 : (tensor<64x256xf8E5M2, #blocked1>) -> !tt.memdesc<64x256xf8E5M2, #shared1>
      // CHECK: triton_gpu.local_alloc
      // CHECK: scf.for
      // CHECK:   triton_nvidia_gpu.dot_async
      // CHECK-NEXT: triton_nvidia_gpu.dot_wait
      %39 = tt.dot %37, %38, %arg4 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !tt.memdesc<128x64xf8E5M2, #shared> * !tt.memdesc<64x256xf8E5M2, #shared1> -> tensor<128x256xf32, #mma>
      %40 = tt.addptr %arg5, %cst_6 : tensor<128x64x!tt.ptr<f8E5M2, 1>, #blocked>, tensor<128x64xi32, #blocked>
      %41 = tt.addptr %arg6, %cst_5 : tensor<64x256x!tt.ptr<f8E5M2, 1>, #blocked1>, tensor<64x256xi32, #blocked1>
      scf.yield %39, %40, %41 : tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f8E5M2, 1>, #blocked>, tensor<64x256x!tt.ptr<f8E5M2, 1>, #blocked1>
    }
    %23 = arith.truncf %22#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %24 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %25 = tt.expand_dims %24 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %26 = arith.muli %25, %cst_3 : tensor<128x1xi32, #blocked1>
    %27 = tt.splat %arg2 : !tt.ptr<f8E5M2, 1> -> tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked1>
    %28 = tt.addptr %27, %26 : tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked1>, tensor<128x1xi32, #blocked1>
    %29 = tt.expand_dims %2 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %30 = tt.broadcast %28 : tensor<128x1x!tt.ptr<f8E5M2, 1>, #blocked1> -> tensor<128x256x!tt.ptr<f8E5M2, 1>, #blocked1>
    %31 = tt.broadcast %29 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %32 = tt.addptr %30, %31 : tensor<128x256x!tt.ptr<f8E5M2, 1>, #blocked1>, tensor<128x256xi32, #blocked1>
    %33 = tt.fp_to_fp %23 {rounding = 1 : i32} : tensor<128x256xf16, #mma> -> tensor<128x256xf8E5M2, #mma>
    %34 = triton_gpu.convert_layout %33 : tensor<128x256xf8E5M2, #mma> -> tensor<128x256xf8E5M2, #blocked1>
    tt.store %32, %34 {cache = 1 : i32, evict = 1 : i32} : tensor<128x256xf8E5M2, #blocked1>
    tt.return
  }
}

// -----

// A dot can be properly async if all its uses follow a synchronous MMAv3 dot.
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: async_following_sync
  tt.func @async_following_sync(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}) -> (tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>) {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32

    // Add a "dummy" early return here to test that we don't crash in the
    // presence of unstructured control flow.
    %cond = arith.constant 0 : i1
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %zero = arith.constant 0.0 : f32
    %t1 = tt.splat %zero : f32 -> tensor<128x64xf32, #mma>
    %t2 = tt.splat %zero : f32 -> tensor<128x16xf32, #mma1>
    tt.return %t1, %t2 : tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>
  ^bb2:  // pred: ^bb0

    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16, 1>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16, 1>, i64
    %2 = tt.splat %1 : !tt.ptr<f16, 1> -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %3 = tt.addptr %2, %cst_1 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1> -> tensor<128x64x!tt.ptr<f16, 1>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16, 1>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16, 1> -> tensor<1x16x!tt.ptr<f16, 1>, #blocked>
    %11 = tt.addptr %10, %cst_0 : tensor<1x16x!tt.ptr<f16, 1>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16, 1>, #blocked> -> tensor<64x16x!tt.ptr<f16, 1>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<64x16xi32, #blocked>
    %18 = tt.load %16 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x16xf16, #blocked>
    %19 = triton_gpu.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !tt.memdesc<128x64xf16, #shared>
    %20 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared1>
    // CHECK:          %[[LOOP:[^ :]+]]{{.*}} scf.for {{.*}} iter_args(%[[PREV_DOT2:[^ ]+]]
    // CHECK-NOT:        triton_nvidia_gpu.dot_wait
    // CHECK:            %[[DOT0:.+]] = triton_nvidia_gpu.dot_async
    // CHECK-NOT:        triton_nvidia_gpu.dot_wait
    // CHECK:            %[[DOT1:.+]] = triton_nvidia_gpu.dot_async
    // CHECK-NEXT:       triton_nvidia_gpu.dot_wait
    // CHECK-DAG-SAME:     %[[DOT0]]
    // CHECK-DAG-SAME:     %[[DOT1]]
    // CHECK-DAG-SAME:     %[[PREV_DOT2]]
    // CHECK-SAME:         {pendings = 0 : i32}
    // CHECK:            %[[DOT2:.+]] = triton_nvidia_gpu.dot_async
    // CHECK-NOT:        triton_nvidia_gpu.dot_wait
    // CHECK:          scf.yield %[[DOT2]]
    // CHECK:          triton_nvidia_gpu.dot_wait %[[LOOP]]#3, %[[LOOP]]#0 {pendings = 0 : i32}
    %17:4 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%prev_dot2 = %cst_3, %arg5 = %16, %prev_dot1 = %cst_2, %prev_dot0 = %cst_2) -> (tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1>)  : i32 {
      // This one can be async.
      %dot0 = tt.dot %19, %20, %prev_dot1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : !tt.memdesc<128x64xf16, #shared> * !tt.memdesc<64x16xf16, #shared1> -> tensor<128x16xf32, #mma1>
      // This can't be async because its result is modified before it's yielded.
      %dot1 = tt.dot %19, %20, %prev_dot1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : !tt.memdesc<128x64xf16, #shared> * !tt.memdesc<64x16xf16, #shared1> -> tensor<128x16xf32, #mma1>
      %dot1.1 = arith.addf %dot1, %dot1 : tensor<128x16xf32, #mma1>
      %l = tt.load %arg5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x16xf16, #blocked>
      %c = triton_gpu.local_alloc %l : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared1>
      %23 = tt.trans %c {order=array<i32: 1,0>} : !tt.memdesc<64x16xf16, #shared1> -> !tt.memdesc<16x64xf16, #shared>
      // This dot can be async even though %prev_dot2 is not used directly by an
      // async dot, because that use follows the synchronous dot above.
      %prev_dot2.1 = arith.addf %prev_dot2, %prev_dot2 : tensor<128x64xf32, #mma>
      %dot2 = tt.dot %cst_4, %23, %prev_dot2.1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<16x64xf16, #shared> -> tensor<128x64xf32, #mma>
      %26 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<64x16xi32, #blocked>
      scf.yield %dot2, %26, %dot1.1, %dot0 : tensor<128x64xf32, #mma>, tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<128x16xf32, #mma1>, tensor<128x16xf32, #mma1>
    }
    tt.return %17#0, %17#2 : tensor<128x64xf32, #mma>, tensor<128x16xf32, #mma1>
  }
}
