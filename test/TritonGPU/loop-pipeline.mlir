// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
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
// CHECK:   %[[arg_b0_dot_op_1:.*]] = arith.mulf %[[arg_b0_dot_op_0]]
// CHECK:   tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op_1]], {{.*}}
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
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.compute-capability" = 80} {
tt.func @matmul_loop(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
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

  %b_scale = arith.constant dense<4.> : tensor<32x128xf16, #B>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    %b__ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b_ = triton_gpu.convert_layout %b__ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %b = arith.mulf %b_, %b_scale: tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

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
tt.func @matmul_loop_nested(%lb : index, %ub : index, %step : index,
                         %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                         %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C>{

  %c_start = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %loop1:1 = scf.for %iv0 = %lb to %ub step %step iter_args(%c_init = %c_start) -> (tensor<128x128xf32, #C>) {
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

    %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

    %loop2:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
      %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
      %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
      %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
      %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

      %c = tt.dot %a, %b, %prev_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

      %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
      %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
      scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
    }

    scf.yield %loop2#2 : tensor<128x128xf32, #C>
  }
  tt.return %loop1#0 : tensor<128x128xf32, #C>
}

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
tt.func @matmul_loop_single_pipeline(%lb : index, %ub : index, %step : index,
                                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
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

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_b_ptr, %c : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#1 : tensor<128x128xf32, #C>
}

// CHECK-LABEL: tt.func @indirect_bmm_scalar
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_commit_group
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %{{.*}}, {{.*}}
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]]
// CHECK: %[[IND_BUFFER_0:.*]] = tt.load %{{.*}}, {{.*}}
// CHECK: %[[IND_BUFFER_1:.*]] = arith.muli {{.*}}, %[[IND_BUFFER_0]]
// CHECK: %[[IND_BUFFER_2:.*]] = tt.splat %[[IND_BUFFER_1]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[IND_BUFFER_2]]
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_0]]
// CHECK: triton_gpu.async_wait {{.*}} {num = 2 : i32}
tt.func @indirect_bmm_scalar(%77: i64 {tt.divisibility=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: !tt.ptr<i64>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>) {
    %82 = tt.load %arg20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
    %83 = tt.load %arg21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
    %84 = arith.muli %77, %83 : i64
    %85 = tt.splat %84 : i64 -> tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32 : !tt.ptr<i64>, i32
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK-LABEL: tt.func @indirect_bmm_scalar_dist_one
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_commit_group
// CHECK: scf.for %{{.*}} iter_args(%{{.*}}, %{{.*}}, %{{.*}}, %[[IND_BUFFER_PREV:.*]] = {{.*}}, %{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) ->
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %{{.*}}, {{.*}}
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]]
// CHECK: %[[IND_BUFFER_0:.*]] = tt.load %{{.*}}, {{.*}}
// CHECK: %[[IND_BUFFER_1:.*]] = arith.muli {{.*}}, %[[IND_BUFFER_PREV]]
// CHECK: %[[IND_BUFFER_2:.*]] = tt.splat %[[IND_BUFFER_1]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[IND_BUFFER_2]]
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_0]]
// CHECK: triton_gpu.async_wait {{.*}} {num = 2 : i32}
// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}, %[[IND_BUFFER_0]]
tt.func @indirect_bmm_scalar_dist_one(%77: i64 {tt.divisibility=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: !tt.ptr<i64>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %50 = tt.load %75 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
  %51 = tt.addptr %75, %c1_i32 : !tt.ptr<i64>, i32
  %79:4 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %51, %arg22 = %50) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>, i64) {
    %82 = tt.load %arg20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
    %83 = tt.load %arg21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
    %84 = arith.muli %77, %arg22 : i64
    %85 = tt.splat %84 : i64 -> tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32 : !tt.ptr<i64>, i32
    scf.yield %90, %91, %92, %83 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>, i64
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK-LABEL: tt.func @indirect_bmm_vector
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_commit_group
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %{{.*}}, {{.*}}
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]]
// CHECK: %[[IND_BUFFER_0:.*]] = triton_gpu.memdesc_subview
// CHECK: %[[IND_BUFFER_1:.*]] = triton_gpu.local_load %[[IND_BUFFER_0]]
// CHECK: %[[IND_BUFFER_2:.*]] = tt.expand_dims %[[IND_BUFFER_1]] {axis = 1 : i32}
// CHECK: %[[IND_BUFFER_3:.*]] = tt.broadcast %[[IND_BUFFER_2]]
// CHECK: %[[IND_BUFFER_4:.*]] = arith.muli {{.*}}, %[[IND_BUFFER_3]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[IND_BUFFER_4]]
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_0]]
// CHECK: triton_gpu.async_wait {{.*}} {num = 1 : i32}
tt.func @indirect_bmm_vector(%77: tensor<16x16xi64, #BL> {tt.divisibility=16: i32, tt.constancy=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: tensor<16x!tt.ptr<i64>, #BLs1>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_splat = tt.splat %c1_i32 : i32 -> tensor<16xi32, #BLs1>
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>) {
    %82 = tt.load %arg20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
    %83 = tt.load %arg21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16xi64, #BLs1>
    %84 = tt.expand_dims %83 {axis=1: i32}: tensor<16xi64, #BLs1> -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : tensor<16x1xi64, #BL> -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK-LABEL: tt.func @post_load_inv
// CHECK: scf.for
// CHECK-DAG: %[[IV:.*]] = arith.index_cast
// CHECK: %[[NEXT_IV:.*]] = arith.addi %[[IV]], %c1_i32 : i32
// CHECK: arith.index_cast
// CHECK-NOT: arith.addi %[[NEXT_IV]]
tt.func @post_load_inv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg3: i32 {tt.divisibility = 16 : i32},
                       %arg4: i32 {tt.divisibility = 16 : i32},
                       %arg5: i32 {tt.divisibility = 16 : i32},
                       %arg6: i32 {tt.divisibility = 16 : i32},
                       %arg7: i32 {tt.divisibility = 16 : i32},
                       %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #C> {
  %c0_index = arith.constant 0 : index
  %c1_index = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %84 = arith.constant 900 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #C>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #AL>
  %50 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #AL>
  %59 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %81 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %66 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #AL>
  %60 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %82 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %85:3 = scf.for %arg9 = %c0_index to %84 step %c1_index iter_args(%arg10 = %cst, %arg11 = %59, %arg12 = %81) -> (tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>)  {
    %130 = arith.index_cast %arg9 : index to i32
    %107 = arith.muli %130, %c32_i32 : i32
    %108 = arith.subi %arg5, %107 : i32
    %109 = tt.splat %108 : i32 -> tensor<1x32xi32, #AL>
    %110 = arith.cmpi "slt", %50, %109 : tensor<1x32xi32, #AL>
    %111 = tt.broadcast %110 : tensor<1x32xi1, #AL> -> tensor<32x32xi1, #AL>
    %112 = tt.load %arg11, %111, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %113 = tt.splat %108 : i32 -> tensor<32x1xi32, #AL>
    %114 = arith.cmpi "slt", %66, %113 : tensor<32x1xi32, #AL>
    %115 = tt.broadcast %114 : tensor<32x1xi1, #AL> -> tensor<32x32xi1, #AL>
    %116 = tt.load %arg12, %115, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %117 = triton_gpu.convert_layout %112 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>>
    %118 = triton_gpu.convert_layout %116 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>>
    %119 = tt.dot %117, %118, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>> -> tensor<32x32xf32, #C>
    %131 = arith.index_cast %arg9 : index to i32
    %120 = arith.addi %131, %c1_i32 : i32
    %121 = arith.muli %120, %c32_i32 : i32
    %122 = tt.splat %121 : i32 -> tensor<32x32xi32, #AL>
    %123 = tt.addptr %60, %122 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %124 = arith.muli %121, %arg7 : i32
    %125 = tt.splat %124 : i32 -> tensor<32x32xi32, #AL>
    %126 = tt.addptr %82, %125 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    scf.yield %119, %123, %126 : tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>
  }
  tt.return %85#0 : tensor<32x32xf32, #C>
}

// CHECK-LABEL: tt.func @cross_iter_dep
// TODO: enable pipelining with distance of 2
// CHECK-NOT: triton_gpu.async_commit_group
// CHECK: scf.for
// CHECK: scf.yield
tt.func @cross_iter_dep(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg3: i32 {tt.divisibility = 16 : i32},
                        %arg4: i32 {tt.divisibility = 16 : i32},
                        %arg5: i32 {tt.divisibility = 16 : i32},
                        %arg6: i32 {tt.divisibility = 16 : i32},
                        %arg7: i32 {tt.divisibility = 16 : i32},
                        %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #C> {
  %c0_i32 = arith.constant 0 : index
  %118 = arith.constant 32 : index
  %c1_i32 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #C>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #AL>
  %78 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %110 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %112 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %113 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %116 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %65 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #AL>
  %88 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #AL>
  %80 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %119:5 = scf.for %arg9 = %c0_i32 to %118 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %78, %arg12 = %110, %arg13 = %113, %arg14 = %116) -> (tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>)  {
    %161 = arith.index_cast %arg9 : index to i32
    %141 = arith.muli %161, %c32_i32 : i32
    %142 = arith.subi %arg5, %141 : i32
    %143 = tt.splat %142 : i32 -> tensor<1x32xi32, #AL>
    %144 = arith.cmpi "slt", %65, %143 : tensor<1x32xi32, #AL>
    %145 = tt.broadcast %144 : tensor<1x32xi1, #AL> -> tensor<32x32xi1, #AL>
    %146 = tt.load %arg11, %145, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %147 = tt.splat %142 : i32 -> tensor<32x1xi32, #AL>
    %148 = arith.cmpi "slt", %88, %147 : tensor<32x1xi32, #AL>
    %149 = tt.broadcast %148 : tensor<32x1xi1, #AL> -> tensor<32x32xi1, #AL>
    %150 = tt.load %arg12, %149, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #AL>
    %151 = triton_gpu.convert_layout %146 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>>
    %152 = triton_gpu.convert_layout %150 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>>
    %153 = tt.dot %151, %152, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>> -> tensor<32x32xf32, #C>
    %162 = arith.index_cast %arg9 : index to i32
    %154 = arith.addi %162, %c2_i32 : i32
    %155 = arith.muli %154, %c32_i32 : i32
    %156 = tt.splat %155 : i32 -> tensor<32x32xi32, #AL>
    %157 = tt.addptr %80, %156 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %158 = arith.muli %155, %arg7 : i32
    %159 = tt.splat %158 : i32 -> tensor<32x32xi32, #AL>
    %160 = tt.addptr %112, %159 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    scf.yield %153, %arg13, %arg14, %157, %160 : tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>
  }
  tt.return %119#0 : tensor<32x32xf32, #C>
}

// CHECK-LABEL: tt.func @dep_arg_two_uses
// CHECK: tt.expand_dims
// CHECK: tt.expand_dims
// CHECK: tt.expand_dims %arg5
// CHECK-NEXT: tt.expand_dims %arg5
// CHECK: %[[PTR0:.*]] = tt.splat %arg6
// CHECK: %[[PTR1:.*]] = tt.addptr %[[PTR0]]
// CHECK-NEXT: tt.load %[[PTR1]]
tt.func @dep_arg_two_uses(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                          %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                          %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
  %23 = arith.constant 100 : index
  %c64 = arith.constant 64 : i64
  %56 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
  %57 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
  %58 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #BL}>>
  %83 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
  %85 = tt.splat %c64 : i64 -> tensor<1x32xi64, #AL>
  %86 = tt.splat %c64 : i64 -> tensor<1x32xi64, #AL>
  %68 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %c32_index = arith.constant 32 : index
  %c32_i32 = arith.index_cast %c32_index : index to i32
  %80 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
  %cst_6 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #BL>
  %88 = arith.truncf %cst_6 : tensor<32x128xf32, #BL> to tensor<32x128xf16, #BL>
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #C>
  %90 = tt.splat %c64 : i64 -> tensor<32x128xi64, #BL>
  %92 = tt.addptr %arg1, %c32_i32 : !tt.ptr<i32>, i32
  %c0_index = arith.constant 0 : index
  %91:5 = scf.for %arg19 = %c0_index to %23 step %c32_index iter_args(%arg20 = %68, %arg21 = %83, %arg22 = %92, %arg23 = %cst, %arg24 = %80) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>, !tt.ptr<i32>, tensor<128x128xf32, #C>, tensor<32x128x!tt.ptr<f16>, #BL>)   {
    %1750 = arith.subi %23, %arg19 : index
    %175 = arith.index_cast %1750 : index to i32
    %176 = tt.splat %175 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %177 = tt.splat %175 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #BL}>>
    %178 = arith.cmpi "slt", %57, %176 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %179 = arith.cmpi "slt", %58, %177 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #BL}>>
    %180 = tt.expand_dims %178 {axis = 0 : i32} : tensor<32xi1, #triton_gpu.slice<{dim = 0, parent = #AL}>> -> tensor<1x32xi1, #AL>
    %181 = tt.expand_dims %179 {axis = 1 : i32} : tensor<32xi1, #triton_gpu.slice<{dim = 1, parent = #BL}>> -> tensor<32x1xi1, #BL>
    %182 = tt.expand_dims %arg21 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>> -> tensor<1x32xi32, #AL>
    %183 = tt.expand_dims %arg21 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>> -> tensor<1x32xi32, #AL>
    %184 = arith.extsi %182 : tensor<1x32xi32, #AL> to tensor<1x32xi64, #AL>
    %185 = arith.extsi %183 : tensor<1x32xi32, #AL> to tensor<1x32xi64, #AL>
    %186 = arith.muli %184, %85 : tensor<1x32xi64, #AL>
    %187 = arith.muli %185, %86 : tensor<1x32xi64, #AL>
    %188 = tt.broadcast %186 : tensor<1x32xi64, #AL> -> tensor<128x32xi64, #AL>
    %189 = tt.broadcast %187 : tensor<1x32xi64, #AL> -> tensor<128x32xi64, #AL>
    %190 = tt.addptr %arg20, %188 : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi64, #AL>
    %191 = tt.addptr %arg20, %189 : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi64, #AL>
    %192 = tt.broadcast %180 : tensor<1x32xi1, #AL> -> tensor<128x32xi1, #AL>
    %193 = tt.load %191, %192 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    %194 = tt.splat %arg22 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %195 = tt.addptr %194, %56 : tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #AL}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %196 = tt.load %195 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %197 = tt.addptr %arg22, %c32_i32 : !tt.ptr<i32>, i32
    %198 = tt.broadcast %181 : tensor<32x1xi1, #BL> -> tensor<32x128xi1, #BL>
    %199 = tt.load %arg24, %198, %88 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %200 = triton_gpu.convert_layout %193 : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>>
    %201 = triton_gpu.convert_layout %199 : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>>
    %202 = tt.dot %200, %201, %arg23 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>> -> tensor<128x128xf32, #C>
    %203 = tt.addptr %arg24, %90 : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi64, #BL>
    scf.yield %190, %196, %197, %202, %203 : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>, !tt.ptr<i32>, tensor<128x128xf32, #C>, tensor<32x128x!tt.ptr<f16>, #BL>
  }
  tt.return %91#3 : tensor<128x128xf32, #C>
}
}  // end module

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tt.func @load_two_users
  tt.func @load_two_users(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16, 1>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16, 1>, i64
    %2 = tt.splat %1 : !tt.ptr<f16, 1> -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %3 = tt.addptr %2, %cst_0 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1> -> tensor<128x64x!tt.ptr<f16, 1>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16, 1>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16, 1> -> tensor<1x16x!tt.ptr<f16, 1>, #blocked>
    %11 = tt.addptr %10, %cst : tensor<1x16x!tt.ptr<f16, 1>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16, 1>, #blocked> -> tensor<64x16x!tt.ptr<f16, 1>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: triton_gpu.async_wait {{.*}} {num = 1 : i32}
    // CHECK: scf.for
    // CHECK:   tt.dot
    // CHECK:   tt.dot
    // CHECK:   triton_gpu.async_copy_global_to_local
    // CHECK:   triton_gpu.async_wait {{.*}} {num = 1 : i32}
    // CHECK:   scf.yield
    // CHECK: triton_gpu.async_wait {num = 0 : i32}

    %17:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %18 = tt.load %16 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x16xf16, #blocked>
      %19 = triton_gpu.convert_layout %9 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = triton_gpu.convert_layout %18 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %19, %20, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %23 = triton_gpu.convert_layout %22 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %24 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared>
      %25 = tt.trans %24 {order=array<i32: 1,0>} : !tt.memdesc<64x16xf16, #shared> -> !tt.memdesc<16x64xf16, #shared1>
      %26 = triton_gpu.local_load %25 : !tt.memdesc<16x64xf16, #shared1> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %23, %26, %arg4 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %21, %27 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
    }
    tt.return %17#0, %17#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tt.func @load_two_users_incompatible_layouts
  tt.func @load_two_users_incompatible_layouts(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16, 1>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16, 1>, i64
    %2 = tt.splat %1 : !tt.ptr<f16, 1> -> tensor<128x1x!tt.ptr<f16, 1>, #blocked1>
    %3 = tt.addptr %2, %cst_0 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16, 1>, #blocked1> -> tensor<128x64x!tt.ptr<f16, 1>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16, 1>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16, 1> -> tensor<1x16x!tt.ptr<f16, 1>, #blocked>
    %11 = tt.addptr %10, %cst : tensor<1x16x!tt.ptr<f16, 1>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16, 1>, #blocked> -> tensor<64x16x!tt.ptr<f16, 1>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16, 1>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK-NOT: triton_gpu.insert_slice_async
    // CHECK: scf.for
    %17:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %18 = tt.load %16 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x16xf16, #blocked>
      %19 = triton_gpu.convert_layout %9 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = triton_gpu.convert_layout %18 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %19, %20, %cst_1 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %23 = triton_gpu.convert_layout %22 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %24 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared>
      %25 = tt.trans %24 {order=array<i32: 1,0>} : !tt.memdesc<64x16xf16, #shared> -> !tt.memdesc<16x64xf16, #shared1>
      %26 = triton_gpu.local_load %25 : !tt.memdesc<16x64xf16, #shared1> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %23, %26, %arg4 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %21, %27 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
    }
    tt.return %17#0, %17#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----

// CHECK-LABEL: nested_loops
// CHECK: triton_gpu.local_alloc
// CHECK: scf.for
// CHECK-NOT: triton_gpu.local_alloc
// CHECK:   scf.for
// CHECK:     scf.yield
// CHECK:   triton_gpu.async_wait {num = 0 : i32}
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.async_commit_group
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.async_commit_group
// CHECK:   scf.yield
//
// The following code has the structure:
//
// ```
// for {
//   %a = load()
//   for {
//     %b = load()
//     dot(%a, %b)
//   }
// }
// ```
//
// Only the outer for should be pipelined. The regression this tests
// causes an assertion to fail while pipelining the outer `for`, in
// particular while predicating the operations scheduled to be emitted
// in the prologue.
//
// We check that there is no allocation before the first occurrence of
// scf.for because that would mean that the first load `%a = load()`
// would be pipelined.
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @nested_loops(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32, 1> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<320> : tensor<32x1xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c10_i32 = arith.constant 10 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %3 = arith.muli %2, %cst_0 : tensor<32x1xi32, #blocked>
    %4 = tt.splat %arg1 : !tt.ptr<f32, 1> -> tensor<32x1x!tt.ptr<f32, 1>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x1x!tt.ptr<f32, 1>, #blocked>, tensor<32x1xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<32x1x!tt.ptr<f32, 1>, #blocked> -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<32x1x!tt.ptr<f32, 1>, #blocked>
    %8 = tt.splat %arg3 : !tt.ptr<f32, 1> -> tensor<32x1x!tt.ptr<f32, 1>, #blocked>
    scf.for %arg4 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %9 = arith.muli %arg4, %c32_i32 : i32
      %10 = tt.splat %9 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %11 = tt.splat %9 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %12 = arith.addi %10, %0 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %13 = arith.addi %11, %1 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %14 = tt.expand_dims %12 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %15 = tt.broadcast %14 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
      %16 = tt.addptr %6, %15 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
      %17 = tt.load %16 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
      %18 = tt.expand_dims %13 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
      %19 = arith.muli %18, %cst_0 : tensor<32x1xi32, #blocked>
      %20 = tt.addptr %7, %19 : tensor<32x1x!tt.ptr<f32, 1>, #blocked>, tensor<32x1xi32, #blocked>
      %21 = tt.broadcast %20 : tensor<32x1x!tt.ptr<f32, 1>, #blocked> -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
      %22 = tt.addptr %8, %19 : tensor<32x1x!tt.ptr<f32, 1>, #blocked>, tensor<32x1xi32, #blocked>
      %23 = tt.broadcast %22 : tensor<32x1x!tt.ptr<f32, 1>, #blocked> -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
      scf.for %arg5 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
        %24 = arith.muli %arg5, %c32_i32 : i32
        %25 = tt.splat %24 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %26 = arith.addi %25, %0 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %27 = tt.expand_dims %26 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
        %28 = tt.broadcast %27 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
        %29 = tt.addptr %21, %28 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
        %30 = tt.load %29 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
        %31 = triton_gpu.convert_layout %30 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %32 = triton_gpu.convert_layout %17 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
        %33 = tt.dot %31, %32, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
        %34 = tt.addptr %23, %28 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
        %35 = triton_gpu.convert_layout %33 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
        tt.store %34, %35 {cache = 1 : i32, evict = 1 : i32} : tensor<32x32xf32, #blocked>
      }
    }
    tt.return
  }
}  // end module

// -----

// CHECK-NOT:  triton_gpu.convert_layout {{.*}} : tensor<32x64xf32, #shared> -> tensor<32x64xf32, #shared1>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64, 1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.get_program_id y : i32
    %3 = tt.load %arg3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.splat %1 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %6 = arith.addi %5, %4 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %8 = tt.splat %3 : i64 -> tensor<64x1xi64, #blocked>
    %9 = arith.extsi %7 : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked>
    %10 = arith.addi %8, %9 : tensor<64x1xi64, #blocked>
    %11 = arith.extsi %arg5 : i32 to i64
    %12 = tt.splat %11 : i64 -> tensor<64x1xi64, #blocked>
    %13 = arith.muli %10, %12 : tensor<64x1xi64, #blocked>
    %14 = arith.muli %2, %arg5 : i32
    %15 = arith.extsi %14 : i32 to i64
    %16 = tt.splat %15 : i64 -> tensor<64x1xi64, #blocked>
    %17 = arith.addi %13, %16 : tensor<64x1xi64, #blocked>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.expand_dims %18 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %21 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %22 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked>
    %23 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %24 = arith.muli %20, %22 : tensor<1x64xi32, #blocked>
    %25 = arith.muli %21, %23 : tensor<1x64xi32, #blocked1>
    %26 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
    %27 = arith.extsi %24 : tensor<1x64xi32, #blocked> to tensor<1x64xi64, #blocked>
    %28 = arith.extsi %25 : tensor<1x64xi32, #blocked1> to tensor<1x64xi64, #blocked1>
    %29 = tt.broadcast %27 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
    %30 = arith.addi %26, %29 : tensor<64x64xi64, #blocked>
    %31 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %33 = tt.splat %3 : i64 -> tensor<32x1xi64, #blocked1>
    %34 = arith.extsi %32 : tensor<32x1xi32, #blocked1> to tensor<32x1xi64, #blocked1>
    %35 = arith.addi %33, %34 : tensor<32x1xi64, #blocked1>
    %36 = tt.splat %11 : i64 -> tensor<32x1xi64, #blocked1>
    %37 = arith.muli %35, %36 : tensor<32x1xi64, #blocked1>
    %38 = tt.splat %15 : i64 -> tensor<32x1xi64, #blocked1>
    %39 = arith.addi %37, %38 : tensor<32x1xi64, #blocked1>
    %40 = tt.broadcast %39 : tensor<32x1xi64, #blocked1> -> tensor<32x64xi64, #blocked1>
    %41 = tt.broadcast %28 : tensor<1x64xi64, #blocked1> -> tensor<32x64xi64, #blocked1>
    %42 = arith.addi %40, %41 : tensor<32x64xi64, #blocked1>
    %43 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %44 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %45 = tt.expand_dims %43 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %46 = tt.expand_dims %44 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %47 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked1>
    %48 = tt.splat %arg5 : i32 -> tensor<1x32xi32, #blocked>
    %49 = arith.muli %45, %47 : tensor<1x32xi32, #blocked1>
    %50 = arith.muli %46, %48 : tensor<1x32xi32, #blocked>
    %51 = tt.broadcast %39 : tensor<32x1xi64, #blocked1> -> tensor<32x32xi64, #blocked1>
    %52 = arith.extsi %49 : tensor<1x32xi32, #blocked1> to tensor<1x32xi64, #blocked1>
    %53 = arith.extsi %50 : tensor<1x32xi32, #blocked> to tensor<1x32xi64, #blocked>
    %54 = tt.broadcast %52 : tensor<1x32xi64, #blocked1> -> tensor<32x32xi64, #blocked1>
    %55 = arith.addi %51, %54 : tensor<32x32xi64, #blocked1>
    %56 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<64x64x!tt.ptr<f32, 1>, #blocked>
    %57 = tt.addptr %56, %30 : tensor<64x64x!tt.ptr<f32, 1>, #blocked>, tensor<64x64xi64, #blocked>
    %58 = tt.splat %arg1 : !tt.ptr<f32, 1> -> tensor<32x64x!tt.ptr<f32, 1>, #blocked1>
    %59 = tt.addptr %58, %42 : tensor<32x64x!tt.ptr<f32, 1>, #blocked1>, tensor<32x64xi64, #blocked1>
    %60 = tt.splat %arg2 : !tt.ptr<f32, 1> -> tensor<32x32x!tt.ptr<f32, 1>, #blocked1>
    %61 = tt.addptr %60, %55 : tensor<32x32x!tt.ptr<f32, 1>, #blocked1>, tensor<32x32xi64, #blocked1>
    %62 = tt.load %57 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, #blocked>
    %63 = scf.for %arg6 = %c0_i32 to %c64_i32 step %c32_i32 iter_args(%arg7 = %cst) -> (tensor<64x32xf32, #mma>)  : i32 {
      %70 = tt.load %59 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf32, #blocked1>
      %71 = triton_gpu.convert_layout %62 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %72 = triton_gpu.local_alloc %70 : (tensor<32x64xf32, #blocked1>) -> !tt.memdesc<32x64xf32, #shared>
      %73 = tt.trans %72 {order=array<i32: 1,0>} : !tt.memdesc<32x64xf32, #shared> -> !tt.memdesc<64x32xf32, #shared1>
      %74 = triton_gpu.local_load %73 : !tt.memdesc<64x32xf32, #shared1> -> tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %75 = tt.dot %71, %74, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x64xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      %76 = tt.load %61 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked1>
      %77 = triton_gpu.convert_layout %75 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %78 = triton_gpu.convert_layout %76 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %79 = tt.dot %77, %78, %arg7 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      scf.yield %79 : tensor<64x32xf32, #mma>
    }
    %64 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %65 = tt.broadcast %53 : tensor<1x32xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %66 = arith.addi %64, %65 : tensor<64x32xi64, #blocked>
    %67 = tt.splat %arg4 : !tt.ptr<f32, 1> -> tensor<64x32x!tt.ptr<f32, 1>, #blocked>
    %68 = tt.addptr %67, %66 : tensor<64x32x!tt.ptr<f32, 1>, #blocked>, tensor<64x32xi64, #blocked>
    %69 = triton_gpu.convert_layout %63 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #blocked>
    tt.store %68, %69 {cache = 1 : i32, evict = 1 : i32} : tensor<64x32xf32, #blocked>
    tt.return
  }
} // end module

// -----
// CHECK: #[[$SHARED_LAYOUT:shared.*]] = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
// CHECK-LABEL: tt.func @indirect_load_shared_layout
// CHECK: scf.for
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %{{.*}}, {{.*}}
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]]
// CHECK: %[[IND_BUFFER_0:.*]] = triton_gpu.memdesc_subview {{.*}} : !tt.memdesc<1x16xi64, #[[$SHARED_LAYOUT]], mutable> -> !tt.memdesc<16xi64, #[[$SHARED_LAYOUT]], mutable>
// CHECK: %[[IND_BUFFER_1:.*]] = triton_gpu.local_load %[[IND_BUFFER_0]]
// CHECK: %[[IND_BUFFER_2:.*]] = tt.expand_dims %[[IND_BUFFER_1]] {axis = 1 : i32}
// CHECK: %[[IND_BUFFER_3:.*]] = tt.broadcast %[[IND_BUFFER_2]]
// CHECK: %[[IND_BUFFER_4:.*]] = arith.muli {{.*}}, %[[IND_BUFFER_3]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[IND_BUFFER_4]]
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_0]]
// CHECK: triton_gpu.async_wait {{.*}} {num = 1 : i32}
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
module attributes {"triton_gpu.compute-capability" = 86 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func @indirect_load_shared_layout(%77: tensor<16x16xi64, #BL> {tt.divisibility=16: i32, tt.constancy=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: tensor<16x!tt.ptr<i64>, #BLs1>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_splat = tt.splat %c1_i32 : i32 -> tensor<16xi32, #BLs1>
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>) {
    %82 = tt.load %arg20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
    %83 = tt.load %arg21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16xi64, #BLs1>
    %84 = tt.expand_dims %83 {axis=1: i32}: tensor<16xi64, #BLs1> -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : tensor<16x1xi64, #BL> -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}
}


// -----

// CHECK-LABEL: @kernel_yield_constant
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.memdesc_subview
// CHECK: scf.for
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.memdesc_subview
// CHECK: tt.return
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"triton_gpu.compute-capability" = 86 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_yield_constant(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_1 = arith.constant dense<2.000000e+00> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %0 = tt.get_program_id x : i32
    %7 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %12 = arith.addi %arg4, %c31_i32 : i32
    %13 = arith.divsi %12, %c32_i32 : i32
    %14 = tt.expand_dims %7 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %22 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
    %34 = tt.splat %arg1 : !tt.ptr<f32, 1> -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
    %42 = scf.for %arg7 = %c0_i32 to %13 step %c1_i32 iter_args(%arg8 = %cst) -> (tensor<32x32xf32, #mma>)  : i32 {
      %43 = arith.muli %arg7, %c32_i32 : i32
      %44 = arith.muli %43, %arg5 : i32
      %45 = tt.splat %44 : i32 -> tensor<32x32xi32, #blocked>
      %46 = tt.addptr %22, %45 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
      %47 = arith.subi %arg4, %43 : i32
      %48 = tt.splat %47 : i32 -> tensor<32x1xi32, #blocked>
      %49 = arith.cmpi slt, %14, %48 : tensor<32x1xi32, #blocked>
      %50 = tt.broadcast %49 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %51 = tt.load %46, %50, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
      %52 = triton_gpu.convert_layout %51 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %53 = tt.dot %cst_1, %52, %arg8 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %54 = triton_gpu.convert_layout %53 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %34, %54 {cache = 1 : i32, evict = 1 : i32} : tensor<32x32xf32, #blocked>
      scf.yield %cst1 : tensor<32x32xf32, #mma>
    }
    tt.return
  }
}


// -----

// CHECK-LABEL: @add_kernel
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:   %[[ABUFFER:.*]] = triton_gpu.local_alloc
// CHECK:   %[[BBUFFER:.*]] = triton_gpu.local_alloc
// CHECK:   %[[A0BUFFER:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local {{.*}}, %[[A0BUFFER]]
// CHECK:   %[[B0BUFFER:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local {{.*}}, %[[B0BUFFER]]
// CHECK:   %[[A1BUFFER:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local {{.*}}, %[[A1BUFFER]]
// CHECK:   %[[B1BUFFER:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local {{.*}}, %[[B1BUFFER]]
// CHECK:   scf.for
#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1016800_i32 = arith.constant 1016800 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1016800_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<1024x!tt.ptr<f32, 1>, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<f32, 1> -> tensor<1024x!tt.ptr<f32, 1>, #blocked>
    %6 = tt.splat %arg2 : !tt.ptr<f32, 1> -> tensor<1024x!tt.ptr<f32, 1>, #blocked>
    scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32  : i32 {
      %7 = arith.addi %1, %arg4 : i32
      %8 = tt.splat %7 : i32 -> tensor<1024xi32, #blocked>
      %9 = arith.addi %8, %2 : tensor<1024xi32, #blocked>
      %10 = arith.cmpi slt, %9, %3 : tensor<1024xi32, #blocked>
      %11 = tt.addptr %4, %9 : tensor<1024x!tt.ptr<f32, 1>, #blocked>, tensor<1024xi32, #blocked>
      %12 = tt.load %11, %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked>
      %13 = tt.addptr %5, %9 : tensor<1024x!tt.ptr<f32, 1>, #blocked>, tensor<1024xi32, #blocked>
      %14 = tt.load %13, %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked>
      %15 = arith.addf %12, %14 : tensor<1024xf32, #blocked>
      %16 = tt.addptr %6, %9 : tensor<1024x!tt.ptr<f32, 1>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %16, %15, %10 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32, #blocked>
    }{tt.num_stages = 3 : i32}
    tt.return
  }
}


// -----

// CHECK-LABEL: @nested_loops
// CHECK: tt.addptr %{{.*}}, {{.*}}
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %{{.*}}, {{.*}}
// CHECK: %[[BUFFER_1:.*]] = triton_gpu.local_alloc
// CHECK: %[[SUBVIEW_1:.*]] = triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK: %[[ASYNC_COPY_1:.*]] = triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]], %[[SUBVIEW_1]]
// CHECK: triton_gpu.async_commit_group %[[ASYNC_COPY_1]]
// CHECK: %[[SUBVIEW_2:.*]] = triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK: %[[ASYNC_COPY_2:.*]] = triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]], %[[SUBVIEW_2]]
// CHECK: triton_gpu.async_commit_group %[[ASYNC_COPY_2]]
// CHECK: scf.for
// CHECK:   %[[LOAD_1:.*]] = tt.load %[[NEXT_BUFFER_1]]
// CHECK:   %[[BUFFER_2:.*]] = triton_gpu.local_alloc %[[LOAD_1]]
// CHECK:   %[[TRANS:.*]] = tt.trans %[[BUFFER_2]]
// CHECK:   %[[LOCAL_LOAD_1:.*]] = triton_gpu.local_load %[[TRANS]]
// CHECK:   triton_gpu.async_wait
// CHECK:   triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK:   scf.for
// CHECK:     %[[LOCAL_LOAD_2:.*]] = triton_gpu.local_load
// CHECK:     %[[DOT:.*]] = tt.dot %[[LOCAL_LOAD_2]], %[[LOCAL_LOAD_1]]
// CHECK:     %[[CONVERT_LAYOUT_3:.*]] = triton_gpu.convert_layout %[[DOT]]
// CHECK:     %[[SUBVIEW_4:.*]] = triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK:     %[[ASYNC_COPY_3:.*]] = triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]], %[[SUBVIEW_4]]
// CHECK:     triton_gpu.async_commit_group %[[ASYNC_COPY_3]]
// CHECK:     triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK:   %[[SUBVIEW_6:.*]] = triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK:   %[[ASYNC_COPY_4:.*]] = triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]], %[[SUBVIEW_6]] mask
// CHECK:   %[[COMMIT_1:.*]] = triton_gpu.async_commit_group %[[ASYNC_COPY_4]]
// CHECK:   %[[SUBVIEW_7:.*]] = triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK:   %[[ASYNC_COPY_5:.*]] = triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]], %[[SUBVIEW_7]] mask
// CHECK:   %[[COMMIT_2:.*]] = triton_gpu.async_commit_group %[[ASYNC_COPY_5]]
// CHECK:   scf.yield %[[COMMIT_1]], %[[COMMIT_2]]
// CHECK: triton_gpu.local_dealloc %[[BUFFER_1]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 2], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @nested_loops(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %2 = arith.muli %1, %cst_0 : tensor<16x1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<16x1x!tt.ptr<f32, 1>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<16x1x!tt.ptr<f32, 1>, #blocked>, tensor<16x1xi32, #blocked>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %7 = tt.broadcast %4 : tensor<16x1x!tt.ptr<f32, 1>, #blocked> -> tensor<16x16x!tt.ptr<f32, 1>, #blocked>
    %8 = tt.broadcast %6 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %9 = tt.addptr %7, %8 : tensor<16x16x!tt.ptr<f32, 1>, #blocked>, tensor<16x16xi32, #blocked>
    scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %10 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf32, #blocked>
      %11 = triton_gpu.local_alloc %10 : (tensor<16x16xf32, #blocked>) -> !tt.memdesc<16x16xf32, #shared>
      %12 = tt.trans %11 {order = array<i32: 1, 0>} : !tt.memdesc<16x16xf32, #shared> -> !tt.memdesc<16x16xf32, #shared1>
      %13 = triton_gpu.local_load %12 : !tt.memdesc<16x16xf32, #shared1> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %14 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf32, #blocked>
        %15 = triton_gpu.convert_layout %14 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %16 = tt.dot %15, %13, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x16xf32, #mma>
        %17 = triton_gpu.convert_layout %16 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
        tt.store %9, %17 {cache = 1 : i32, evict = 1 : i32} : tensor<16x16xf32, #blocked>
      }
    }
    tt.return
  }
}

// -----

  // CHECK-LABEL: @int4_matmul_ampere
#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [16, 1, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [1, 8, 1], order = [2, 0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [16, 2, 1], threadsPerWarp = [4, 1, 8], warpsPerCTA = [1, 1, 8], order = [1, 0, 2]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [32, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @int4_matmul_ampere(
    %arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i8, 1> {tt.divisibility = 16 : i32}
  ) -> tensor<16x256xf32, #mma> attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<64x256xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<16x128xi32, #blocked1>
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x128xf16, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c15_i32 = arith.constant 15 : i32
    %cst_2 = arith.constant dense<4> : tensor<64x256xi8, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #mma>

    %35 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %36 = tt.expand_dims %35 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %38 = tt.broadcast %36 : tensor<1x128xi32, #blocked1> -> tensor<16x128xi32, #blocked1>
    %40 = tt.splat %arg0 : !tt.ptr<f16, 1> -> tensor<16x128x!tt.ptr<f16, 1>, #blocked1>
    %41 = tt.addptr %40, %38 : tensor<16x128x!tt.ptr<f16, 1>, #blocked1>, tensor<16x128xi32, #blocked1>

    %42 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %43 = tt.expand_dims %42 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %47 = tt.broadcast %43 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %50 = tt.splat %arg1 : !tt.ptr<i8, 1> -> tensor<64x256x!tt.ptr<i8, 1>, #blocked>
    %51 = tt.addptr %50, %47 : tensor<64x256x!tt.ptr<i8, 1>, #blocked>, tensor<64x256xi32, #blocked>

    // Check that both loads in the loop are pipelined.
    // TODO(jlebar): https://github.com/openai/triton/pull/3472 disables the
    // relevant optimization.  Once we've reenabled it, we can uncomment this test.
    // CHECK: scf.for
    // COM: CHECK-NOT: tt.load
    // CHECK: triton_gpu.async_copy_global_to_local
    // COM: CHECK-NOT: tt.load
    // COM: CHECK: triton_gpu.async_copy_global_to_local
    // COM: CHECK-NOT: tt.load
    // CHECK: scf.yield
    %54:3 = scf.for %arg9 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg10 = %cst_3, %arg11 = %41, %arg12 = %51) -> (tensor<16x256xf32, #mma>, tensor<16x128x!tt.ptr<f16, 1>, #blocked1>, tensor<64x256x!tt.ptr<i8, 1>, #blocked>)  : i32 {
      %78 = tt.load %arg11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x128xf16, #blocked1>
      %79 = tt.load %arg12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x256xi8, #blocked>
      %80 = arith.shli %79, %cst_2 : tensor<64x256xi8, #blocked>
      %81 = arith.shrsi %80, %cst_2 : tensor<64x256xi8, #blocked>
      %82 = arith.shrsi %79, %cst_2 : tensor<64x256xi8, #blocked>
      %83 = arith.sitofp %81 : tensor<64x256xi8, #blocked> to tensor<64x256xf16, #blocked>
      %84 = arith.sitofp %82 : tensor<64x256xi8, #blocked> to tensor<64x256xf16, #blocked>
      %85 = tt.join %83, %84 : tensor<64x256xf16, #blocked> -> tensor<64x256x2xf16, #blocked3>
      %86 = tt.trans %85 {order = array<i32: 0, 2, 1>} : tensor<64x256x2xf16, #blocked3> -> tensor<64x2x256xf16, #blocked4>
      %87 = tt.reshape %86 {allow_reorder = false} : tensor<64x2x256xf16, #blocked4> -> tensor<128x256xf16, #blocked5>
      %88 = triton_gpu.convert_layout %78 : tensor<16x128xf16, #blocked1> -> tensor<16x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %89 = triton_gpu.convert_layout %87 : tensor<128x256xf16, #blocked5> -> tensor<128x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %90 = tt.dot %88, %89, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x256xf32, #mma>
      %91 = tt.addptr %arg11, %cst_0 : tensor<16x128x!tt.ptr<f16, 1>, #blocked1>, tensor<16x128xi32, #blocked1>
      %92 = tt.addptr %arg12, %cst : tensor<64x256x!tt.ptr<i8, 1>, #blocked>, tensor<64x256xi32, #blocked>
      scf.yield %90, %91, %92 : tensor<16x256xf32, #mma>, tensor<16x128x!tt.ptr<f16, 1>, #blocked1>, tensor<64x256x!tt.ptr<i8, 1>, #blocked>
    }
    tt.return %54#0 : tensor<16x256xf32, #mma>
  }
}


// -----

// This test triggered some failure in the verifier, so we only
// included a simple check for the kernel name.
// CHECK-LABEL: @load_convert_layout
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 8]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @load_convert_layout(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64, 1> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant dense<2> : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %0 = tt.load %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %4 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %5 = tt.broadcast %3 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<32x32xi32, #blocked>
    %8 = tt.splat %arg2 : !tt.ptr<i64, 1> -> tensor<32x!tt.ptr<i64, 1>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %9 = tt.addptr %8, %1 : tensor<32x!tt.ptr<i64, 1>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %10 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
    %11 = tt.addptr %10, %7 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
    %12 = tt.load %11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
    %13 = tt.splat %arg4 : !tt.ptr<f32, 1> -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
    %14 = tt.addptr %13, %7 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
    %15 = arith.cmpi slt, %1, %cst_0 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %16 = tt.splat %arg3 : !tt.ptr<f32, 1> -> tensor<32x!tt.ptr<f32, 1>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %17 = scf.for %arg5 = %c0_i64 to %0 step %c32_i64 iter_args(%arg6 = %12) -> (tensor<32x32xf32, #blocked>)  : i64 {
      %18 = tt.load %9, %15 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %19 = tt.addptr %16, %18 : tensor<32x!tt.ptr<f32, 1>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %20 = tt.load %19 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %21 = tt.expand_dims %20 {axis = 0 : i32} : tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xf32, #blocked>
      %22 = tt.broadcast %21 : tensor<1x32xf32, #blocked> -> tensor<32x32xf32, #blocked>
      %23 = arith.addf %arg6, %22 : tensor<32x32xf32, #blocked>
      %24 = triton_gpu.convert_layout %23 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %25 = triton_gpu.convert_layout %23 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %26 = tt.dot %24, %25, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %27 = triton_gpu.convert_layout %26 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %14, %27 {cache = 1 : i32, evict = 1 : i32} : tensor<32x32xf32, #blocked>
      scf.yield %23 : tensor<32x32xf32, #blocked>
    }
    tt.return
  }
}
