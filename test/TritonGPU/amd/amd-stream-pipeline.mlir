// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline=num_stages=2 | FileCheck %s

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

// CHECK-LABEL:  tt.func @matmul_loop
// CHECK:  %[[LOCAL_ALLOC_10:.*]] = triton_gpu.local_alloc
// CHECK:  %[[LOCAL_ALLOC_11:.*]] = triton_gpu.local_alloc
// CHECK:  %[[CMPI_12:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}}
// CHECK:  %[[SPLAT_13:.*]] = tt.splat %[[CMPI_12]]
// CHECK:  %[[LOAD_14:.*]] = tt.load %{{.*}}, %[[SPLAT_13]]
// CHECK:  %[[SPLAT_15:.*]] = tt.splat %[[CMPI_12]]
// CHECK:  %[[LOAD_16:.*]] = tt.load %{{.*}}, %[[SPLAT_15]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_17:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_10]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_14]], %[[MEMDESC_SUBVIEW_17]]
// CHECK:  %[[MEMDESC_SUBVIEW_18:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_11]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_16]], %[[MEMDESC_SUBVIEW_18]]
// CHECK:  %{{.*}}:7 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}-1_i32, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %[[MEMDESC_SUBVIEW_17]], %[[ARG12:.*]] = %[[MEMDESC_SUBVIEW_18]])

// CHECK:  %[[SUBI_20:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_21:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_20]]
// CHECK:  %[[ADDI_22:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_23:.*]] = arith.cmpi slt, %[[ADDI_22]], %{{.*}}
// CHECK:  %[[SELECT_24:.*]] = arith.select %[[CMPI_23]], %[[ADDI_22]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_25:.*]] = triton_gpu.local_load %[[ARG11]]
// CHECK:  %[[CONVERT_LAYOUT_26:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_25]]
// CHECK:  %[[LOCAL_LOAD_27:.*]] = triton_gpu.local_load %[[ARG12]]
// CHECK:  %[[CONVERT_LAYOUT_28:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_27]]
// CHECK:  %[[MULF_29:.*]] = arith.mulf %[[CONVERT_LAYOUT_28]], %{{.*}}
// CHECK:  %[[DOT_30:.*]] = tt.dot %[[CONVERT_LAYOUT_26]], %[[MULF_29]], %[[ARG8]]
// CHECK:  %[[ADDPTR_31:.*]] = tt.addptr %[[ARG6]], %{{.*}}
// CHECK:  %[[ADDPTR_32:.*]] = tt.addptr %[[ARG7]], %{{.*}}
// CHECK:  %[[SPLAT_33:.*]] = tt.splat %[[CMPI_21]]
// CHECK:  %[[LOAD_34:.*]] = tt.load %[[ADDPTR_31]], %[[SPLAT_33]]
// CHECK:  %[[SPLAT_35:.*]] = tt.splat %[[CMPI_21]]
// CHECK:  %[[LOAD_36:.*]] = tt.load %[[ADDPTR_32]], %[[SPLAT_35]], %{{.*}}
// CHECK:  %[[ADDI_37:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_38:.*]] = arith.cmpi slt, %[[ADDI_37]], %{{.*}}
// CHECK:  %[[SELECT_39:.*]] = arith.select %[[CMPI_38]], %[[ADDI_37]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_40:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_10]][%[[SELECT_39]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_34]], %[[MEMDESC_SUBVIEW_40]]
// CHECK:  %[[MEMDESC_SUBVIEW_41:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_11]][%[[SELECT_39]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_36]], %[[MEMDESC_SUBVIEW_41]]
// CHECK:  scf.yield %[[ADDPTR_31]], %[[ADDPTR_32]], %[[DOT_30]], %[[SELECT_24]], %[[SELECT_39]], %[[MEMDESC_SUBVIEW_40]], %[[MEMDESC_SUBVIEW_41]]
// CHECK:  }

// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_10]]
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_11]]

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.target" = "cuda:80"} {
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
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    %b__ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %b_ = triton_gpu.convert_layout %b__ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %b = arith.mulf %b_, %b_scale: tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL:  tt.func @matmul_loop_nested
// CHECK:  %[[LOCAL_ALLOC_11:.*]] = triton_gpu.local_alloc
// CHECK:  %[[LOCAL_ALLOC_12:.*]] = triton_gpu.local_alloc
// CHECK:  %[[CMPI_13:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}}
// CHECK:  %[[SPLAT_14:.*]] = tt.splat %[[CMPI_13]]
// CHECK:  %[[LOAD_15:.*]] = tt.load %{{.*}}, %[[SPLAT_14]], %{{.*}}
// CHECK:  %[[SPLAT_16:.*]] = tt.splat %[[CMPI_13]]
// CHECK:  %[[LOAD_17:.*]] = tt.load %{{.*}}, %[[SPLAT_16]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_18:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_11]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_15]], %[[MEMDESC_SUBVIEW_18]]
// CHECK:  %[[MEMDESC_SUBVIEW_19:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_12]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_17]], %[[MEMDESC_SUBVIEW_19]]
// CHECK:  %{{.*}}:7 = scf.for %[[ARG7:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}-1_i32, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %[[MEMDESC_SUBVIEW_18]], %[[ARG14:.*]] = %[[MEMDESC_SUBVIEW_19]])

// CHECK:  %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_22:.*]] = arith.cmpi slt, %[[ARG7]], %[[SUBI_21]]
// CHECK:  %[[ADDI_23:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_24:.*]] = arith.cmpi slt, %[[ADDI_23]], %{{.*}}
// CHECK:  %[[SELECT_25:.*]] = arith.select %[[CMPI_24]], %[[ADDI_23]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_26:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[CONVERT_LAYOUT_27:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_26]]
// CHECK:  %[[LOCAL_LOAD_28:.*]] = triton_gpu.local_load %[[ARG14]]
// CHECK:  %[[CONVERT_LAYOUT_29:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_28]]
// CHECK:  %[[DOT_30:.*]] = tt.dot %[[CONVERT_LAYOUT_27]], %[[CONVERT_LAYOUT_29]], %[[ARG10]]
// CHECK:  %[[ADDPTR_31:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[ADDPTR_32:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[SPLAT_33:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[LOAD_34:.*]] = tt.load %[[ADDPTR_31]], %[[SPLAT_33]], %{{.*}}
// CHECK:  %[[SPLAT_35:.*]] = tt.splat %[[CMPI_22]]
// CHECK:  %[[LOAD_36:.*]] = tt.load %[[ADDPTR_32]], %[[SPLAT_35]], %{{.*}}
// CHECK:  %[[ADDI_37:.*]] = arith.addi %[[ARG12]], %{{.*}}
// CHECK:  %[[CMPI_38:.*]] = arith.cmpi slt, %[[ADDI_37]], %{{.*}}
// CHECK:  %[[SELECT_39:.*]] = arith.select %[[CMPI_38]], %[[ADDI_37]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_40:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_11]][%[[SELECT_39]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_34]], %[[MEMDESC_SUBVIEW_40]]
// CHECK:  %[[MEMDESC_SUBVIEW_41:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_12]][%[[SELECT_39]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_36]], %[[MEMDESC_SUBVIEW_41]]
// CHECK:  scf.yield %[[ADDPTR_31]], %[[ADDPTR_32]], %[[DOT_30]], %[[SELECT_25]], %[[SELECT_39]], %[[MEMDESC_SUBVIEW_40]], %[[MEMDESC_SUBVIEW_41]]
// CHECK:  }
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_11]]
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_12]]
// CHECK:  scf.yield %{{.*}}#2
// CHECK:  }
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
      %a_ = tt.load %a_ptr, %a_mask, %a_other : tensor<128x32x!tt.ptr<f16>, #AL>
      %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
      %b_ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
      %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

      %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

      %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
      %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
      scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
    }

    scf.yield %loop2#2 : tensor<128x128xf32, #C>
  }
  tt.return %loop1#0 : tensor<128x128xf32, #C>
}

// CHECK-LABEL:  tt.func @matmul_loop_single_pipeline
// CHECK:  %[[LOAD_10:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_11:.*]] = triton_gpu.convert_layout %[[LOAD_10]]
// CHECK:  %[[LOCAL_ALLOC_12:.*]] = triton_gpu.local_alloc
// CHECK:  %[[CMPI_13:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}}
// CHECK:  %[[SPLAT_14:.*]] = tt.splat %[[CMPI_13]]
// CHECK:  %[[LOAD_15:.*]] = tt.load %{{.*}}, %[[SPLAT_14]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_16:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_12]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_15]], %[[MEMDESC_SUBVIEW_16]]
// CHECK:  %{{.*}}:5 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}-1_i32, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %[[MEMDESC_SUBVIEW_16]])
// CHECK:  %[[SUBI_18:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_19:.*]] = arith.cmpi slt, %[[ARG5]], %[[SUBI_18]]
// CHECK:  %[[ADDI_20:.*]] = arith.addi %[[ARG8]], %{{.*}}
// CHECK:  %[[CMPI_21:.*]] = arith.cmpi slt, %[[ADDI_20]], %{{.*}}
// CHECK:  %[[SELECT_22:.*]] = arith.select %[[CMPI_21]], %[[ADDI_20]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_23:.*]] = triton_gpu.local_load %[[ARG10]]
// CHECK:  %[[CONVERT_LAYOUT_24:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_23]]
// CHECK:  %[[DOT_25:.*]] = tt.dot %[[CONVERT_LAYOUT_11]], %[[CONVERT_LAYOUT_24]], %[[ARG7]]
// CHECK:  %[[ADDPTR_26:.*]] = tt.addptr %[[ARG6]], %{{.*}}
// CHECK:  %[[SPLAT_27:.*]] = tt.splat %[[CMPI_19]]
// CHECK:  %[[LOAD_28:.*]] = tt.load %[[ADDPTR_26]], %[[SPLAT_27]], %{{.*}}
// CHECK:  %[[ADDI_29:.*]] = arith.addi %[[ARG9]], %{{.*}}
// CHECK:  %[[CMPI_30:.*]] = arith.cmpi slt, %[[ADDI_29]], %{{.*}}
// CHECK:  %[[SELECT_31:.*]] = arith.select %[[CMPI_30]], %[[ADDI_29]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_32:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_12]][%[[SELECT_31]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_28]], %[[MEMDESC_SUBVIEW_32]]
// CHECK:  scf.yield %[[ADDPTR_26]], %[[DOT_25]], %[[SELECT_22]], %[[SELECT_31]], %[[MEMDESC_SUBVIEW_32]]
// CHECK:  }
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_12]]
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

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x32x!tt.ptr<f16>, #AL>
  %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %b_ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_b_ptr, %c : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#1 : tensor<128x128xf32, #C>
}

// CHECK-LABEL:  tt.func @indirect_bmm_scalar
// CHECK:  %[[LOCAL_ALLOC_0:.*]] = triton_gpu.local_alloc
// CHECK:  %[[LOCAL_ALLOC_1:.*]] = triton_gpu.local_alloc
// CHECK:  %[[CMPI_2:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}}
// CHECK:  %[[SPLAT_3:.*]] = tt.splat %[[CMPI_2]]
// CHECK:  %[[LOAD_4:.*]] = tt.load %{{.*}}, %[[SPLAT_3]]
// CHECK:  %[[LOAD_5:.*]] = tt.load %{{.*}}, %[[CMPI_2]]
// CHECK:  %[[MULI_6:.*]] = arith.muli %{{.*}}, %[[LOAD_5]]
// CHECK:  %[[SPLAT_7:.*]] = tt.splat %[[MULI_6]]
// CHECK:  %[[ADDPTR_8:.*]] = tt.addptr %{{.*}}, %[[SPLAT_7]]
// CHECK:  %[[SPLAT_9:.*]] = tt.splat %[[CMPI_2]]
// CHECK:  %[[LOAD_10:.*]] = tt.load %[[ADDPTR_8]], %[[SPLAT_9]]
// CHECK:  %[[CMPI_11:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}}
// CHECK:  %[[ADDPTR_12:.*]] = tt.addptr %{{.*}}, %{{.*}}
// CHECK:  %[[ADDPTR_13:.*]] = tt.addptr %{{.*}}, %{{.*}}
// CHECK:  %[[SPLAT_14:.*]] = tt.splat %[[CMPI_11]]
// CHECK:  %[[LOAD_15:.*]] = tt.load %[[ADDPTR_12]], %[[SPLAT_14]]
// CHECK:  %[[LOAD_16:.*]] = tt.load %[[ADDPTR_13]], %[[CMPI_11]]
// CHECK:  %[[MULI_17:.*]] = arith.muli %{{.*}}, %[[LOAD_16]]
// CHECK:  %[[SPLAT_18:.*]] = tt.splat %[[MULI_17]]
// CHECK:  %[[ADDPTR_19:.*]] = tt.addptr %{{.*}}, %[[SPLAT_18]]
// CHECK:  %[[SPLAT_20:.*]] = tt.splat %[[CMPI_11]]
// CHECK:  %[[LOAD_21:.*]] = tt.load %[[ADDPTR_19]], %[[SPLAT_20]]
// CHECK:  %[[MEMDESC_SUBVIEW_22:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_0]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_4]], %[[MEMDESC_SUBVIEW_22]]
// CHECK:  %[[MEMDESC_SUBVIEW_23:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_1]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_10]], %[[MEMDESC_SUBVIEW_23]]
// CHECK:  %{{.*}}:9 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %[[ADDPTR_12]], %[[ARG9:.*]] = %[[ADDPTR_13]], %[[ARG10:.*]] = %{{.*}}-1_i32, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %[[MEMDESC_SUBVIEW_22]], %[[ARG13:.*]] = %[[MEMDESC_SUBVIEW_23]], %[[ARG14:.*]] = %[[LOAD_15]], %[[ARG15:.*]] = %[[LOAD_21]])

// CHECK:  %[[SUBI_25:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_26:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_25]]
// CHECK:  %[[ADDI_27:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_28:.*]] = arith.cmpi slt, %[[ADDI_27]], %{{.*}}
// CHECK:  %[[SELECT_29:.*]] = arith.select %[[CMPI_28]], %[[ADDI_27]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_30:.*]] = triton_gpu.local_load %[[ARG12]]
// CHECK:  %[[LOCAL_LOAD_31:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[CONVERT_LAYOUT_32:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_30]]
// CHECK:  %[[CONVERT_LAYOUT_33:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_31]]
// CHECK:  %[[DOT_34:.*]] = tt.dot %[[CONVERT_LAYOUT_32]], %[[CONVERT_LAYOUT_33]], %[[ARG7]]
// CHECK:  %[[ADDPTR_35:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[ADDPTR_36:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[SPLAT_37:.*]] = tt.splat %[[CMPI_26]]
// CHECK:  %[[LOAD_38:.*]] = tt.load %[[ADDPTR_35]], %[[SPLAT_37]]
// CHECK:  %[[LOAD_39:.*]] = tt.load %[[ADDPTR_36]], %[[CMPI_26]]
// CHECK:  %[[MULI_40:.*]] = arith.muli %{{.*}}, %[[LOAD_39]]
// CHECK:  %[[SPLAT_41:.*]] = tt.splat %[[MULI_40]]
// CHECK:  %[[ADDPTR_42:.*]] = tt.addptr %{{.*}}, %[[SPLAT_41]]
// CHECK:  %[[SPLAT_43:.*]] = tt.splat %[[CMPI_26]]
// CHECK:  %[[LOAD_44:.*]] = tt.load %[[ADDPTR_42]], %[[SPLAT_43]]
// CHECK:  %[[ADDI_45:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_46:.*]] = arith.cmpi slt, %[[ADDI_45]], %{{.*}}
// CHECK:  %[[SELECT_47:.*]] = arith.select %[[CMPI_46]], %[[ADDI_45]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_48:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_0]][%[[SELECT_47]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[ARG14]], %[[MEMDESC_SUBVIEW_48]]
// CHECK:  %[[MEMDESC_SUBVIEW_49:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_1]][%[[SELECT_47]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[ARG15]], %[[MEMDESC_SUBVIEW_49]]
// CHECK:  scf.yield %[[DOT_34]], %[[ADDPTR_35]], %[[ADDPTR_36]], %[[SELECT_29]], %[[SELECT_47]], %[[MEMDESC_SUBVIEW_48]], %[[MEMDESC_SUBVIEW_49]], %[[LOAD_38]], %[[LOAD_44]]
// CHECK:  }

// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_0]]
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_1]]

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
    %82 = tt.load %arg20 : tensor<16x16x!tt.ptr<f16>, #AL>
    %83 = tt.load %arg21 : !tt.ptr<i64>
    %84 = arith.muli %77, %83 : i64
    %85 = tt.splat %84 : i64 -> tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 : tensor<16x16x!tt.ptr<f16>, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32 : !tt.ptr<i64>, i32
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>
  } {tt.num_stages = 3 : i32}
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK-LABEL:  tt.func @indirect_bmm_scalar_dist_one
// CHECK:  %[[LOAD_0:.*]] = tt.load %{{.*}}
// CHECK:  %[[ADDPTR_1:.*]] = tt.addptr %{{.*}}, %{{.*}}
// CHECK:  %[[LOCAL_ALLOC_2:.*]] = triton_gpu.local_alloc
// CHECK:  %[[LOCAL_ALLOC_3:.*]] = triton_gpu.local_alloc
// CHECK:  %[[CMPI_4:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}}
// CHECK:  %[[SPLAT_5:.*]] = tt.splat %[[CMPI_4]]
// CHECK:  %[[LOAD_6:.*]] = tt.load %{{.*}}, %[[SPLAT_5]]
// CHECK:  %[[LOAD_7:.*]] = tt.load %[[ADDPTR_1]], %[[CMPI_4]]
// CHECK:  %[[MULI_8:.*]] = arith.muli %{{.*}}, %[[LOAD_0]]
// CHECK:  %[[SPLAT_9:.*]] = tt.splat %[[MULI_8]]
// CHECK:  %[[ADDPTR_10:.*]] = tt.addptr %{{.*}}, %[[SPLAT_9]]
// CHECK:  %[[SPLAT_11:.*]] = tt.splat %[[CMPI_4]]
// CHECK:  %[[LOAD_12:.*]] = tt.load %[[ADDPTR_10]], %[[SPLAT_11]]
// CHECK:  %[[ADDPTR_13:.*]] = tt.addptr %[[ADDPTR_1]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_14:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_2]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_6]], %[[MEMDESC_SUBVIEW_14]]
// CHECK:  %[[MEMDESC_SUBVIEW_15:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_3]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_12]], %[[MEMDESC_SUBVIEW_15]]
// CHECK:  %{{.*}}:8 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %[[ADDPTR_13]], %[[ARG10:.*]] = %[[LOAD_7]], %[[ARG11:.*]] = %{{.*}}-1_i32, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %[[MEMDESC_SUBVIEW_14]], %[[ARG14:.*]] = %[[MEMDESC_SUBVIEW_15]])

// CHECK:  %[[SUBI_17:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_18:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_17]]
// CHECK:  %[[ADDI_19:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_20:.*]] = arith.cmpi slt, %[[ADDI_19]], %{{.*}}
// CHECK:  %[[SELECT_21:.*]] = arith.select %[[CMPI_20]], %[[ADDI_19]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_22:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[LOCAL_LOAD_23:.*]] = triton_gpu.local_load %[[ARG14]]
// CHECK:  %[[CONVERT_LAYOUT_24:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_22]]
// CHECK:  %[[CONVERT_LAYOUT_25:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_23]]
// CHECK:  %[[DOT_26:.*]] = tt.dot %[[CONVERT_LAYOUT_24]], %[[CONVERT_LAYOUT_25]], %[[ARG7]]
// CHECK:  %[[ADDPTR_27:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[SPLAT_28:.*]] = tt.splat %[[CMPI_18]]
// CHECK:  %[[LOAD_29:.*]] = tt.load %[[ADDPTR_27]], %[[SPLAT_28]]
// CHECK:  %[[LOAD_30:.*]] = tt.load %[[ARG9]], %[[CMPI_18]]
// CHECK:  %[[MULI_31:.*]] = arith.muli %{{.*}}, %[[ARG10]]
// CHECK:  %[[SPLAT_32:.*]] = tt.splat %[[MULI_31]]
// CHECK:  %[[ADDPTR_33:.*]] = tt.addptr %{{.*}}, %[[SPLAT_32]]
// CHECK:  %[[SPLAT_34:.*]] = tt.splat %[[CMPI_18]]
// CHECK:  %[[LOAD_35:.*]] = tt.load %[[ADDPTR_33]], %[[SPLAT_34]]
// CHECK:  %[[ADDPTR_36:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[ADDI_37:.*]] = arith.addi %[[ARG12]], %{{.*}}
// CHECK:  %[[CMPI_38:.*]] = arith.cmpi slt, %[[ADDI_37]], %{{.*}}
// CHECK:  %[[SELECT_39:.*]] = arith.select %[[CMPI_38]], %[[ADDI_37]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_40:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_2]][%[[SELECT_39]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_29]], %[[MEMDESC_SUBVIEW_40]]
// CHECK:  %[[MEMDESC_SUBVIEW_41:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_3]][%[[SELECT_39]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_35]], %[[MEMDESC_SUBVIEW_41]]
// CHECK:  scf.yield %[[DOT_26]], %[[ADDPTR_27]], %[[ADDPTR_36]], %[[LOAD_30]], %[[SELECT_21]], %[[SELECT_39]], %[[MEMDESC_SUBVIEW_40]], %[[MEMDESC_SUBVIEW_41]]
// CHECK:  }
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_2]]
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_3]]

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
  %50 = tt.load %75 : !tt.ptr<i64>
  %51 = tt.addptr %75, %c1_i32 : !tt.ptr<i64>, i32
  %79:4 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %51, %arg22 = %50) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>, i64) {
    %82 = tt.load %arg20 : tensor<16x16x!tt.ptr<f16>, #AL>
    %83 = tt.load %arg21 : !tt.ptr<i64>
    %84 = arith.muli %77, %arg22 : i64
    %85 = tt.splat %84 : i64 -> tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 : tensor<16x16x!tt.ptr<f16>, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32 : !tt.ptr<i64>, i32
    scf.yield %90, %91, %92, %83 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>, i64
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK-LABEL:  tt.func @indirect_bmm_vector
// CHECK:  %[[LOCAL_ALLOC_0:.*]] = triton_gpu.local_alloc
// CHECK:  %[[LOCAL_ALLOC_1:.*]] = triton_gpu.local_alloc
// CHECK:  %[[CMPI_2:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}}
// CHECK:  %[[SPLAT_3:.*]] = tt.splat %[[CMPI_2]]
// CHECK:  %[[LOAD_4:.*]] = tt.load %{{.*}}, %[[SPLAT_3]]
// CHECK:  %[[CMPI_5:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}}
// CHECK:  %[[ADDPTR_6:.*]] = tt.addptr %{{.*}}, %{{.*}}
// CHECK:  %[[SPLAT_7:.*]] = tt.splat %[[CMPI_2]]
// CHECK:  %[[LOAD_8:.*]] = tt.load %{{.*}}, %[[SPLAT_7]]
// CHECK:  %[[EXPAND_DIMS_9:.*]] = tt.expand_dims %[[LOAD_4]] {axis = 1 : i32}
// CHECK:  %[[BROADCAST_10:.*]] = tt.broadcast %[[EXPAND_DIMS_9]]
// CHECK:  %[[MULI_11:.*]] = arith.muli %{{.*}}, %[[BROADCAST_10]]
// CHECK:  %[[ADDPTR_12:.*]] = tt.addptr %{{.*}}, %[[MULI_11]]
// CHECK:  %[[SPLAT_13:.*]] = tt.splat %[[CMPI_2]]
// CHECK:  %[[LOAD_14:.*]] = tt.load %[[ADDPTR_12]], %[[SPLAT_13]]
// CHECK:  %[[SPLAT_15:.*]] = tt.splat %[[CMPI_5]]
// CHECK:  %[[LOAD_16:.*]] = tt.load %[[ADDPTR_6]], %[[SPLAT_15]]
// CHECK:  %[[MEMDESC_SUBVIEW_17:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_0]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_8]], %[[MEMDESC_SUBVIEW_17]]
// CHECK:  %[[MEMDESC_SUBVIEW_18:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_1]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_14]], %[[MEMDESC_SUBVIEW_18]]
// CHECK:  %{{.*}}:8 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %[[ADDPTR_6]], %[[ARG10:.*]] = %{{.*}}-1_i32, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %[[MEMDESC_SUBVIEW_17]], %[[ARG13:.*]] = %[[MEMDESC_SUBVIEW_18]], %[[ARG14:.*]] = %[[LOAD_16]])

// CHECK:  %[[SUBI_20:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_21:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_20]]
// CHECK:  %[[SUBI_22:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_23:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_22]]
// CHECK:  %[[ADDI_24:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_25:.*]] = arith.cmpi slt, %[[ADDI_24]], %{{.*}}
// CHECK:  %[[SELECT_26:.*]] = arith.select %[[CMPI_25]], %[[ADDI_24]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_27:.*]] = triton_gpu.local_load %[[ARG12]]
// CHECK:  %[[LOCAL_LOAD_28:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[CONVERT_LAYOUT_29:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_27]]
// CHECK:  %[[CONVERT_LAYOUT_30:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_28]]
// CHECK:  %[[DOT_31:.*]] = tt.dot %[[CONVERT_LAYOUT_29]], %[[CONVERT_LAYOUT_30]], %[[ARG7]]
// CHECK:  %[[ADDPTR_32:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[ADDPTR_33:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[SPLAT_34:.*]] = tt.splat %[[CMPI_23]]
// CHECK:  %[[LOAD_35:.*]] = tt.load %[[ADDPTR_32]], %[[SPLAT_34]]
// CHECK:  %[[EXPAND_DIMS_36:.*]] = tt.expand_dims %[[ARG14]] {axis = 1 : i32}
// CHECK:  %[[BROADCAST_37:.*]] = tt.broadcast %[[EXPAND_DIMS_36]]
// CHECK:  %[[MULI_38:.*]] = arith.muli %{{.*}}, %[[BROADCAST_37]]
// CHECK:  %[[ADDPTR_39:.*]] = tt.addptr %{{.*}}, %[[MULI_38]]
// CHECK:  %[[SPLAT_40:.*]] = tt.splat %[[CMPI_23]]
// CHECK:  %[[LOAD_41:.*]] = tt.load %[[ADDPTR_39]], %[[SPLAT_40]]
// CHECK:  %[[SPLAT_42:.*]] = tt.splat %[[CMPI_21]]
// CHECK:  %[[LOAD_43:.*]] = tt.load %[[ADDPTR_33]], %[[SPLAT_42]]
// CHECK:  %[[ADDI_44:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_45:.*]] = arith.cmpi slt, %[[ADDI_44]], %{{.*}}
// CHECK:  %[[SELECT_46:.*]] = arith.select %[[CMPI_45]], %[[ADDI_44]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_47:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_0]][%[[SELECT_46]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_35]], %[[MEMDESC_SUBVIEW_47]]
// CHECK:  %[[MEMDESC_SUBVIEW_48:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_1]][%[[SELECT_46]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_41]], %[[MEMDESC_SUBVIEW_48]]
// CHECK:  scf.yield %[[DOT_31]], %[[ADDPTR_32]], %[[ADDPTR_33]], %[[SELECT_26]], %[[SELECT_46]], %[[MEMDESC_SUBVIEW_47]], %[[MEMDESC_SUBVIEW_48]], %[[LOAD_43]]
// CHECK:  }
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_0]]
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_1]]

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
    %82 = tt.load %arg20 : tensor<16x16x!tt.ptr<f16>, #AL>
    %83 = tt.load %arg21 : tensor<16x!tt.ptr<i64>, #BLs1>
    %84 = tt.expand_dims %83 {axis=1: i32}: tensor<16xi64, #BLs1> -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : tensor<16x1xi64, #BL> -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 : tensor<16x16x!tt.ptr<f16>, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>
  } {tt.num_stages = 3 : i32}
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
    %112 = tt.load %arg11, %111, %cst_0 : tensor<32x32x!tt.ptr<f32>, #AL>
    %113 = tt.splat %108 : i32 -> tensor<32x1xi32, #AL>
    %114 = arith.cmpi "slt", %66, %113 : tensor<32x1xi32, #AL>
    %115 = tt.broadcast %114 : tensor<32x1xi1, #AL> -> tensor<32x32xi1, #AL>
    %116 = tt.load %arg12, %115, %cst_0 : tensor<32x32x!tt.ptr<f32>, #AL>
    %117 = triton_gpu.convert_layout %112 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>>
    %118 = triton_gpu.convert_layout %116 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>>
    %119 = tt.dot %117, %118, %arg10 : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>> -> tensor<32x32xf32, #C>
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
// CHECK-NOT: triton_gpu.local_load
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
    %146 = tt.load %arg11, %145, %cst_1 : tensor<32x32x!tt.ptr<f32>, #AL>
    %147 = tt.splat %142 : i32 -> tensor<32x1xi32, #AL>
    %148 = arith.cmpi "slt", %88, %147 : tensor<32x1xi32, #AL>
    %149 = tt.broadcast %148 : tensor<32x1xi1, #AL> -> tensor<32x32xi1, #AL>
    %150 = tt.load %arg12, %149, %cst_1 : tensor<32x32x!tt.ptr<f32>, #AL>
    %151 = triton_gpu.convert_layout %146 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>>
    %152 = triton_gpu.convert_layout %150 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>>
    %153 = tt.dot %151, %152, %arg10 : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>> -> tensor<32x32xf32, #C>
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
    %193 = tt.load %191, %192 : tensor<128x32x!tt.ptr<f16>, #AL>
    %194 = tt.splat %arg22 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %195 = tt.addptr %194, %56 : tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #AL}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %196 = tt.load %195 : tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %197 = tt.addptr %arg22, %c32_i32 : !tt.ptr<i32>, i32
    %198 = tt.broadcast %181 : tensor<32x1xi1, #BL> -> tensor<32x128xi1, #BL>
    %199 = tt.load %arg24, %198, %88 : tensor<32x128x!tt.ptr<f16>, #BL>
    %200 = triton_gpu.convert_layout %193 : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>>
    %201 = triton_gpu.convert_layout %199 : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>>
    %202 = tt.dot %200, %201, %arg23 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>> -> tensor<128x128xf32, #C>
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
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tt.func @load_two_users
  tt.func @load_two_users(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_0 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: triton_gpu.local_store
    // CHECK: scf.for
    // CHECK:   tt.dot
    // CHECK:   tt.dot
    // CHECK:   tt.load
    // CHECK:   triton_gpu.local_store
    // CHECK:   scf.yield

    %17:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = triton_gpu.convert_layout %9 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = triton_gpu.convert_layout %18 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %19, %20, %cst_1 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %23 = triton_gpu.convert_layout %22 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %24 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory>
      %25 = tt.trans %24 {order=array<i32: 1,0>} : !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory>
      %26 = triton_gpu.local_load %25 : !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %23, %26, %arg4 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
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
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tt.func @load_two_users_incompatible_layouts
  tt.func @load_two_users_incompatible_layouts(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_0 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK-NOT: triton_gpu.local_store
    // CHECK: scf.for
    %17:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = triton_gpu.convert_layout %9 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = triton_gpu.convert_layout %18 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %19, %20, %cst_1 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %23 = triton_gpu.convert_layout %22 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %24 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory>
      %25 = tt.trans %24 {order=array<i32: 1,0>} : !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory>
      %26 = triton_gpu.local_load %25 : !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %23, %26, %arg4 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %21, %27 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
    }
    tt.return %17#0, %17#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----

// CHECK-LABEL: tt.func public @nested_loops
// CHECK: scf.for
// CHECK: triton_gpu.local_alloc
// CHECK-NOT: triton_gpu.local_alloc
// CHECK:   scf.for
// CHECK:     scf.yield
// CHECK-DIS:   scf.yield
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
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @nested_loops(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %8 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %9 = arith.muli %arg4, %c32_i32 : i32
      %10 = tt.splat %9 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %11 = tt.splat %9 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %12 = arith.addi %10, %0 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %13 = arith.addi %11, %1 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %14 = tt.expand_dims %12 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %15 = tt.broadcast %14 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
      %16 = tt.addptr %6, %15 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %17 = tt.load %16 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %18 = tt.expand_dims %13 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
      %19 = arith.muli %18, %cst_0 : tensor<32x1xi32, #blocked>
      %20 = tt.addptr %7, %19 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
      %21 = tt.broadcast %20 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
      %22 = tt.addptr %8, %19 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
      %23 = tt.broadcast %22 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
      scf.for %arg5 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
        %24 = arith.muli %arg5, %c32_i32 : i32
        %25 = tt.splat %24 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %26 = arith.addi %25, %0 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %27 = tt.expand_dims %26 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
        %28 = tt.broadcast %27 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
        %29 = tt.addptr %21, %28 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %30 = tt.load %29 : tensor<32x32x!tt.ptr<f32>, #blocked>
        %31 = triton_gpu.convert_layout %30 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %32 = triton_gpu.convert_layout %17 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
        %33 = tt.dot %31, %32, %cst : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
        %34 = tt.addptr %23, %28 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %35 = triton_gpu.convert_layout %33 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
        tt.store %34, %35 : tensor<32x32x!tt.ptr<f32>, #blocked>
      }
    }
    tt.return
  }
}  // end module

// -----

// CHECK-LABEL: tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de
// CHECK-NOT:  triton_gpu.convert_layout {{.*}} : tensor<32x64xf32, #shared> -> tensor<32x64xf32, #shared1>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.get_program_id y : i32
    %3 = tt.load %arg3 : !tt.ptr<i64>
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
    %56 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked>
    %57 = tt.addptr %56, %30 : tensor<64x64x!tt.ptr<f32>, #blocked>, tensor<64x64xi64, #blocked>
    %58 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #blocked1>
    %59 = tt.addptr %58, %42 : tensor<32x64x!tt.ptr<f32>, #blocked1>, tensor<32x64xi64, #blocked1>
    %60 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %61 = tt.addptr %60, %55 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi64, #blocked1>
    %62 = tt.load %57 : tensor<64x64x!tt.ptr<f32>, #blocked>
    %63 = scf.for %arg6 = %c0_i32 to %c64_i32 step %c32_i32 iter_args(%arg7 = %cst) -> (tensor<64x32xf32, #mma>)  : i32 {
      %70 = tt.load %59 : tensor<32x64x!tt.ptr<f32>, #blocked1>
      %71 = triton_gpu.convert_layout %62 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %72 = triton_gpu.local_alloc %70 : (tensor<32x64xf32, #blocked1>) -> !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory>
      %73 = tt.trans %72 {order=array<i32: 1,0>} : !tt.memdesc<32x64xf32, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<64x32xf32, #shared1, #triton_gpu.shared_memory>
      %74 = triton_gpu.local_load %73 : !tt.memdesc<64x32xf32, #shared1, #triton_gpu.shared_memory> -> tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %75 = tt.dot %71, %74, %cst : tensor<64x64xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      %76 = tt.load %61 : tensor<32x32x!tt.ptr<f32>, #blocked1>
      %77 = triton_gpu.convert_layout %75 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %78 = triton_gpu.convert_layout %76 : tensor<32x32xf32, #blocked1> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %79 = tt.dot %77, %78, %arg7 : tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      scf.yield %79 : tensor<64x32xf32, #mma>
    }
    %64 = tt.broadcast %17 : tensor<64x1xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %65 = tt.broadcast %53 : tensor<1x32xi64, #blocked> -> tensor<64x32xi64, #blocked>
    %66 = arith.addi %64, %65 : tensor<64x32xi64, #blocked>
    %67 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<64x32x!tt.ptr<f32>, #blocked>
    %68 = tt.addptr %67, %66 : tensor<64x32x!tt.ptr<f32>, #blocked>, tensor<64x32xi64, #blocked>
    %69 = triton_gpu.convert_layout %63 : tensor<64x32xf32, #mma> -> tensor<64x32xf32, #blocked>
    tt.store %68, %69 : tensor<64x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
} // end module

// -----
// CHECK-DIS: #[[$SHARED_LAYOUT:shared.*]] = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
// CHECK-LABEL: tt.func @indirect_load_shared_layout
// CHECK:  %{{.*}}:8 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}}, %[[ARG14:.*]] = %{{.*}})

// CHECK:  %[[SUBI_20:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_21:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_20]]
// CHECK:  %[[SUBI_22:.*]] = arith.subi %{{.*}}, %{{.*}}
// CHECK:  %[[CMPI_23:.*]] = arith.cmpi slt, %[[ARG6]], %[[SUBI_22]]
// CHECK:  %[[ADDI_24:.*]] = arith.addi %[[ARG10]], %{{.*}}
// CHECK:  %[[CMPI_25:.*]] = arith.cmpi slt, %[[ADDI_24]], %{{.*}}
// CHECK:  %[[SELECT_26:.*]] = arith.select %[[CMPI_25]], %[[ADDI_24]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_27:.*]] = triton_gpu.local_load %[[ARG12]]
// CHECK:  %[[LOCAL_LOAD_28:.*]] = triton_gpu.local_load %[[ARG13]]
// CHECK:  %[[CONVERT_LAYOUT_29:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_27]]
// CHECK:  %[[CONVERT_LAYOUT_30:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_28]]
// CHECK:  %[[DOT_31:.*]] = tt.dot %[[CONVERT_LAYOUT_29]], %[[CONVERT_LAYOUT_30]], %[[ARG7]]
// CHECK:  %[[ADDPTR_32:.*]] = tt.addptr %[[ARG8]], %{{.*}}
// CHECK:  %[[ADDPTR_33:.*]] = tt.addptr %[[ARG9]], %{{.*}}
// CHECK:  %[[SPLAT_34:.*]] = tt.splat %[[CMPI_23]]
// CHECK:  %[[LOAD_35:.*]] = tt.load %[[ADDPTR_32]], %[[SPLAT_34]]
// CHECK:  %[[EXPAND_DIMS_36:.*]] = tt.expand_dims %[[ARG14]] {axis = 1 : i32}
// CHECK:  %[[BROADCAST_37:.*]] = tt.broadcast %[[EXPAND_DIMS_36]]
// CHECK:  %[[MULI_38:.*]] = arith.muli %{{.*}}, %[[BROADCAST_37]]
// CHECK:  %[[ADDPTR_39:.*]] = tt.addptr %{{.*}}, %[[MULI_38]]
// CHECK:  %[[SPLAT_40:.*]] = tt.splat %[[CMPI_23]]
// CHECK:  %[[LOAD_41:.*]] = tt.load %[[ADDPTR_39]], %[[SPLAT_40]]
// CHECK:  %[[SPLAT_42:.*]] = tt.splat %[[CMPI_21]]
// CHECK:  %[[LOAD_43:.*]] = tt.load %[[ADDPTR_33]], %[[SPLAT_42]]
// CHECK:  %[[ADDI_44:.*]] = arith.addi %[[ARG11]], %{{.*}}
// CHECK:  %[[CMPI_45:.*]] = arith.cmpi slt, %[[ADDI_44]], %{{.*}}
// CHECK:  %[[SELECT_46:.*]] = arith.select %[[CMPI_45]], %[[ADDI_44]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_47:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_46]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_35]], %[[MEMDESC_SUBVIEW_47]]
// CHECK:  %[[MEMDESC_SUBVIEW_48:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_46]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_41]], %[[MEMDESC_SUBVIEW_48]]
// CHECK:  scf.yield %[[DOT_31]], %[[ADDPTR_32]], %[[ADDPTR_33]], %[[SELECT_26]], %[[SELECT_46]], %[[MEMDESC_SUBVIEW_47]], %[[MEMDESC_SUBVIEW_48]], %[[LOAD_43]]
// CHECK:  }

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
module attributes {"triton_gpu.target" = "cuda:86", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
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
    %82 = tt.load %arg20 : tensor<16x16x!tt.ptr<f16>, #AL>
    %83 = tt.load %arg21 : tensor<16x!tt.ptr<i64>, #BLs1>
    %84 = tt.expand_dims %83 {axis=1: i32}: tensor<16xi64, #BLs1> -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : tensor<16x1xi64, #BL> -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 : tensor<16x16x!tt.ptr<f16>, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>
  } {tt.num_stages = 3 : i32}
  tt.return %79#0 : tensor<16x16xf32, #C>
}
}


// -----

// CHECK-LABEL: @kernel_yield_constant
// CHECK: tt.load
// CHECK: triton_gpu.memdesc_subview
// CHECK: triton_gpu.local_store
// CHECK: scf.for
// CHECK: tt.load
// CHECK: triton_gpu.memdesc_subview
// CHECK: triton_gpu.local_store
// CHECK: tt.return
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"triton_gpu.target" = "cuda:86", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_yield_constant(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
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
    %22 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %34 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %42 = scf.for %arg7 = %c0_i32 to %13 step %c1_i32 iter_args(%arg8 = %cst) -> (tensor<32x32xf32, #mma>)  : i32 {
      %43 = arith.muli %arg7, %c32_i32 : i32
      %44 = arith.muli %43, %arg5 : i32
      %45 = tt.splat %44 : i32 -> tensor<32x32xi32, #blocked>
      %46 = tt.addptr %22, %45 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %47 = arith.subi %arg4, %43 : i32
      %48 = tt.splat %47 : i32 -> tensor<32x1xi32, #blocked>
      %49 = arith.cmpi slt, %14, %48 : tensor<32x1xi32, #blocked>
      %50 = tt.broadcast %49 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %51 = tt.load %46, %50, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %52 = triton_gpu.convert_layout %51 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %53 = tt.dot %cst_1, %52, %arg8 : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %54 = triton_gpu.convert_layout %53 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %34, %54 : tensor<32x32x!tt.ptr<f32>, #blocked>
      scf.yield %cst1 : tensor<32x32xf32, #mma>
    }
    tt.return
  }
}


// -----

// CHECK-LABEL:  tt.func public @add_kernel
// CHECK:  %[[LOAD_11:.*]] = tt.load %{{.*}}, %{{.*}}
// CHECK:  %[[ADDPTR_12:.*]] = tt.addptr %{{.*}}, %{{.*}}
// CHECK:  %[[LOAD_13:.*]] = tt.load %[[ADDPTR_12]], %{{.*}}
// CHECK:  %[[ADDI_14:.*]] = arith.addi %{{.*}}, %{{.*}}
// CHECK:  %[[SPLAT_15:.*]] = tt.splat %[[ADDI_14]]
// CHECK:  %[[ADDI_16:.*]] = arith.addi %[[SPLAT_15]], %{{.*}}
// CHECK:  %[[CMPI_17:.*]] = arith.cmpi slt, %[[ADDI_16]], %{{.*}}
// CHECK:  %[[ADDPTR_18:.*]] = tt.addptr %{{.*}}, %[[ADDI_16]]
// CHECK:  %[[LOAD_19:.*]] = tt.load %[[ADDPTR_18]], %[[CMPI_17]]
// CHECK:  %[[ADDPTR_20:.*]] = tt.addptr %{{.*}}, %[[ADDI_16]]
// CHECK:  %[[LOAD_21:.*]] = tt.load %[[ADDPTR_20]], %[[CMPI_17]]
// CHECK:  scf.for
#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.target" = "cuda:90", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1016800_i32 = arith.constant 1016800 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1016800_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32  : i32 {
      %7 = arith.addi %1, %arg4 : i32
      %8 = tt.splat %7 : i32 -> tensor<1024xi32, #blocked>
      %9 = arith.addi %8, %2 : tensor<1024xi32, #blocked>
      %10 = arith.cmpi slt, %9, %3 : tensor<1024xi32, #blocked>
      %11 = tt.addptr %4, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %12 = tt.load %11, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
      %13 = tt.addptr %5, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %14 = tt.load %13, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
      %15 = arith.addf %12, %14 : tensor<1024xf32, #blocked>
      %16 = tt.addptr %6, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %16, %15, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}


// -----

// CHECK-LABEL:  tt.func public @nested_loops
// CHECK:  %[[LOAD_10:.*]] = tt.load %{{.*}}
// CHECK:  %[[LOCAL_ALLOC_11:.*]] = triton_gpu.local_alloc %[[LOAD_10]]
// CHECK:  %[[TRANS_12:.*]] = tt.trans %[[LOCAL_ALLOC_11]] {order = array<i32: 1, 0>}
// CHECK:  %[[LOCAL_LOAD_13:.*]] = triton_gpu.local_load %[[TRANS_12]]
// CHECK:  %[[LOCAL_ALLOC_14:.*]] = triton_gpu.local_alloc
// CHECK:  %[[LOAD_15:.*]] = tt.load %{{.*}}, %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_16:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_14]][%{{.*}}, %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_15]], %[[MEMDESC_SUBVIEW_16]]
// CHECK:  %{{.*}}:3 = scf.for %[[ARG2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG3:.*]] = %{{.*}}-1_i32, %[[ARG4:.*]] = %{{.*}}, %[[ARG5:.*]] = %[[MEMDESC_SUBVIEW_16]])

// CHECK:  %[[CMPI_18:.*]] = arith.cmpi slt, %[[ARG2]], %{{.*}}
// CHECK:  %[[ADDI_19:.*]] = arith.addi %[[ARG3]], %{{.*}}
// CHECK:  %[[CMPI_20:.*]] = arith.cmpi slt, %[[ADDI_19]], %{{.*}}
// CHECK:  %[[SELECT_21:.*]] = arith.select %[[CMPI_20]], %[[ADDI_19]], %{{.*}}
// CHECK:  %[[LOCAL_LOAD_22:.*]] = triton_gpu.local_load %[[ARG5]]
// CHECK:  %[[CONVERT_LAYOUT_23:.*]] = triton_gpu.convert_layout %[[LOCAL_LOAD_22]]
// CHECK:  %[[DOT_24:.*]] = tt.dot %[[CONVERT_LAYOUT_23]], %[[LOCAL_LOAD_13]], %{{.*}}
// CHECK:  %[[CONVERT_LAYOUT_25:.*]] = triton_gpu.convert_layout %[[DOT_24]]
// CHECK:  tt.store %{{.*}}, %[[CONVERT_LAYOUT_25]]
// CHECK:  %[[SPLAT_26:.*]] = tt.splat %[[CMPI_18]]
// CHECK:  %[[LOAD_27:.*]] = tt.load %{{.*}}, %[[SPLAT_26]]
// CHECK:  %[[ADDI_28:.*]] = arith.addi %[[ARG4]], %{{.*}}
// CHECK:  %[[CMPI_29:.*]] = arith.cmpi slt, %[[ADDI_28]], %{{.*}}
// CHECK:  %[[SELECT_30:.*]] = arith.select %[[CMPI_29]], %[[ADDI_28]], %{{.*}}
// CHECK:  %[[MEMDESC_SUBVIEW_31:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_14]][%[[SELECT_30]], %{{.*}}, %{{.*}}]
// CHECK:  triton_gpu.local_store %[[LOAD_27]], %[[MEMDESC_SUBVIEW_31]]
// CHECK:  scf.yield %[[SELECT_21]], %[[SELECT_30]], %[[MEMDESC_SUBVIEW_31]]
// CHECK:  }
// CHECK:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_14]]

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 2], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @nested_loops(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %2 = arith.muli %1, %cst_0 : tensor<16x1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %7 = tt.broadcast %4 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %8 = tt.broadcast %6 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %9 = tt.addptr %7, %8 : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %10 = tt.load %9 : tensor<16x16x!tt.ptr<f32>, #blocked>
      %11 = triton_gpu.local_alloc %10 : (tensor<16x16xf32, #blocked>) -> !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory>
      %12 = tt.trans %11 {order = array<i32: 1, 0>} : !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<16x16xf32, #shared1, #triton_gpu.shared_memory>
      %13 = triton_gpu.local_load %12 : !tt.memdesc<16x16xf32, #shared1, #triton_gpu.shared_memory> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %14 = tt.load %9 : tensor<16x16x!tt.ptr<f32>, #blocked>
        %15 = triton_gpu.convert_layout %14 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %16 = tt.dot %15, %13, %cst : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x16xf32, #mma>
        %17 = triton_gpu.convert_layout %16 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
        tt.store %9, %17 : tensor<16x16x!tt.ptr<f32>, #blocked>
      }
    }
    tt.return
  }
}

// -----

// This test triggered some failure in the verifier, so we only
// included a simple check for the kernel name.
// CHECK-LABEL: @load_convert_layout
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func @load_convert_layout(%77: tensor<16x16xi64, #BL> {tt.divisibility=16: i32, tt.constancy=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: tensor<16x!tt.ptr<i64>, #BLs1>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #BLs1>
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %cst_0 = arith.constant dense<2> : tensor<16xi32, #BLs1>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_splat = tt.splat %c1_i32 : i32 -> tensor<16xi32, #BLs1>
  %15 = arith.cmpi slt, %1, %cst_0 : tensor<16xi32, #BLs1>
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>) {
    %82 = tt.load %arg20 : tensor<16x16x!tt.ptr<f16>, #AL>
    %83 = tt.load %arg21, %15 : tensor<16x!tt.ptr<i64>, #BLs1>
    %84 = tt.expand_dims %83 {axis=1: i32}: tensor<16xi64, #BLs1> -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : tensor<16x1xi64, #BL> -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 : tensor<16x16x!tt.ptr<f16>, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>
  } {tt.num_stages = 3 : i32}
  tt.return %79#0 : tensor<16x16xf32, #C>
}
}


// -----

// This test captured some ICE in MatmulLoopPipeline pass, so we only
// included a simple check for the kernel name.
// CHECK-LABEL: @matmul_indirect_pipeline
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 8]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_indirect_pipeline(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %4 = tt.broadcast %2 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %5 = tt.broadcast %3 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %6 = arith.addi %4, %5 : tensor<32x32xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %6 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %9 = tt.load %8 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %6 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %12 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %13 = tt.addptr %12, %0 : tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
      %15 = tt.load %13 : tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %16 = tt.addptr %14, %15 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %17 = tt.load %16 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %18 = tt.expand_dims %17 {axis = 0 : i32} : tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xf32, #blocked>
      %19 = tt.broadcast %18 : tensor<1x32xf32, #blocked> -> tensor<32x32xf32, #blocked>
      %20 = arith.addf %9, %19 : tensor<32x32xf32, #blocked>
      %21 = triton_gpu.convert_layout %9 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %22 = triton_gpu.convert_layout %20 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %23 = tt.dot %21, %22, %cst : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %24 = triton_gpu.convert_layout %23 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %11, %24 : tensor<32x32x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

// CHECK-LABEL: @dont_pipeline_128x1
// CHECK-NOT: local_load{{.*}}128x1
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @dont_pipeline_128x1(%arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_4 = arith.constant dense<-1.000000e+30> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>

    %99:1 = scf.for %arg25 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg31 = %cst_4) -> (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>)  : i32 {
      %94 = tt.splat %arg6 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>, #blocked>
      %151 = tt.load %94 : tensor<128x1x!tt.ptr<i32>, #blocked>
      %161 = triton_gpu.convert_layout %151 : tensor<128x1xi32, #blocked> -> tensor<128x1xi32, #mma>
      %162 = tt.broadcast %161 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
      %170 = arith.sitofp %162 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>

      %173 = "tt.reduce"(%170) <{axis = 1 : i32}> ({
      ^bb0(%arg33: f32, %arg34: f32):
        %207 = arith.maxnumf %arg33, %arg34 : f32
        tt.reduce.return %207 : f32
      }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %175 = arith.maxnumf %arg31, %173 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>

      %201 = arith.truncf %170 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
      %202 = triton_gpu.convert_layout %201 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>

      %192 = arith.constant dense<0.> : tensor<128x64xf32, #mma>
      %203 = arith.constant dense<0.> : tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %204 = tt.dot %202, %203, %192 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>

      scf.yield %175 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    }
    tt.return
  }
}

// -----

// Check that the dependencies across ops of different nesting does not cause crash or
// incorrect schedule that fails to pipeline.
// CHECK-LABEL: @matmul_nested_ops
// CHECK: triton_gpu.local_load

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.target" = "cuda:80"} {
tt.func @matmul_nested_ops(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %ext : index) -> tensor<128x128xf32, #C> {
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
  %b_ptr = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>

  %b_ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
  %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x128xf32, #C>) {
    %cnd = arith.cmpi slt, %iv, %ext : index
    %inc_a_ptr = scf.if %cnd -> (tensor<128x32x!tt.ptr<f16>, #AL>) {
      %a_ptr_ = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
      scf.yield %a_ptr_ : tensor<128x32x!tt.ptr<f16>, #AL>
    } else {
      scf.yield %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    }
    %a_ = tt.load %inc_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %inc_a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    scf.yield %next_a_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#1: tensor<128x128xf32, #C>
}
}

// -----

// Pipeline the if ops at the beginning and the end of the loop
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: dot_prologue_epilogue
  // CHECK-SAME: {{.*}}, {{.*}}, %[[EXT:.*]]: i32, {{.*}}
  tt.func @dot_prologue_epilogue(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %cst2 = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %2 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %10 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %10 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: scf.for %[[IND_VAR:.*]] = %[[C0]] to
    // CHECK-NOT: load
    // CHECK: %[[CND:.*]] = arith.cmpi slt, %[[IND_VAR]], %[[EXT]]
    // CHECK: scf.if %[[CND]]
    // CHECK: dot
    // CHECK: scf.if %[[CND]]
    // CHECK:   arith.mulf
    // CHECK:   scf.yield
    // CHECK-NOT: tt.addptr
    // CHECK: scf.yield
    %17:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2, %arg5 = %16, %arg6 = %8) -> (tensor<128x16xf32, #mma1>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %9 = tt.load %arg6 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %inc_ptr = scf.if %cnd -> tensor<64x16x!tt.ptr<f16>, #blocked> {
        %ptr = tt.addptr %arg5, %inc : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
        scf.yield %ptr : tensor<64x16x!tt.ptr<f16>, #blocked>
      } else {
        scf.yield %arg5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      }
      %18 = tt.load %inc_ptr : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = triton_gpu.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %20 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>
      %acc = triton_nvidia_gpu.warp_group_dot %19, %20, %arg4 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %acc_ = scf.if %cnd -> (tensor<128x16xf32, #mma1>) {
        %acc_zero = arith.mulf %acc, %cst_2 : tensor<128x16xf32, #mma1>
        scf.yield %acc_zero : tensor<128x16xf32, #mma1>
      } else {
        scf.yield %acc : tensor<128x16xf32, #mma1>
      }
      %22 = tt.addptr %arg5, %cst : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      %23 = tt.addptr %arg6, %cst2 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      scf.yield %acc_, %22, %23 : tensor<128x16xf32, #mma1>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>
    }
    tt.return %17#0 : tensor<128x16xf32, #mma1>
  }
}

// -----

// Verify that uses of the ops scheduled in partucular place of the loop (like epilogue if) are correctly scheduled too.
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: pipeline_downstream_dependencies
  // CHECK: {{.*}}, {{.*}}, %[[EXT:.*]]: i32, {{.*}}
  tt.func @pipeline_downstream_dependencies(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ext: i32, %inc: tensor<64x16xi32, #blocked> {tt.divisibility = 16 : i32}) -> tensor<128x16xf32, #mma1> {
    %cst = arith.constant dense<0> : tensor<64x16xi32, #blocked>
    %cst1 = arith.constant dense<1> : tensor<64x16xi32, #blocked>
    %cst2 = arith.constant dense<0> : tensor<128x64xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %2 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %2 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %10 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %10 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // CHECK: %[[C0:.*]] = arith.constant 0 : i32
    // CHECK: scf.for %[[IND_VAR:.*]] = %[[C0]] to
    // CHECK: load
    // CHECK-NOT: load
    // CHECK: dot
    // CHECK: %[[CND:.*]] = arith.cmpi slt, %[[IND_VAR]], %[[EXT]]
    // CHECK: %[[IFRET:.*]]:2 = scf.if %[[CND]]
    // CHECK:   arith.mulf
    // CHECK:   scf.yield
    // CHECK: tt.addptr {{.*}}, %[[IFRET]]#1
    // CHECK: scf.yield
    %17:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst_2, %arg5 = %16, %arg6 = %8) -> (tensor<128x16xf32, #mma1>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %9 = tt.load %arg6 : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %18 = tt.load %arg5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = triton_gpu.local_alloc %9 : (tensor<128x64xf16, #blocked1>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
      %20 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory>
      %acc = triton_nvidia_gpu.warp_group_dot %19, %20, %arg4 : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x16xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x16xf32, #mma1>
      %cnd = arith.cmpi slt, %arg3, %ext : i32
      %if_ret:2 = scf.if %cnd -> (tensor<128x16xf32, #mma1>, tensor<64x16xi32, #blocked>) {
        %acc_zero = arith.mulf %acc, %cst_2 : tensor<128x16xf32, #mma1>
        scf.yield %acc_zero, %cst : tensor<128x16xf32, #mma1>, tensor<64x16xi32, #blocked>
      } else {
        scf.yield %acc, %cst1 : tensor<128x16xf32, #mma1>, tensor<64x16xi32, #blocked>
      }
      %22 = tt.addptr %arg5, %if_ret#1 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
      %23 = tt.addptr %arg6, %cst2 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      scf.yield %if_ret#0, %22, %23 : tensor<128x16xf32, #mma1>, tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked1>
    }
    tt.return %17#0 : tensor<128x16xf32, #mma1>
  }
}

// -----

// CHECK-LABEL: @masked_add_kernel
// CHECK: %[[CONSTANT:.*]] = arith.constant dense<0xFF800000>
// CHECK: tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]
// CHECK: tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]
// CHECK: tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]
// CHECK: tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]
// CHECK: scf.for
// CHECK:   arith.select
// CHECK:   arith.select
// CHECK:   arith.addf
// CHECK:   %[[A:.*]] = tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]
// CHECK:   %[[B:.*]] = tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.target" = "cuda:90", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @masked_add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1016800_i32 = arith.constant 1016800 : i32
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1016800_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32  : i32 {
      %7 = arith.addi %1, %arg4 : i32
      %8 = tt.splat %7 : i32 -> tensor<1024xi32, #blocked>
      %9 = arith.addi %8, %2 : tensor<1024xi32, #blocked>
      %10 = arith.cmpi slt, %9, %3 : tensor<1024xi32, #blocked>
      %11 = tt.addptr %4, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %12 = tt.load %11, %10, %cst : tensor<1024x!tt.ptr<f32>, #blocked>
      %13 = tt.addptr %5, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %14 = tt.load %13, %10, %cst : tensor<1024x!tt.ptr<f32>, #blocked>
      %15 = arith.addf %12, %14 : tensor<1024xf32, #blocked>
      %16 = tt.addptr %6, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %16, %15, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}
