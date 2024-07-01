// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func @test_empty_kernel(%arg0: i64, %arg1: !llvm.ptr<1>)
  // Here the 128 comes from the 4 in module attribute multiples 32
  // CHECK: nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>
  tt.func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {
    // CHECK:  llvm.return
    tt.return
  }
} // end module

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_load
  tt.func @basic_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK: llvm.inline_asm
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: vectorized_load
  tt.func @vectorized_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b32
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: vectorized_load_f16
  tt.func @vectorized_load_f16(%a_ptr_init: tensor<256x!tt.ptr<f16>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf16, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b16
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.b16
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f16>, #blocked0>
    tt.return
  }
}

// -----

// TODO: masked load with vectorization is pending on TODO
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: masked_load_const_other
  tt.func @masked_load_const_other(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

// TODO: masked load with vectorization is pending on TODO
#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: masked_load_const_other_vec
  tt.func @masked_load_const_other_vec(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_with_cache_attr
  tt.func @store_with_cache_attr(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    //      CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.L1::evict_last.b32
    //      CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.L1::evict_last.b32
    tt.store %a_ptr_init, %cst_0, %cst evictionPolicy = evict_last cacheModifier = ca : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_no_vec
  tt.func @global_load_store_no_vec(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 4 elements from vector0
    // CHECK: mov.u32 $0, 0x0
    // CHECK: @${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: @${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: @${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: @${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 4 elements from vector1
    // CHECK: mov.u32 $0, 0x0
    // CHECK: @${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: @${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: @${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: mov.u32 $0, 0x0
    // CHECK: @${{.*}} ld.global.b32 { ${{.*}} }, [ ${{.*}} + 0 ];
    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 4 elements to global
    // CHECK: @${{.*}} st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: @${{.*}} st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: @${{.*}} st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: @${{.*}} st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_vec4
  tt.func @global_load_store_vec4(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 4 elements from A with single one vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 4 elements from B with single one vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 4 elements to global with single one vectorized store instruction
    // CHECK: @$5 st.global.v4.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

// This test verifies the vectorization of Load and Store Ops.
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
// Note, the %n_elements doesn't have a "tt.divisibility" hint, so Triton assumes it's divisibility is 1, this should effect the mask's alignment and further restrict the load/store ops' vector width to be 1.
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32} {
  tt.func @vecadd_masked_vec1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32) {
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<64xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<64xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %9 = tt.splat %n_elements : i32 -> tensor<64xi32, #blocked>
    %10 = arith.cmpi "slt", %4, %9 : tensor<64xi32, #blocked>
    // load op has a vector width = 1 due to the %mask's alignment
    // CHECK: ld.global.b32
    %11 = tt.load %6, %10 : tensor<64x!tt.ptr<f32>, #blocked>
    %12 = tt.load %8, %10 : tensor<64x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %11, %12 : tensor<64xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    tt.store %15, %13, %10 : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec2
    tt.func @global_load_store_vec2(%arg0: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with four vectorized load instruction
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 8 elements from B with four vectorized load instruction
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with four vectorized store instruction
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // CHECK-LABEL: global_load_store_vec2
    tt.func @global_load_store_vec2(%arg0: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with four vectorized load instruction
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 8 elements from B with four vectorized load instruction
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v2.b32 { ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with four vectorized store instruction
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    // CHECK: @${{.*}} st.global.v2.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec8
    tt.func @global_load_store_vec8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with two vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 8 elements from B with two vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with two vectorized store instruction
    // CHECK: @$5 st.global.v4.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    // CHECK: @$5 st.global.v4.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// TODO: Add a testcase to verify the optimization when ptr of the LoadOp
//       is from an addptr with const idx

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_view_broadcast
  tt.func @basic_view_broadcast(%arg : tensor<256xf32,#blocked0>) {
    // CHECK: llvm.mlir.undef
    // CHECK: %[[T0:.*]] = llvm.extractvalue
    // CHECK: %[[T1:.*]] = llvm.extractvalue
    %0 = tt.reshape %arg {allow_reorder = true} : tensor<256xf32, #blocked0> -> tensor<256x1xf32,#blocked2>
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    %1 = tt.broadcast %0 : tensor<256x1xf32,#blocked2> -> tensor<256x4xf32, #blocked2>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: basic_make_range
  tt.func @basic_make_range() {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addf
  tt.func @basic_addf(%arg0 : tensor<256xf32,#blocked0>, %arg1 : tensor<256xf32,#blocked0>) {
    // CHECK: llvm.fadd
    // CHECK: llvm.fadd
    %1 = arith.addf %arg0, %arg1 : tensor<256xf32,#blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addi
  tt.func @basic_addi(%arg0 : tensor<256xi32,#blocked0>, %arg1 : tensor<256xi32,#blocked0>) {
    // CHECK: llvm.add
    // CHECK: llvm.add
    %1 = arith.addi %arg0, %arg1 : tensor<256xi32,#blocked0>
    tt.return
  }
}

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_program_id
  tt.func @basic_program_id() {
    // CHECK: llvm.inline_asm asm_dialect = att operand_attrs = [] "mov.u32 $0, %ctaid.x;", "=r"  : () -> i32
    %0 = tt.get_program_id x : i32
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addptr
  tt.func @basic_addptr(%arg0 : tensor<256x!tt.ptr<f32>,#blocked0>, %arg1 : tensor<256xi32,#blocked0>) {
    // CHECK: llvm.getelementptr
    // CHECK: llvm.getelementptr
    %0 = tt.addptr %arg0, %arg1 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: basic_alloc_tensor
  tt.func @basic_alloc_tensor() {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK-NEXT: llvm.getelementptr
    // CHECK-NEXT: llvm.mlir.constant
    %0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #shared0, #triton_gpu.shared_memory>
    tt.return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: basic_subview
  tt.func @basic_subview() {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.getelementptr
    %index = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    %0 = triton_gpu.local_alloc : () -> !tt.memdesc<128x16x32xf32, #shared0, #triton_gpu.shared_memory>
    %1 = triton_gpu.memdesc_subview %0[%index, %zero, %zero] : !tt.memdesc<128x16x32xf32, #shared0, #triton_gpu.shared_memory> -> !tt.memdesc<16x32xf32, #shared0, #triton_gpu.shared_memory>
    tt.return
  }
}

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_async_wait
  tt.func @basic_async_wait() {
    // CHECK: cp.async.wait_group 0x4
    triton_gpu.async_wait {num = 4: i32}
    tt.return
  }
}

// -----

#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 8], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#slice1d0 = #triton_gpu.slice<{dim = 0, parent = #blocked1}>
#shared1D = #triton_gpu.shared<{vec = 2, perPhase = 1, maxPhase = 8, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0], hasLeadingOffset = true}>
#shared2D = #triton_gpu.shared<{vec = 2, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_1d
  tt.func @basic_insert_slice_async_1d(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<64> : tensor<64xi32, #slice1d0>
    %58 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<64x!tt.ptr<i64>, #slice1d0>
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice1d0>
    %59 = tt.addptr %58, %24 : tensor<64x!tt.ptr<i64>, #slice1d0>, tensor<64xi32, #slice1d0>
    %66 = tt.addptr %59, %cst_2 : tensor<64x!tt.ptr<i64>, #slice1d0>, tensor<64xi32, #slice1d0>
    %71 = triton_gpu.local_alloc : () -> !tt.memdesc<2x64xi64, #shared2D, #triton_gpu.shared_memory, mutable>
    %subview = triton_gpu.memdesc_subview %71[%c0_i32, %c0_i32] :
      !tt.memdesc<2x64xi64, #shared2D, #triton_gpu.shared_memory, mutable> ->
      !tt.memdesc<64xi64, #shared1D, #triton_gpu.shared_memory, mutable>
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, 0x8
    // CHECK: cp.async.commit_group
    %73 = triton_gpu.async_copy_global_to_local %66, %subview : tensor<64x!tt.ptr<i64>, #slice1d0> -> !tt.memdesc<64xi64, #shared1D, #triton_gpu.shared_memory, mutable>
    triton_gpu.async_commit_group %73
    tt.return
  }
}

// -----

#block0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#block3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice2d1 = #triton_gpu.slice<{dim = 1, parent=#block2}>
#slice3d0 = #triton_gpu.slice<{dim = 0, parent=#block3}>
#AL = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#A = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v4
  tt.func @basic_insert_slice_async_v4(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
    %off0_ = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice2d1>
    %off1_ = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : tensor<16xi32, #slice2d1> -> tensor<16x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : tensor<64xi32, #slice3d0> -> tensor<1x64xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : tensor<16x1xi32, #block2> -> tensor<16x64xi32, #block2>
    %cst_scalar = arith.constant 64 : i32
    %cst = tt.splat %cst_scalar : i32 -> tensor<16x64xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<16x64xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : tensor<1x64xi32, #block3> -> tensor<16x64xi32, #block3>
    %broadcast_off0 = triton_gpu.convert_layout %broadcast_off0_ : tensor<16x64xi32, #block2> -> tensor<16x64xi32, #AL>
    %broadcast_off1 = triton_gpu.convert_layout %broadcast_off1_ : tensor<16x64xi32, #block3> -> tensor<16x64xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<16x64xi32, #AL>
    %a_init = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<16x64x!tt.ptr<f32>, #AL>, tensor<16x64xi32, #AL>
    %tensor = triton_gpu.local_alloc : () -> !tt.memdesc<16x64xf32, #A, #triton_gpu.shared_memory>
    %index = arith.constant 1 : i32

    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-SAME: cp.async.cg.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x10, 0x10
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-SAME: cp.async.cg.shared.global [ ${{.*}} + 16 ], [ ${{.*}} + 0 ], 0x10, 0x10
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att
    // CHECK-SAME: cp.async.commit_group
    %a = triton_gpu.async_copy_global_to_local %a_ptr, %tensor : tensor<16x64x!tt.ptr<f32>, #AL> -> !tt.memdesc<16x64xf32, #A, #triton_gpu.shared_memory>
    triton_gpu.async_commit_group
    tt.return
  }
}

// -----

#block0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#block3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice2d1 = #triton_gpu.slice<{dim = 1, parent=#block2}>
#slice3d0 = #triton_gpu.slice<{dim = 0, parent=#block3}>
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#A = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v1
  tt.func @basic_insert_slice_async_v1(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}) {
    %off0_ = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice2d1>
    %off1_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : tensor<16xi32, #slice2d1> -> tensor<16x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : tensor<32xi32, #slice3d0> -> tensor<1x32xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : tensor<16x1xi32, #block2> -> tensor<16x32xi32, #block2>
    %cst_scalar = arith.constant 32 : i32
    %cst = tt.splat %cst_scalar : i32 -> tensor<16x32xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<16x32xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : tensor<1x32xi32, #block3> -> tensor<16x32xi32, #block3>
    %broadcast_off0 = triton_gpu.convert_layout %broadcast_off0_ : tensor<16x32xi32, #block2> -> tensor<16x32xi32, #AL>
    %broadcast_off1 = triton_gpu.convert_layout %broadcast_off1_ : tensor<16x32xi32, #block3> -> tensor<16x32xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<16x32xi32, #AL>
    %a_init = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<16x32x!tt.ptr<f32>, #AL>, tensor<16x32xi32, #AL>
    %tensor = triton_gpu.local_alloc : () -> !tt.memdesc<16x32xf32, #A, #triton_gpu.shared_memory>
    %index = arith.constant 1 : i32

    // CHECK: llvm.inline_asm
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.commit_group
    %a = triton_gpu.async_copy_global_to_local %a_ptr, %tensor : tensor<16x32x!tt.ptr<f32>, #AL> -> !tt.memdesc<16x32xf32, #A, #triton_gpu.shared_memory>
    triton_gpu.async_commit_group
    tt.return
  }
}

// -----

#block0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#block3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice2d1 = #triton_gpu.slice<{dim = 1, parent=#block2}>
#slice3d0 = #triton_gpu.slice<{dim = 0, parent=#block3}>
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#A = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_v1_multictas
  tt.func @basic_insert_slice_async_v1_multictas(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}) {
    %off0_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice2d1>
    %off1_ = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : tensor<32xi32, #slice2d1> -> tensor<32x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : tensor<32xi32, #slice3d0> -> tensor<1x32xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : tensor<32x1xi32, #block2> -> tensor<32x32xi32, #block2>
    %cst_scalar = arith.constant 32 : i32
    %cst = tt.splat %cst_scalar : i32 -> tensor<32x32xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<32x32xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : tensor<1x32xi32, #block3> -> tensor<32x32xi32, #block3>
    %broadcast_off0 = triton_gpu.convert_layout %broadcast_off0_ : tensor<32x32xi32, #block2> -> tensor<32x32xi32, #AL>
    %broadcast_off1 = triton_gpu.convert_layout %broadcast_off1_ : tensor<32x32xi32, #block3> -> tensor<32x32xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<32x32xi32, #AL>
    %a_init = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %tensor = triton_gpu.local_alloc : () -> !tt.memdesc<32x32xf32, #A, #triton_gpu.shared_memory>
    %index = arith.constant 1 : i32

    // CHECK: llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.mul
    // CHECK: llvm.add
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x4, 0x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: cp.async.commit_group
    %a = triton_gpu.async_copy_global_to_local %a_ptr, %tensor : tensor<32x32x!tt.ptr<f32>, #AL> -> !tt.memdesc<32x32xf32, #A, #triton_gpu.shared_memory>
    triton_gpu.async_commit_group
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: basic_splat
  tt.func @basic_splat(%ptr: !tt.ptr<f32>) {
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    %0 = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>,#blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_store
  tt.func @basic_store(%ptrs: tensor<256x!tt.ptr<f32>, #blocked0>, %vals: tensor<256xf32, #blocked0>, %mask: tensor<256xi1, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    // CHECK: llvm.inline_asm
    // CHECK-SAME: st.global.b32 [ ${{.*}} + 0 ], { ${{.*}} };
    tt.store %ptrs, %vals, %mask : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_blocked_blocked
  tt.func @convert_layout_blocked_blocked(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf32, #blocked0> -> tensor<16x16xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_blocked_blocked_vec
  tt.func @convert_layout_blocked_blocked_vec(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf32, #blocked0> -> tensor<16x16xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_blocked_blocked_multi_rep
  tt.func @convert_layout_blocked_blocked_multi_rep(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.mlir.addressof @global_smem
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf32, #blocked0> -> tensor<16x16xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma0 = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot
  tt.func @convert_dot(%A: tensor<16x16xf16, #blocked0>, %B: tensor<16x16xf16, #blocked0>) {
    %AA = triton_gpu.local_alloc %A : (tensor<16x16xf16, #blocked0>) -> !tt.memdesc<16x16xf16, #shared0, #triton_gpu.shared_memory>
    %BB = triton_gpu.local_alloc %B : (tensor<16x16xf16, #blocked0>) -> !tt.memdesc<16x16xf16, #shared0, #triton_gpu.shared_memory>
    // CHECK: llvm.inline_asm
    // CHECK: ldmatrix.sync.aligned.m8n8.x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4
    %AA_DOT = triton_gpu.local_load %AA : !tt.memdesc<16x16xf16, #shared0, #triton_gpu.shared_memory> -> tensor<16x16xf16, #dot_operand_a>
    %BB_DOT = triton_gpu.local_load %BB : !tt.memdesc<16x16xf16, #shared0, #triton_gpu.shared_memory> -> tensor<16x16xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma0>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>

    tt.return
  }
}

// TODO: problems in MLIR's parser on slice layout
// #blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
// module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
//   tt.func @make_range_sliced_layout() {
//     %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked0}>>
//     tt.return
//   }
// }

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_mmav2_block
  tt.func @convert_layout_mmav2_blocked(%arg0: tensor<32x16xf32, #mma>) {
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x16xf32, #mma> -> tensor<32x16xf32, #blocked0>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 1, versionMinor = 3, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 16]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_mmav1_block
  tt.func @convert_layout_mmav1_blocked(%arg0: tensor<32x64xf32, #mma>) {
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: nvvm.barrier0
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x64xf32, #mma> -> tensor<32x64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_mmav3_transpose
  tt.func @convert_layout_mmav3_transpose(%arg0: tensor<128x256xf8E5M2, #mma>) {
    // CHECK-COUNT-128: llvm.store %{{.*}} : vector<1xi8>, !llvm.ptr<3>
    // CHECK: nvvm.barrier0
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xi8>
    %0 = triton_gpu.convert_layout %arg0 : tensor<128x256xf8E5M2, #mma> -> tensor<128x256xf8E5M2, #blocked>
    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK: llvm.mlir.global external @global_smem
  // CHECK-LABEL: convert_layout_blocked_shared
  tt.func @convert_layout_blocked_shared(%arg0: tensor<128x32xf32, #blocked0>) {
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: !llvm.ptr<3>
    %0 = triton_gpu.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !tt.memdesc<128x32xf32, #shared0, #triton_gpu.shared_memory>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked1d_to_slice0
  tt.func @convert_blocked1d_to_slice0(%src:tensor<32xi32, #blocked0>) {
    // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xi32>
    %cvt = triton_gpu.convert_layout %src : tensor<32xi32, #blocked0> -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked1d_to_slice1
  tt.func @convert_blocked1d_to_slice1(%src:tensor<32xi32, #blocked0>) {
    // CHECK-COUNT-8: llvm.load {{.*}} : !llvm.ptr<3>
    %cvt = triton_gpu.convert_layout %src : tensor<32xi32, #blocked0> -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked_to_blocked_ptr
  tt.func @convert_blocked_to_blocked_ptr(%src:tensor<32x!tt.ptr<f32>, #blocked0>) {
    // CHECK: llvm.ptrtoint
    // CHECK: llvm.store
    // CHECK: nvvm.barrier0
    // CHECK: llvm.inttoptr
    // CHECK-COUNT-4: llvm.insertvalue
    %cvt = triton_gpu.convert_layout %src : tensor<32x!tt.ptr<f32>, #blocked0> -> tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @matmul_kernel_dot_operand_layout(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory>, %b:!tt.memdesc<32x256xf16, #shared, #triton_gpu.shared_memory>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    // CHECK: ldmatrix.sync.aligned.m8n8.x4.shared.b16
    %a_mat = triton_gpu.local_load %a : !tt.memdesc<128x32xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x32xf16, #dot_operand_a>
    %b_mat = triton_gpu.local_load %b : !tt.memdesc<32x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<32x256xf16, #dot_operand_b>

    %28 = tt.dot %a_mat, %b_mat, %cst : tensor<128x32xf16, #dot_operand_a> * tensor<32x256xf16, #dot_operand_b> -> tensor<128x256xf32, #mma>
    %38 = triton_gpu.convert_layout %28 : tensor<128x256xf32, #mma> -> tensor<128x256xf32, #blocked>

    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 1, versionMinor = 3, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 16]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @matmul884_kernel_dot_operand_layout(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!tt.memdesc<32x64xf16, #shared0, #triton_gpu.shared_memory>, %b:!tt.memdesc<64x64xf16, #shared1, #triton_gpu.shared_memory>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    // CHECK: ldmatrix.sync.aligned.m8n8.x4.shared.b16
    %a_mat = triton_gpu.local_load %a : !tt.memdesc<32x64xf16, #shared0, #triton_gpu.shared_memory> -> tensor<32x64xf16, #dot_operand_a>
    %b_mat = triton_gpu.local_load %b : !tt.memdesc<64x64xf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x64xf16, #dot_operand_b>

    %28 = tt.dot %a_mat, %b_mat, %cst : tensor<32x64xf16, #dot_operand_a> * tensor<64x64xf16, #dot_operand_b> -> tensor<32x64xf32, #mma>
    %38 = triton_gpu.convert_layout %28 : tensor<32x64xf32, #mma> -> tensor<32x64xf32, #blocked>
    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<32x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @matmul_fmadot(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!tt.memdesc<32x16xf32, #shared, #triton_gpu.shared_memory>, %b:!tt.memdesc<16x32xf32, #shared, #triton_gpu.shared_memory>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    // CHECK: llvm.intr.fmuladd
    %a_mat = triton_gpu.local_load %a : !tt.memdesc<32x16xf32, #shared, #triton_gpu.shared_memory> -> tensor<32x16xf32, #dot_operand_a>
    %b_mat = triton_gpu.local_load %b : !tt.memdesc<16x32xf32, #shared, #triton_gpu.shared_memory> -> tensor<16x32xf32, #dot_operand_b>

    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = ieee : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #blocked>
    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %28 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma, kWidth=1}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma, kWidth=1}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_tf32dot
  tt.func @matmul_tf32dot(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!tt.memdesc<32x16xf32, #shared, #triton_gpu.shared_memory>, %b:!tt.memdesc<16x32xf32, #shared, #triton_gpu.shared_memory>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4.shared.b16
    // CHECK-SAME: (i32, i32, i32, i32)
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4.shared.b16
    // CHECK-SAME: (i32, i32, i32, i32)
    %a_mat = triton_gpu.local_load %a : !tt.memdesc<32x16xf32, #shared, #triton_gpu.shared_memory> -> tensor<32x16xf32, #dot_operand_a>
    %b_mat = triton_gpu.local_load %b : !tt.memdesc<16x32xf32, #shared, #triton_gpu.shared_memory> -> tensor<16x32xf32, #dot_operand_b>

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = tf32 : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #mma>
    %38 = triton_gpu.convert_layout %28 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>

    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.gpu.relaxed.add.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.gpu.relaxed.add.f32
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.return
  }
}

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32_scalar
  tt.func @atomic_add_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : i1, %arg2 : f32) {
    // CHECK: llvm.icmp "eq"
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.gpu.relaxed.add.f32
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32_sys_scope(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.sys.relaxed.add.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$3 atom.global.sys.relaxed.add.f32
    %0 = tt.atomic_rmw fadd, relaxed, sys, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_f32
  tt.func @store_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$2 st.global.b32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$2 st.global.b32
    tt.store %arg0, %arg1 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_f32_scalar
  tt.func @store_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : f32) {
    // CHECK: llvm.icmp "eq"
    // CHECK: llvm.inline_asm
    // CHECK-SAME: @$2 st.global.b32
    tt.store %arg0, %arg1 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
// CHECK-LABEL: test_get_program_id
tt.func @test_get_program_id(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
  %blockidx = tt.get_program_id x: i32
  %blockidy = tt.get_program_id y: i32
  %blockidz = tt.get_program_id z: i32
  // CHECK: ctaid.x
  // CHECK: ctaid.y
  // CHECK: ctaid.z
  %v0 = arith.addi %blockidx, %blockidy : i32
  %v1 = arith.addi %v0, %blockidz : i32
  %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
  tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

  tt.return
}

}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [4], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 4 : i32, "triton_gpu.num-warps" = 4 : i32} {
// CHECK-LABEL: test_get_program_id
tt.func @test_get_program_id(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
  %blockidx = tt.get_program_id x: i32
  %blockidy = tt.get_program_id y: i32
  %blockidz = tt.get_program_id z : i32
  // CHECK: clusterid.x
  // CHECK: clusterid.y
  // CHECK: clusterid.z
  %v0 = arith.addi %blockidx, %blockidy : i32
  %v1 = arith.addi %v0, %blockidz : i32
  %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
  tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

  tt.return
}

}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: test_get_num_program
  tt.func @test_get_num_program(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
    %blockdimx = tt.get_num_programs x : i32
    %blockdimy = tt.get_num_programs y : i32
    %blockdimz = tt.get_num_programs z : i32
    // CHECK: nctaid.x
    // CHECK: nctaid.y
    // CHECK: nctaid.z
    %v0 = arith.addi %blockdimx, %blockdimy : i32
    %v1 = arith.addi %v0, %blockdimz : i32
    %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
    tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [4], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 4 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @test_get_num_program(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
    %blockdimx = tt.get_num_programs x : i32
    %blockdimy = tt.get_num_programs y : i32
    %blockdimz = tt.get_num_programs z : i32
    // CHECK: nclusterid.x
    // CHECK: nclusterid.y
    // CHECK: nclusterid.z
    %v0 = arith.addi %blockdimx, %blockdimy : i32
    %v1 = arith.addi %v0, %blockdimz : i32
    %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
    tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: test_index_cache
  tt.func @test_index_cache() {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: test_base_index_cache
  tt.func @test_base_index_cache(%arg0: tensor<128x32xf32, #blocked0>) {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    %0 = triton_gpu.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !tt.memdesc<128x32xf32, #shared0, #triton_gpu.shared_memory>
    %1 = triton_gpu.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !tt.memdesc<128x32xf32, #shared0, #triton_gpu.shared_memory>
    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: test_index_cache_different_block
  tt.func @test_index_cache_different_block(%arg0: tensor<128x32xf32, #blocked0>, %arg1: i1) {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    %0 = triton_gpu.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !tt.memdesc<128x32xf32, #shared0, #triton_gpu.shared_memory>
    cf.cond_br %arg1, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %1 = triton_gpu.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !tt.memdesc<128x32xf32, #shared0, #triton_gpu.shared_memory>
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      tt.return
  }
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma, kWidth=1}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma, kWidth=1}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_tf32_cst_b
  tt.func @matmul_tf32_cst_b(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a: tensor<32x16xf32, #dot_operand_a>, %c: tensor<32x32xf32, #mma>) {
  // CHECK: %[[CST:.+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[BC:.+]] = llvm.bitcast %[[CST]] : f32 to i32
  // CHECK: %[[SI:.+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
  // CHECK: llvm.insertvalue %[[BC]], %[[SI]][0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %b_mat = arith.constant dense<1.000000e+00> : tensor<16x32xf32, #dot_operand_b>
    %28 = tt.dot %a, %b_mat, %c, inputPrecision = tf32 : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #mma>
    %38 = triton_gpu.convert_layout %28 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_f16_cst_operands
  tt.func public @matmul_f16_cst_operands(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
  // CHECK: %[[C1f:.+]] = llvm.mlir.constant(1.000000e+00 : f16) : f16
  // CHECK: %[[Ci16:.+]] = llvm.bitcast %[[C1f]] : f16 to i16
  // CHECK: %[[U:.+]] = llvm.mlir.undef : vector<2xi16>
  // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[V0:.+]] = llvm.insertelement %[[Ci16]], %[[U]][%[[C0]] : i32] : vector<2xi16>
  // CHECK: %[[C1:.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[V1:.+]] = llvm.insertelement %[[Ci16]], %[[V0]][%[[C1]] : i32] : vector<2xi16>
  // CHECK: %[[BC:.+]] = llvm.bitcast %[[V1]] : vector<2xi16> to i32
  // CHECK: %[[SU:.+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
  // CHECK: llvm.insertvalue %[[BC]], %[[SU]][0] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_2 = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = triton_gpu.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %4 = arith.muli %3, %cst_2 : tensor<32x1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %7 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %9 = tt.broadcast %6 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %10 = tt.broadcast %8 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %11 = tt.addptr %9, %10 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %12 = arith.truncf %1 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    tt.store %11, %12 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: test_s8_to_bf16_conversion
  tt.func @test_s8_to_bf16_conversion(%in: tensor<32xi8, #blocked>) {
    // We can't vectorize if we only process
    // CHECK-NOT: llvm.inline_asm
    // CHECK: llvm.sitofp
    // CHECK-NOT: llvm.sitofp
    %out = arith.sitofp %in : tensor<32xi8, #blocked> to tensor<32xbf16, #blocked>
    tt.return
  }
}

// -----
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], instrShape = [16, 8]}>
#dot = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: test_s8_to_bf16_vectorized_conversion
  tt.func @test_s8_to_bf16_vectorized_conversion(%in: tensor<16x16xi8, #mma>) {
    // CHECK-NOT: llvm.sitofp
    // 8 elements per thread => we should process 2 vectors of 4
    // CHECK: llvm.inline_asm
    // CHECK: llvm.inline_asm
    // CHECK-NOT: llvm.inline_asm
    %out = arith.sitofp %in : tensor<16x16xi8, #mma> to tensor<16x16xbf16, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: sum_reduction
//       CHECK:  %[[M:.+]] = llvm.mlir.constant(-1 : i32) : i32
//       CHECK:   nvvm.redux.sync  add %{{.*}}, %[[M]]
//       CHECK:   nvvm.barrier0
//       CHECK:   nvvm.shfl.sync bfly
//       CHECK:   nvvm.shfl.sync bfly
//       CHECK:   nvvm.barrier0
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @sum_reduction(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1024> : tensor<1x1xi32, #blocked>
    %0 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #blocked1>
    %1 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1xi32, #blocked>
    %3 = arith.muli %2, %cst : tensor<1x1xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x1x!tt.ptr<i32>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<1x1x!tt.ptr<i32>, #blocked>, tensor<1x1xi32, #blocked>
    %6 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<1024xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xi32, #blocked>
    %8 = tt.broadcast %5 : tensor<1x1x!tt.ptr<i32>, #blocked> -> tensor<1x1024x!tt.ptr<i32>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<1x1024x!tt.ptr<i32>, #blocked>, tensor<1x1024xi32, #blocked>
    %10 = tt.load %9 : tensor<1x1024x!tt.ptr<i32>, #blocked>
    %11 = "tt.reduce"(%10) <{axis = 1 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %15 = arith.addi %arg2, %arg3 : i32
      tt.reduce.return %15 : i32
    }) : (tensor<1x1024xi32, #blocked>) -> tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %12 = triton_gpu.convert_layout %11 : tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<1xi32, #blocked1>
    %13 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked1>
    %14 = tt.addptr %13, %0 : tensor<1x!tt.ptr<i32>, #blocked1>, tensor<1xi32, #blocked1>
    tt.store %14, %12 : tensor<1x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice = #triton_gpu.slice<{dim = 1, parent = #blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: reduce_bools
  tt.func public @reduce_bools(%arg: tensor<256x2xi1, #blocked>) {
    // CHECK: llvm.mlir.addressof @global_smem
    %24 = "tt.reduce"(%arg) <{axis = 1 : i32}> ({
    ^bb0(%arg4: i1, %arg5: i1):
      %48 = arith.ori %arg4, %arg5 : i1
      tt.reduce.return %48 : i1
    }) : (tensor<256x2xi1, #blocked>) -> tensor<256xi1, #slice>
    tt.return
  }
}


// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: inline_asm
  tt.func public @inline_asm(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<512x!tt.ptr<i8>, #blocked>, tensor<512xi32, #blocked>
    %3 = tt.load %2 : tensor<512x!tt.ptr<i8>, #blocked>
// CHECK: %{{.*}} = llvm.inline_asm asm_dialect = att "shl.b32 $0, $0, 3;", "=r,r" %{{.*}} : (vector<4xi8>) -> vector<4xi8>
    %4 = tt.elementwise_inline_asm "shl.b32 $0, $0, 3;" {constraints = "=r,r", packed_element = 4 : i32, pure = true} %3 : tensor<512xi8, #blocked> -> tensor<512xi8, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>, #blocked>
    %6 = tt.addptr %5, %0 : tensor<512x!tt.ptr<i8>, #blocked>, tensor<512xi32, #blocked>
    tt.store %6, %4 : tensor<512x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: inline_asm_pack_16bit
  tt.func public @inline_asm_pack_16bit(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<512x!tt.ptr<i8>, #blocked>, tensor<512xi32, #blocked>
    %3 = tt.load %2 : tensor<512x!tt.ptr<i8>, #blocked>
// CHECK: %{{.*}} = llvm.inline_asm asm_dialect = att "shl.b16 $0, $0, 3;", "=h,h" %{{.*}} : (vector<2xi8>) -> vector<2xi8>
    %4 = tt.elementwise_inline_asm "shl.b16 $0, $0, 3;" {constraints = "=h,h", packed_element = 2 : i32, pure = true} %3 : tensor<512xi8, #blocked> -> tensor<512xi8, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>, #blocked>
    %6 = tt.addptr %5, %0 : tensor<512x!tt.ptr<i8>, #blocked>, tensor<512xi32, #blocked>
    tt.store %6, %4 : tensor<512x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

// -----

//  CHECK-LABEL: reduce_slice
//  CHECK-NOT: st.shared
//  CHECK-NOT: ld.shared
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 4, 2], warpsPerCTA = [2, 4, 2], order = [2, 0, 1], CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [0, 1, 2]}>
#sliced2 = #triton_gpu.slice<{dim = 2, parent = #blocked}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 16 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_slice() attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<4x1xi1, #sliced2>
    %0 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
    ^bb0(%arg0: i1, %arg1: i1):
      %1 = arith.ori %arg0, %arg1 : i1
      tt.reduce.return %1 : i1
    }) : (tensor<4x1xi1, #sliced2>) -> tensor<4xi1, #triton_gpu.slice<{dim = 1, parent = #sliced2}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: reduce_md_slice
//  CHECK: st.shared
//  CHECK: st.shared
//  CHECK: ld.shared
//  CHECK: st.shared
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#sliced = #triton_gpu.slice<{dim = 2, parent = #blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_md_slice(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x128xf32, #triton_gpu.slice<{dim = 2, parent = #blocked}>>
    %0 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %18 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %18 : f32
    }) {allocation.offset = 0 : i32} : (tensor<2x128xf32, #sliced>) -> tensor<2xf32, #triton_gpu.slice<{dim = 1, parent = #sliced}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: volta_dot
#mma = #triton_gpu.nvidia_mma<{versionMajor = 1, versionMinor = 2, warpsPerCTA = [1, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 16]}>
module attributes {"triton_gpu.target" = "cuda:70", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @volta_dot() {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %a = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
    %b = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>

    %87 = tt.dot %a, %b, %cst : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"triton_gpu.target" = "cuda:75", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: convert_single_element
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  // CHECK: llvm.insertvalue
  // CHECK: llvm.extractvalue
  tt.func public @convert_single_element() attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+03> : tensor<1xf32, #blocked1>
    %0 = triton_gpu.convert_layout %cst : tensor<1xf32, #blocked1> -> tensor<1xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"triton_gpu.target" = "cuda:75", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: convert_single_element_and_add
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  // CHECK: llvm.insertvalue
  // CHECK: llvm.extractvalue
  tt.func public @convert_single_element_and_add() attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+03> : tensor<1xf32, #blocked1>
    %cst2 = arith.constant dense<1.000000e+03> : tensor<1xf32, #blocked>
    %0 = triton_gpu.convert_layout %cst : tensor<1xf32, #blocked1> -> tensor<1xf32, #blocked>
    %1 = arith.addf %0, %cst2 : tensor<1xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.target" = "cuda:90", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @vectorize_shmem_load
  // CHECK: llvm.load
  // CHECK-SAME: {alignment = 8 : i64} : !llvm.ptr<3> -> vector<8xi8>
  // CHECK-NOT: llvm.load
  tt.func public @vectorize_shmem_load(%shmem : !tt.memdesc<16x16xi8, #shared, #triton_gpu.shared_memory>) {
    %0 = triton_gpu.local_load %shmem : !tt.memdesc<16x16xi8, #shared, #triton_gpu.shared_memory> -> tensor<16x16xi8, #blocked>
    tt.return
  }
}

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.target" = "cuda:90", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @vectorize_shmem_store
  // CHECK: llvm.store
  // CHECK-SAME: {alignment = 64 : i64} : vector<16xi32>, !llvm.ptr<3>
  // CHECK-NOT: llvm.store
  tt.func public @vectorize_shmem_store(%block : tensor<64x64xi32, #blocked>) {
    %0 = triton_gpu.local_alloc %block : (tensor<64x64xi32, #blocked>) -> !tt.memdesc<64x64xi32, #shared, #triton_gpu.shared_memory>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: abs_is_int_min_poison
  // CHECK: %{{.*}} = "llvm.intr.abs"(%{{.*}}) <{is_int_min_poison = false}> : (i32) -> i32
  tt.func @abs_is_int_min_poison(%arg0 : tensor<256xi32, #blocked0>) {
    %abs = math.absi %arg0 : tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: test_local_load_bf16
  // CHECK: llvm.extractelement {{.*}} : vector<8xbf16>
  tt.func public @test_local_load_bf16() {
    %c0_i32 = arith.constant 0 : i32
    %19 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x1x2048xbf16, #shared, #triton_gpu.shared_memory, mutable>
    %22 = triton_gpu.memdesc_subview %19[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x1x2048xbf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<1x2048xbf16, #shared, #triton_gpu.shared_memory, mutable>
    %39 = triton_gpu.local_load %22 : !tt.memdesc<1x2048xbf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<1x2048xbf16, #blocked>
    %40 = arith.extf %39 : tensor<1x2048xbf16, #blocked> to tensor<1x2048xf32, #blocked>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: test_local_store
  // CHECK: llvm.store
  tt.func public @test_local_store(%arg0: tensor<1xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = triton_gpu.local_alloc {allocation.offset = 0 : i32} : () -> !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %arg0, %0 : tensor<1xf32, #blocked> -> !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: test_local_store_subview
  // CHECK: llvm.store
  tt.func public @test_local_store_subview(%arg0: tensor<1xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = triton_gpu.local_alloc {allocation.offset = 0 : i32} : () -> !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable>
    %sv = triton_gpu.memdesc_subview %0[%c0_i32] : !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %arg0, %sv : tensor<1xf32, #blocked> -> !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: print_ptr
  // CHECK: llvm.call @vprintf(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
  tt.func @print_ptr(%arg0 : tensor<256x!tt.ptr<i32>, #blocked0>) {
    tt.print "ptr: " {hex = false} : %arg0 : tensor<256x!tt.ptr<i32>, #blocked0>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [2, 2, 8], warpsPerCTA = [1, 1, 1], order = [2, 0, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reduce_3d
  // CHECK: %[[C16:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: llvm.and %{{.+}}, %[[C16]]
  // CHECK: %[[C16_2:.+]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK: nvvm.shfl.sync  bfly %{{.*}}, %{{.*}}, %[[C16_2]]
  tt.func @reduce_3d(%arg0 : tensor<2x2xf32, #triton_gpu.slice<{dim = 2, parent = #blocked}>>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
        ^bb0(%arg1: f32, %arg2: f32):
          %112 = arith.addf %arg1, %arg2 : f32
          tt.reduce.return %112 : f32
        }) : (tensor<2x2xf32, #triton_gpu.slice<{dim = 2, parent = #blocked}>>) -> tensor<2xf32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.slice<{dim = 2, parent = #blocked}>}>>
    tt.return
  }
}
