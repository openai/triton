// RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#slice1dim1 = #triton_gpu.slice<{dim = 1, parent = #blocked1}>
#slice2dim0 = #triton_gpu.slice<{dim = 0, parent = #blocked2}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {

// CHECK: [[row_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: [[col_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: [[load_ptr:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64x!tt.ptr<f32, 1>, [[row_layout]]>
// CHECK: [[load_mask:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64xi1, [[row_layout]]>
// CHECK: [[load_other:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64xf32, [[row_layout]]>
// CHECK: [[load_val:%.*]] = tt.load [[load_ptr]], [[load_mask]], [[load_other]] {{.*}} : tensor<64x64xf32, [[row_layout]]>
// CHECK: [[store_ptr:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64x!tt.ptr<f32, 1>, [[col_layout]]>
// CHECK: [[store_val:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64xf32, [[col_layout]]>
// CHECK: [[store_mask:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x64xi1, [[col_layout]]>
// CHECK: tt.store [[store_ptr]], [[store_val]], [[store_mask]]
tt.func @transpose(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                %arg1: i32 {tt.divisibility = 16 : i32},
                %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                %arg3: i32 {tt.divisibility = 16 : i32}) {
  %cst = arith.constant dense<true> : tensor<64x64xi1, #blocked1>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
  %00 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice1dim1>
  %01 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice2dim0>
  %1 = tt.expand_dims %00 {axis = 1 : i32} : tensor<64xi32, #slice1dim1> -> tensor<64x1xi32, #blocked1>
  %2 = tt.splat %arg1 : i32 -> tensor<64x1xi32, #blocked1>
  %3 = arith.muli %1, %2 : tensor<64x1xi32, #blocked1>
  %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %5 = tt.addptr %4, %3 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %6 = tt.expand_dims %01 {axis = 0 : i32} : tensor<64xi32, #slice2dim0> -> tensor<1x64xi32, #blocked2>
  %7 = tt.broadcast %5 : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %8 = tt.broadcast %6 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %9 = triton_gpu.convert_layout %8 : tensor<64x64xi32, #blocked2> -> tensor<64x64xi32, #blocked1>
  %10 = tt.addptr %7, %9 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %11 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked1>
  %12 = tt.addptr %11, %1 : tensor<64x1x!tt.ptr<f32>, #blocked1>, tensor<64x1xi32, #blocked1>
  %13 = tt.splat %arg3 : i32 -> tensor<1x64xi32, #blocked2>
  %14 = arith.muli %6, %13 : tensor<1x64xi32, #blocked2>
  %15 = tt.broadcast %12 : tensor<64x1x!tt.ptr<f32>, #blocked1> -> tensor<64x64x!tt.ptr<f32>, #blocked1>
  %16 = tt.broadcast %14 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
  %17 = triton_gpu.convert_layout %16 : tensor<64x64xi32, #blocked2> -> tensor<64x64xi32, #blocked1>
  %18 = tt.addptr %15, %17 : tensor<64x64x!tt.ptr<f32>, #blocked1>, tensor<64x64xi32, #blocked1>
  %19 = tt.load %10, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, #blocked1>
  tt.store %18, %19, %cst : tensor<64x64xf32, #blocked1>
  tt.return
}

}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {


// CHECK: [[NARROW_LAYOUT:#.*]] = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: [[WIDE_LAYOUT:#.*]] = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
tt.func public @load_tensors_two_types(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<1024x!tt.ptr<f32, 1>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32, 1>, #blocked>, tensor<1024xi32, #blocked>
    %9 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f16, 1> -> tensor<1024x!tt.ptr<f16, 1>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f16, 1>, #blocked>, tensor<1024xi32, #blocked>
    %12 = tt.load %11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf16, #blocked>
    %13 = arith.extf %12 : tensor<1024xf16, #blocked> to tensor<1024xf32, #blocked>
    %14 = arith.addf %9, %13 : tensor<1024xf32, #blocked>
    %15 = tt.splat %arg2 : !tt.ptr<f32, 1> -> tensor<1024x!tt.ptr<f32, 1>, #blocked>
    %16 = tt.addptr %15, %4 : tensor<1024x!tt.ptr<f32, 1>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: tt.store {{.*}} : tensor<1024xf32, [[WIDE_LAYOUT]]>
    tt.store %16, %14 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32, #blocked>
    tt.return
}

}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK-NOT: sizePerThread = [4]
// CHECK: #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-NOT: sizePerThread = [4]
tt.func public @load_tensors_two_types(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<1024x!tt.ptr<f32, 1>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32, 1>, #blocked>, tensor<1024xi32, #blocked>
    %9 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f16, 1> -> tensor<1024x!tt.ptr<f16, 1>, #blocked>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f16, 1>, #blocked>, tensor<1024xi32, #blocked>
    %12 = tt.load %11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf16, #blocked>
    %13 = arith.extf %12 : tensor<1024xf16, #blocked> to tensor<1024xf32, #blocked>
    %14 = arith.addf %9, %13 : tensor<1024xf32, #blocked>
    %15 = tt.splat %arg2 : !tt.ptr<f16, 1> -> tensor<1024x!tt.ptr<f16, 1>, #blocked>
    %16 = tt.addptr %15, %4 : tensor<1024x!tt.ptr<f16, 1>, #blocked>, tensor<1024xi32, #blocked>
    %17 = arith.truncf %14 : tensor<1024xf32, #blocked> to tensor<1024xf16, #blocked>
    tt.store %16, %17 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf16, #blocked>
    tt.return
}

}


// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {

// CHECK: [[NARROW_LAYOUT:#.*]] = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
tt.func public @load_tensors_with_masks(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi "slt", %4, %5 : tensor<1024xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32, 1> -> tensor<1024x!tt.ptr<f32, 1>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32, 1>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: tt.load {{.*}} : tensor<1024xf32, [[NARROW_LAYOUT]]>
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32, #blocked>
    tt.return
}
}
