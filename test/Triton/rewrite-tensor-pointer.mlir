// RUN: triton-opt %s -triton-rewrite-tensor-pointer | FileCheck %s
tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
  %c31_i32 = arith.constant 31 : i32
  %c127_i32 = arith.constant 127 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %1 = tt.get_program_id y : i32
  %2 = arith.addi %arg3, %c127_i32 : i32
  %3 = arith.divsi %2, %c128_i32 : i32
  %4 = arith.addi %arg4, %c31_i32 : i32
  %5 = arith.divsi %4, %c32_i32 : i32
  %6 = arith.muli %5, %c8_i32 : i32
  %7 = arith.divsi %0, %6 : i32
  %8 = arith.muli %7, %c8_i32 : i32
  %9 = arith.subi %3, %8 : i32
  %10 = arith.cmpi slt, %9, %c8_i32 : i32
  %11 = arith.select %10, %9, %c8_i32 : i32
  %12 = arith.remsi %0, %11 : i32
  %13 = arith.addi %8, %12 : i32
  %14 = arith.remsi %0, %6 : i32
  %15 = arith.divsi %14, %11 : i32
  %16 = arith.muli %13, %c128_i32 : i32
  %17 = arith.muli %1, %c32_i32 : i32
  %18 = arith.extsi %arg3 : i32 to i64
  %19 = arith.extsi %arg5 : i32 to i64
  %20 = arith.extsi %arg6 : i32 to i64
  // CHECK-NOT: tt.make_tensor_ptr
  %21 = tt.make_tensor_ptr %arg0, [%18, %19], [%20, %c1_i64], [%16, %17] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
  %22 = arith.muli %15, %c32_i32 : i32
  %23 = arith.extsi %arg4 : i32 to i64
  %24 = arith.extsi %arg7 : i32 to i64
  // CHECK-NOT: tt.make_tensor_ptr
  %25 = tt.make_tensor_ptr %arg1, [%19, %23], [%24, %c1_i64], [%17, %22] {order = array<i32: 1, 0>} : !tt.ptr<tensor<32x32xf16>>
  %26 = arith.addi %arg5, %c31_i32 : i32
  %27 = arith.divsi %26, %c32_i32 : i32
  %28 = arith.index_cast %27 : i32 to index
  %29:3 = scf.for %arg9 = %c0 to %28 step %c1 iter_args(%arg10 = %cst, %arg11 = %21, %arg12 = %25) -> (tensor<128x32xf32>, !tt.ptr<tensor<128x32xf16>>, !tt.ptr<tensor<32x32xf16>>) {
    // CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16>
    %55 = tt.load %arg11 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 2 : i32} : !tt.ptr<tensor<128x32xf16>> -> tensor<128x32xf16>
    // CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf16>
    %56 = tt.load %arg12 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 2 : i32} : !tt.ptr<tensor<32x32xf16>> -> tensor<32x32xf16>
    %57 = tt.dot %55, %56, %arg10 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16> * tensor<32x32xf16> -> tensor<128x32xf32>
    // CHECK-NOT: tt.advance
    %58 = tt.advance %arg11, [%c0_i32, %c32_i32] : !tt.ptr<tensor<128x32xf16>>
    // CHECK-NOT: tt.advance
    %59 = tt.advance %arg12, [%c32_i32, %c0_i32] : !tt.ptr<tensor<32x32xf16>>
    scf.yield %57, %58, %59 : tensor<128x32xf32>, !tt.ptr<tensor<128x32xf16>>, !tt.ptr<tensor<32x32xf16>>
  }
  %30 = arith.truncf %29#0 : tensor<128x32xf32> to tensor<128x32xf16>
  %31 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %32 = tt.splat %16 : i32 -> tensor<128xi32>
  %33 = arith.addi %32, %31 : tensor<128xi32>
  %34 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %35 = tt.splat %22 : i32 -> tensor<32xi32>
  %36 = arith.addi %35, %34 : tensor<32xi32>
  %37 = tt.expand_dims %33 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %38 = tt.splat %arg8 : i32 -> tensor<128x1xi32>
  %39 = arith.muli %37, %38 : tensor<128x1xi32>
  %40 = tt.expand_dims %36 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %41 = tt.broadcast %39 : tensor<128x1xi32> -> tensor<128x32xi32>
  %42 = tt.broadcast %40 : tensor<1x32xi32> -> tensor<128x32xi32>
  %43 = arith.addi %41, %42 : tensor<128x32xi32>
  %44 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
  %45 = tt.addptr %44, %43 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi32>
  %46 = tt.splat %arg3 : i32 -> tensor<128xi32>
  %47 = arith.cmpi slt, %33, %46 : tensor<128xi32>
  %48 = tt.expand_dims %47 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
  %49 = tt.splat %arg4 : i32 -> tensor<32xi32>
  %50 = arith.cmpi slt, %36, %49 : tensor<32xi32>
  %51 = tt.expand_dims %50 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %52 = tt.broadcast %48 : tensor<128x1xi1> -> tensor<128x32xi1>
  %53 = tt.broadcast %51 : tensor<1x32xi1> -> tensor<128x32xi1>
  %54 = arith.andi %52, %53 : tensor<128x32xi1>
  tt.store %45, %30, %54 {cache = 1 : i32, evict = 1 : i32} : tensor<128x32xf16>
  tt.return
}
