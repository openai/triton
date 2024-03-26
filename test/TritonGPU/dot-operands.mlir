// RUN: triton-opt %s -split-input-file -tritongpu-optimize-dot-operands -canonicalize | FileCheck %s

#Cv2 = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#Av2k1 = #triton_gpu.dot_op<{opIdx = 0, parent = #Cv2, kWidth=1}>
#Bv2k1 = #triton_gpu.dot_op<{opIdx = 1, parent = #Cv2, kWidth=1}>
#Av2k2 = #triton_gpu.dot_op<{opIdx = 0, parent = #Cv2, kWidth=2}>
#Bv2k2 = #triton_gpu.dot_op<{opIdx = 1, parent = #Cv2, kWidth=2}>
#Av2k4 = #triton_gpu.dot_op<{opIdx = 0, parent = #Cv2, kWidth=4}>
#Bv2k4 = #triton_gpu.dot_op<{opIdx = 1, parent = #Cv2, kWidth=4}>
#Cv1 = #triton_gpu.nvidia_mma<{versionMajor = 1, warpsPerCTA = [4, 1]}>
#Av1 = #triton_gpu.dot_op<{opIdx = 0, parent = #Cv1}>
#Bv1 = #triton_gpu.dot_op<{opIdx = 1, parent = #Cv1}>
#ALR = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALC = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
#BLR = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLC = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.compute-capability" = 80} {

// CHECK: tt.func @push_elementwise
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[ACVT:.*]] = triton_gpu.convert_layout %[[ALOAD]] {{.*}} #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ACVT]] {{.*}} #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]] {{.*}} #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %[[BCVT:.*]] = triton_gpu.convert_layout %{{.*}} : {{.*}} tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
// CHECK: %[[C:.*]] = tt.dot %[[AF16]], %[[BCVT]]
// CHECK-SAME: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x16xf32, #mma>
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @push_elementwise(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xi8, #ALR>
  %b = tt.load %pb {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BLC>
  %af8 = tt.bitcast %ai8: tensor<16x16xi8, #ALR> -> tensor<16x16xf8E5M2, #ALR>
  %a = tt.fp_to_fp %af8: tensor<16x16xf8E5M2, #ALR> -> tensor<16x16xf16, #ALR>
  %dota = triton_gpu.convert_layout %a : tensor<16x16xf16, #ALR> -> tensor<16x16xf16, #Av2k4>
  %dotb = triton_gpu.convert_layout %b : tensor<16x16xf16, #BLC> -> tensor<16x16xf16, #Bv2k4>
  %newc = tt.dot %dota, %dotb, %c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x16xf16, #Av2k4> * tensor<16x16xf16, #Bv2k4> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}


// CHECK: tt.func @succeeds_if_arg_is_not_convert_layout
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[ACVT:.*]] = triton_gpu.convert_layout %[[ALOAD]]
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ACVT]]
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]]
// CHECK: %[[C:.*]] = tt.dot %[[AF16]]
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @succeeds_if_arg_is_not_convert_layout(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xi8, #ALR>
  %dotai8 = triton_gpu.convert_layout %ai8 : tensor<16x16xi8, #ALR> -> tensor<16x16xi8, #Av2k4>
  %b = tt.load %pb {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BLC>
  %dotaf8 = tt.bitcast %dotai8 : tensor<16x16xi8, #Av2k4> -> tensor<16x16xf8E5M2, #Av2k4>
  %dota = tt.fp_to_fp %dotaf8 : tensor<16x16xf8E5M2, #Av2k4> -> tensor<16x16xf16, #Av2k4>
  %dotb = triton_gpu.convert_layout %b : tensor<16x16xf16, #BLC> -> tensor<16x16xf16, #Bv2k4>
  %newc = tt.dot %dota, %dotb, %c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x16xf16, #Av2k4> * tensor<16x16xf16, #Bv2k4> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}

}

// -----

#blockedA = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #triton_gpu.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.compute-capability" = 80} {

// CHECK: #[[BA:.*]] = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #[[BB:.*]] = #triton_gpu.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: #[[MMA:.*]] = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = []}>

// CHECK: tt.func @push_convert_both_operands
// CHECK: %[[ALOAD:.*]] = tt.load %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #[[BA]]>
// CHECK: %[[BLOAD:.*]] = tt.load %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #[[BB]]>
// CHECK: %[[ACVT:.*]] = triton_gpu.convert_layout %[[ALOAD]] : tensor<16x16xf16, #[[BA]]> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[AEXT:.*]] = arith.extf %[[ACVT]] : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[BCVT:.*]] = triton_gpu.convert_layout %[[BLOAD]] : tensor<16x16xf16, #[[BB]]> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[BEXT:.*]] = arith.extf %[[BCVT]] : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: tt.dot %[[AEXT]], %[[BEXT]], %{{.*}} {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> -> tensor<16x16xf32, #mma>
tt.func @push_convert_both_operands(
                   %pa: tensor<16x16x!tt.ptr<f16>, #blockedA> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #blockedB> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>{
  %a = tt.load %pa {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #blockedA>
  %b = tt.load %pb {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #blockedB>
  %ae = arith.extf %a : tensor<16x16xf16, #blockedA> to tensor<16x16xf32, #blockedA>
  %be = arith.extf %b : tensor<16x16xf16, #blockedB> to tensor<16x16xf32, #blockedB>
  %al = triton_gpu.convert_layout %ae : tensor<16x16xf32, #blockedA> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  %bl = triton_gpu.convert_layout %be : tensor<16x16xf32, #blockedB> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
  %r = tt.dot %al, %bl, %c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
  tt.return %r : tensor<16x16xf32, #mma>
}

}

// -----

#blockedA = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #triton_gpu.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.compute-capability" = 80} {

// CHECK: #[[BA:.*]] = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #[[BB:.*]] = #triton_gpu.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: #[[MMA:.*]] = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = []}>

// CHECK: tt.func @update_kwidth_slice
// CHECK: %[[CST:.+]] = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[ALOAD:.*]] = tt.load %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #[[BA]]>
// CHECK: %[[BLOAD:.*]] = tt.load %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #[[BB]]>
// CHECK: %[[ACVT:.*]] = triton_gpu.convert_layout %[[ALOAD]] : tensor<16x16xf16, #[[BA]]> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[AEXT:.*]] = arith.extf %[[ACVT]] : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[BCVT:.*]] = triton_gpu.convert_layout %[[BLOAD]] : tensor<16x16xf16, #[[BB]]> -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[BEXT:.*]] = arith.extf %[[BCVT]] : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[ADD:.+]] = arith.addf %[[BEXT]], %[[CST]] : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: tt.dot %[[AEXT]], %[[ADD]], %{{.*}} {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> -> tensor<16x16xf32, #mma>
tt.func @update_kwidth_slice(
                   %pa: tensor<16x16x!tt.ptr<f16>, #blockedA> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #blockedB> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>{
  %cst = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #blockedB>
  %a = tt.load %pa {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #blockedA>
  %b = tt.load %pb {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #blockedB>
  %ae = arith.extf %a : tensor<16x16xf16, #blockedA> to tensor<16x16xf32, #blockedA>
  %be = arith.extf %b : tensor<16x16xf16, #blockedB> to tensor<16x16xf32, #blockedB>
  %add = arith.addf %be, %cst : tensor<16x16xf32, #blockedB>
  %al = triton_gpu.convert_layout %ae : tensor<16x16xf32, #blockedA> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  %bl = triton_gpu.convert_layout %add : tensor<16x16xf32, #blockedB> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
  %r = tt.dot %al, %bl, %c {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
  tt.return %r : tensor<16x16xf32, #mma>
}

}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_operand_A
//    CHECK: %[[A:.+]] = triton_gpu.convert_layout %{{.*}} : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
//    CHECK: tt.dot %[[A]], {{.*}} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * !tt.memdesc<64x64xf16, #shared> -> tensor<128x64xf32, #mma>
tt.func @mma_v3_reg_operand_A(%arg0: tensor<128x64xf16, #mma>, %arg1: !tt.memdesc<64x64xf16, #shared>, %arg2: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
  %A = triton_gpu.local_alloc %arg0 : (tensor<128x64xf16, #mma>) -> !tt.memdesc<128x64xf16, #shared1>
  %r = tt.dot %A, %arg1, %arg2 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : !tt.memdesc<128x64xf16, #shared1> * !tt.memdesc<64x64xf16, #shared> -> tensor<128x64xf32, #mma>
  tt.return %r : tensor<128x64xf32, #mma>
}
}

// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_operand_A_fp8
//    CHECK: %[[A:.+]] = triton_gpu.convert_layout %{{.*}} : tensor<128x64xf8E5M2, #mma> -> tensor<128x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
//    CHECK: tt.dot %[[A]], {{.*}} : tensor<128x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * !tt.memdesc<64x64xf8E5M2, #shared> -> tensor<128x64xf32, #mma>
tt.func @mma_v3_reg_operand_A_fp8(%arg0: tensor<128x64xf8E5M2, #mma>, %arg1: !tt.memdesc<64x64xf8E5M2, #shared>, %arg2: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
  %A = triton_gpu.local_alloc %arg0 : (tensor<128x64xf8E5M2, #mma>) -> !tt.memdesc<128x64xf8E5M2, #shared1>
  %r = tt.dot %A, %arg1, %arg2 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : !tt.memdesc<128x64xf8E5M2, #shared1> * !tt.memdesc<64x64xf8E5M2, #shared> -> tensor<128x64xf32, #mma>
  tt.return %r : tensor<128x64xf32, #mma>
}
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @a_impl
// CHECK-NOT: %[[SELECT:.*]] = arith.select {{.*}} : tensor<128x128xi1, #triton_gpu.dot_op<{{.*}}>, tensor<128x128xf16, #triton_gpu.dot_op<{{.*}}>
  tt.func @a_impl(%pa: tensor<128x128x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_3 = arith.constant dense<5> : tensor<128x1xi32, #blocked>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %tl = tt.load %pa {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #blocked>
    %tr = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %te = tt.expand_dims %tr {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %tc = arith.cmpi slt, %te, %cst_3 : tensor<128x1xi32, #blocked>
    %tb = tt.broadcast %tc : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %ts = arith.select %tb, %tl, %cst_4 : tensor<128x128xi1, #blocked>, tensor<128x128xf16, #blocked>
    %conv = triton_gpu.convert_layout %ts : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %td = tt.dot %cst_0, %conv, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    tt.return %td : tensor<128x128xf32, #mma>
  }
}
