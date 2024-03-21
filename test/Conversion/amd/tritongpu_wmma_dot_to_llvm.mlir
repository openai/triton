// RUN: triton-opt %s --split-input-file --convert-triton-amdgpu-to-llvm | FileCheck %s

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
#mma = #triton_gpu.amd_wmma<{warpsPerCTA = [2, 2]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  //  CHECK-LABEL: wmma_dot_operand
  tt.func @wmma_dot_operand(%arg0: !tt.memdesc<64x64xf16, #shared>) {
    // 2 CTA * 4 rep * load_per_thread_per_instr
    // CHECK-COUNT-8: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<16xf16>
    %0 = triton_gpu.local_load %arg0 : !tt.memdesc<64x64xf16, #shared> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
    // CHECK-COUNT-128: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %1 = triton_gpu.local_load %arg0 : !tt.memdesc<64x64xf16, #shared> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
    tt.return
  }

  //  CHECK-LABEL: wmma_dot
  tt.func @wmma_dot(%arg0: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>, %arg1: tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>, %arg2: tensor<16x16xf16, #mma>) {
    // CHECK-COUNT-32: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.extractvalue %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK: llvm.mlir.undef : vector<16xf16>
    // CHECK-COUNT-8: llvm.insertelement {{.*}} : vector<16xf16>
    // CHECK: rocdl.wmma.f16.16x16x16.f16 {{.*}} : (vector<16xf16>, vector<16xf16>, vector<16xf16>, i1) -> vector<16xf16>
    %0 = tt.dot %arg0, %arg1, %arg2 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<16x16xf16, #mma>
    // CHECK-COUNT-8: llvm.extractelement {{.*}} : vector<16xf16>
    // CHECK: llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-COUNT-8: llvm.insertvalue {{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    tt.return
  }
}
