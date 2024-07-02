// RUN: triton-opt %s -split-input-file -tritonamdgpu-canonicalize-pointers | FileCheck %s

module {
  // CHECK-LABEL: tt.func @simple
  tt.func @simple(%arg0: !tt.ptr<f32>)-> tensor<1024xf32>{
     %c1024_i32 = arith.constant 1024 : i32
     %0 = tt.get_program_id x : i32
     %1 = arith.muli %0, %c1024_i32 : i32
     %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
     %3 = tt.splat %1 : i32 -> tensor<1024xi32>
     %4 = arith.addi %3, %2 : tensor<1024xi32>
     // CHECK: %[[offset0:.*]] = arith.addi {{.*}}, {{.*}} : tensor<1024xi32>
     // CHECK: %[[zero_offset:.*]] = tt.splat {{.*}} : i32 -> tensor<1024xi32>
     // CHECK: %[[basePtr:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
     // CHECK: %[[newOffset0:.*]] = arith.addi %[[zero_offset]], %[[offset0]]
     // CHECK: %[[newOffset1:.*]] = arith.addi %[[newOffset0]], %[[offset0]]
     // CHECK: %[[newPtr:.*]] = tt.addptr %[[basePtr]], %[[newOffset1]]
     // CHECK: {{.*}} = tt.load %[[newPtr]]
     %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
     %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
     %7 = tt.addptr %6, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
     %8 = tt.load %7 : tensor<1024x!tt.ptr<f32>>
     tt.return %8 : tensor<1024xf32>
  }

  // CHECK-LABEL: tt.func @forOp
  tt.func @forOp(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32>)-> tensor<1024xf32>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // CHECK: %[[offset0:.*]] = arith.addi {{.*}}, {{.*}}
    // CHECK: %[[zero_offset:.*]] = tt.splat {{.*}}
    // CHECK: %[[basePtr:.*]] = tt.splat %arg0
    // CHECK: %[[offset1:.*]] = arith.addi %[[zero_offset]], %[[offset0]]
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // CHECK: {{.*}}:3 = scf.for {{.*}} iter_args(%[[loopPtr:.*]] = %[[basePtr]], {{.*}}, %[[loopOffset:.*]] = %[[offset1]]
    %52:2 = scf.for %arg9 = %c0 to %c128 step %c1 iter_args(%arg1 = %6, %arg2 = %init) -> (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>){
        // CHECK: %[[offset_i:.*]] = arith.addi %[[loopOffset]], %[[offset0]]
        // CHECK: %[[newPtr:.*]] = tt.addptr %[[loopPtr]], %[[offset_i]]
        // CHECK: {{.*}} = tt.load %[[newPtr]]
        // CHECK: scf.yield %[[loopPtr]], {{.*}}, %[[offset_i]]
        %11 = tt.addptr %arg1, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        %9 = tt.load %11 : tensor<1024x!tt.ptr<f32>>
        %10 = arith.addf %9, %arg2 : tensor<1024xf32>
        scf.yield %11, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>
    }
    %8 = tt.addptr %52#0, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %11 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %11 : tensor<1024xf32>
  }

  // CHECK-LABEL  tt.func @forOp2
  tt.func @forOp2(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32>)-> tensor<1024xf32>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // CHECK: %[[splatOffset:.*]] = tt.splat
    // CHECK: %[[offset0:.*]] = arith.addi %[[splatOffset]], {{.*}} : tensor<1024xi32>
    // CHECK: %[[zero_offset:.*]] = tt.splat %c0{{.*}}
    // CHECK: %[[basePtr:.*]] = tt.splat %arg0
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // CHECK: %[[forOut:.*]]:3 = scf.for {{.*}} iter_args(%[[loopPtr:.*]] = %[[basePtr]], {{.*}}, %[[loopOffset:.*]] = %[[zero_offset]]){{.*}}{
    %52:2 = scf.for %arg9 = %c0 to %c128 step %c1 iter_args(%arg1 = %5, %arg2 = %init) -> (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>){
        // CHECK: %[[offset_i:.*]] = arith.addi %[[loopOffset]], %[[offset0]] : tensor<1024xi32>
        // CHECK: %[[newPtr:.*]] = tt.addptr %[[loopPtr]], %[[offset_i]]
        // CHEC: tt.load %[[newPtr]]
        %11 = tt.addptr %arg1, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        %9 = tt.load %11 : tensor<1024x!tt.ptr<f32>>
        %10 = arith.addf %9, %arg2 : tensor<1024xf32>
        scf.yield %11, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>
    }
    // CHECK: %[[tailOffset:.*]] = arith.addi %[[forOut]]#2, %[[offset0]] : tensor<1024xi32>
    // CHECK: %[[tailPtr:.*]] = tt.addptr %[[forOut]]#0, %[[tailOffset]]
    // CHECK: tt.load %[[tailPtr]]
    %8 = tt.addptr %52#0, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %11 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %11 : tensor<1024xf32>
  }

  // CHECK-LABEL  tt.func @forNested
  tt.func @forNested(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32>)-> tensor<1024xf32>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // CHECK: %[[splatOffset:.*]] = tt.splat
    // CHECK: %[[offset0:.*]] = arith.addi %[[splatOffset]], {{.*}} : tensor<1024xi32>
    // CHECK: %[[zero_offset:.*]] = tt.splat %c0{{.*}}
    // CHECK: %[[basePtr:.*]] = tt.splat %arg0
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>

    // CHECK: %[[forOut0:.*]]:3 = scf.for {{.*}} iter_args(%[[loopPtr0:.*]] = %[[basePtr]], {{.*}}, %[[loopOffset0:.*]] = %[[zero_offset]]){{.*}}{
    // CHECK: %[[forOut1:.*]]:3 = scf.for {{.*}} iter_args(%[[loopPtr1:.*]] = %[[loopPtr0]], {{.*}}, %[[loopOffset1:.*]] = %[[loopOffset0]]){{.*}}{
    // CHECK: %[[offset_i:.*]] = arith.addi %[[loopOffset1]], %[[offset0]] : tensor<1024xi32>
    // CHECK: %[[newPtr:.*]] = tt.addptr %[[loopPtr1]], %[[offset_i]]
    // CHECK: tt.load %[[newPtr]]
    // CHECK: scf.yield %[[loopPtr1]], {{.*}}, %[[offset_i]]
    // CHECK: scf.yield %[[forOut1]]#0, {{.*}}, %[[forOut1]]#2

    %52:2 = scf.for %arg9 = %c0 to %c128 step %c1 iter_args(%arg1 = %5, %arg2 = %init) -> (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>){
        %53:2 = scf.for %arg10 = %c0 to %c128 step %c1 iter_args(%arg3 = %arg1, %arg4 = %arg2) -> (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>){
            %11 = tt.addptr %arg3, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
            %9 = tt.load %11 : tensor<1024x!tt.ptr<f32>>
            %10 = arith.addf %9, %arg4 : tensor<1024xf32>
            scf.yield %11, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>
        }
        scf.yield %53#0, %53#1: tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>
    }
    %8 = tt.addptr %52#0, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %11 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %11 : tensor<1024xf32>
  }

  // CHECK-LABEL  tt.func @ifOp
  tt.func @ifOp(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32>, %cond : i1)-> tensor<1024xf32>{
     %c1024_i32 = arith.constant 1024 : i32
     %c0 = arith.constant 0: index
     %c128 = arith.constant 128: index
     %c1 = arith.constant 1 : index
     %0 = tt.get_program_id x : i32
     %1 = arith.muli %0, %c1024_i32 : i32
     %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // CHECK: %[[splatOffset:.*]] = tt.splat
    // CHECK: %[[offset0:.*]] = arith.addi %[[splatOffset]], {{.*}} : tensor<1024xi32>
    // CHECK: %[[zero_offset:.*]] = tt.splat %c0{{.*}}
    // CHECK: %[[basePtr:.*]] = tt.splat %arg0
     %3 = tt.splat %1 : i32 -> tensor<1024xi32>
     %4 = arith.addi %3, %2 : tensor<1024xi32>
     %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
     // CHECK: %[[ifOut:.*]]:2 = scf.if {{.*}} -> (tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) {
     %6 = scf.if %cond -> (tensor<1024x!tt.ptr<f32>>){
         // CHECK: %[[if_offset:.*]] = arith.addi %[[zero_offset]], %[[offset0]]
         // CHECK: scf.yield %[[basePtr]], %[[if_offset]]
         %true = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
         scf.yield %true : tensor<1024x!tt.ptr<f32>>
     } else {
         // CHECK: %[[if_offset:.*]] = arith.addi %[[zero_offset]], %[[splatOffset]]
         // CHECK: scf.yield %[[basePtr]], %[[if_offset]]
         %false = tt.addptr %5, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
         scf.yield %false : tensor<1024x!tt.ptr<f32>>
     }
     // CHECK: %[[newPtr:.*]] = tt.addptr %[[ifOut]]#0, %[[ifOut]]#1
     // CHECK: tt.load %[[newPtr]]
     %11 = tt.load %6 : tensor<1024x!tt.ptr<f32>>
     tt.return %11 : tensor<1024xf32>
  }

  // CHECK-LABEL  tt.func @whileOp
  tt.func @whileOp(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32>, %cond : i1)-> tensor<1024xf32>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // CHECK: %[[splatOffset:.*]] = tt.splat
    // CHECK: %[[offset0:.*]] = arith.addi %[[splatOffset]], {{.*}} : tensor<1024xi32>
    // CHECK: %[[zero_offset:.*]] = tt.splat %c0{{.*}}
    // CHECK: %[[basePtr:.*]] = tt.splat %arg0
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // CHECK: %[[whileOut:.*]]:2 = scf.while (%[[loopPtr:.*]] = %[[basePtr]], {{.*}}, %[[loopOffset:.*]]= %[[zero_offset]])
    %6 = scf.while (%arg1 = %5, %arg2 = %cond) : (tensor<1024x!tt.ptr<f32>>, i1) -> (tensor<1024x!tt.ptr<f32>>) {
        // CHECK: scf.condition({{.*}}) %[[loopPtr]], %[[loopOffset]]
        scf.condition(%arg2) %arg1 : tensor<1024x!tt.ptr<f32>>
        } do {
        // CHECK: ^bb{{.*}}(%[[blockPtr:.*]]: {{.*}}, %[[blockOffset:.*]]: {{.*}}):
        ^bb0(%arg1: tensor<1024x!tt.ptr<f32>>):
        // CHECK: scf.yield %[[blockPtr]], {{.*}}, %[[blockOffset]]
        scf.yield %arg1, %cond : tensor<1024x!tt.ptr<f32>>, i1
        }
    // CHECK: %[[newPtr:.*]] = tt.addptr %[[whileOut]]#0, %[[whileOut]]#1
    // CHECK: tt.load %[[newPtr]]
    %11 = tt.load %6 : tensor<1024x!tt.ptr<f32>>
    tt.return %11 : tensor<1024xf32>
  }

  // CHECK-LABEL  tt.func @condBranch
  tt.func @condBranch(%arg0 : !tt.ptr<f32>, %i1 : i1) -> tensor<1024xf32>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // CHECK: %[[splatOffset:.*]] = tt.splat
    // CHECK: %[[offset0:.*]] = arith.addi %[[splatOffset]], {{.*}} : tensor<1024xi32>
    // CHECK: %[[zero_offset:.*]] = tt.splat %c0{{.*}}
    // CHECK: %[[basePtr:.*]] = tt.splat %arg0
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // CHECK: %[[offset1:.*]] = arith.addi %[[zero_offset]], %[[offset0]] : tensor<1024xi32>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // CHECK: cf.cond_br {{.*}}, ^bb1(%[[basePtr]], %[[zero_offset]] : {{.*}}), ^bb2(%[[basePtr]], %[[offset1]] : {{.*}})
    cf.cond_br %i1, ^bb1(%5 : tensor<1024x!tt.ptr<f32>>), ^bb2(%6 : tensor<1024x!tt.ptr<f32>>)
  // CHECK: ^bb1(%[[block1Ptr:.*]]: {{.*}}, %[[block1Offset:.*]]: {{.*}})
  ^bb1(%arg1 : tensor<1024x!tt.ptr<f32>>):
    // CHECK: %[[newPtr1:.*]] = tt.addptr %[[block1Ptr]], %[[block1Offset]]
    // CHECK: tt.load %[[newPtr1]]
    %out1 = tt.load %arg1 : tensor<1024x!tt.ptr<f32>>
    tt.return %out1 : tensor<1024xf32>
  // CHECK: ^bb2(%[[block2Ptr:.*]]: {{.*}}, %[[block2Offset:.*]]: {{.*}})
  ^bb2(%arg2 : tensor<1024x!tt.ptr<f32>>):  // 2 preds: ^bb0, ^bb1
    // CHECK: %[[newPtr2:.*]] = tt.addptr %[[block2Ptr]], %[[block2Offset]]
    // CHECK: tt.load %[[newPtr2]]
    %out2 = tt.load %arg2 : tensor<1024x!tt.ptr<f32>>
    tt.return %out2 : tensor<1024xf32>
  }

  // CHECK-LABEL  tt.func @branch
  tt.func @branch(%arg0 : !tt.ptr<f32>, %i1 : i1) -> tensor<1024xf32>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // CHECK: %[[splatOffset:.*]] = tt.splat
    // CHECK: %[[offset0:.*]] = arith.addi %[[splatOffset]], {{.*}} : tensor<1024xi32>
    // CHECK: %[[zero_offset:.*]] = tt.splat %c0{{.*}}
    // CHECK: %[[basePtr:.*]] = tt.splat %arg0
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // CHECK: %[[offset1:.*]] = arith.addi %[[zero_offset]], %[[offset0]] : tensor<1024xi32>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // CHECK: cf.br ^bb1(%[[basePtr]], %[[offset1]] : {{.*}})
  // CHECK: ^bb1(%[[block1Ptr:.*]]: {{.*}}, %[[block1Offset:.*]]: {{.*}})
    cf.br ^bb1(%6 : tensor<1024x!tt.ptr<f32>>)
  ^bb1(%arg1 : tensor<1024x!tt.ptr<f32>>):
    // CHECK: %[[newPtr1:.*]] = tt.addptr %[[block1Ptr]], %[[block1Offset]]
    // CHECK: tt.load %[[newPtr1]]
    %out1 = tt.load %arg1 : tensor<1024x!tt.ptr<f32>>
    tt.return %out1 : tensor<1024xf32>
    }
}
