//RUN: mlir-db-opt --torch-backend-to-linalg-on-tensors-backend-pipeline --canonicalize --inline --scf-bufferize --linalg-bufferize --refback-munge-memref-copy --func-bufferize --arith-bufferize --tensor-bufferize -finalizing-bufferize --refback-insert-rng-globals --convert-linalg-to-affine-loops --affine-loop-fusion --affine-loop-unroll="unroll-full unroll-num-reps=3" --affine-scalrep --canonicalize --lower-affine --canonicalize  --simplify-memrefs --db-simplify-to-arith --simplify-arithmetics --canonicalize  -symbol-privatize="exclude=main" --symbol-dce %s | FileCheck %s

// CHECK-DAG:    %[[C1:.*]] = arith.constant 2.14476395 : f32

// CHECK-DAG:    %[[A:.*]] = relalg.getcol %arg0 @R::@a : f32
// CHECK-DAG:    %[[CMP:.*]] = arith.cmpf olt, %[[A]], %[[C1]] : f32
module {
  func.func @forward(%arg0: !torch.vtensor<[1,1],f32>) -> !torch.vtensor<[1,1],f32> {
    %0 = torch.vtensor.literal(dense<0.581083298> : tensor<1xf32>) : !torch.vtensor<[1],f32>
    %1 = torch.vtensor.literal(dense<2.06032777> : tensor<1x1xf32>) : !torch.vtensor<[1,1],f32>
    %2 = torch.aten.linear %arg0, %1, %0 : !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1],f32> -> !torch.vtensor<[1,1],f32>
    return %2 : !torch.vtensor<[1,1],f32>
  }

  func.func @user(%val :f32) -> f32 {
    %as_tensor=tensor.from_elements %val : tensor<1x1xf32>
    %v_tensor = torch_c.from_builtin_tensor %as_tensor : tensor<1x1xf32> -> !torch.vtensor<[1,1],f32>
    %res_v_tensor = func.call @forward(%v_tensor) : (!torch.vtensor<[1,1],f32>) -> !torch.vtensor<[1,1],f32>
    %res_tensor = torch_c.to_builtin_tensor %res_v_tensor : !torch.vtensor<[1,1],f32> -> tensor<1x1xf32>
    %0 = arith.constant 0 : index
    %res = tensor.extract %res_tensor[%0,%0] : tensor<1x1xf32>
    return %res : f32
    }

  func.func @main () -> !dsa.table attributes { torch.ignore = unit }  {
    %1 = relalg.basetable {table_identifier="R"} columns: {a=>@R::@a({type=f32})}
    %3 = relalg.selection %1 (%4: !relalg.tuple) {
      %5 = relalg.getcol %4 @R::@a : f32
      %7 = func.call @user(%5) : (f32) -> f32
      %8 = db.constant (5.0) : f32
      %9 = db.compare lt %7 : f32 , %8 : f32
      relalg.return %9 : i1
    }
    %8 = relalg.materialize %3 [@S::@a ] => ["a"] : !dsa.table
    return %8 : !dsa.table
  }
}