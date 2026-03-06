// RUN: env LINGODB_EXECUTION_MODE=DEFAULT mlir-db-opt %s  -split-input-file -mlir-print-debuginfo -mlir-print-local-scope --lower-graphalg-to-graphalg-core | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope --lower-graphalg-to-graphalg-core | FileCheck %s %}

#dim = #graphalg.dim<distinct[0]<>>


module @"<input>" {
  // CHECK-LABEL: func.func @SSSP(
  // CHECK-SAME:      %[[ARG0:.*]]: !graphalg.mat<distinct[0]<> x distinct[0]<> x !graphalg.trop_f64>,
  // CHECK-SAME:      %[[ARG1:.*]]: !graphalg.mat<distinct[0]<> x 1 x i1>) -> !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64> {
  func.func @SSSP(%arg0: !graphalg.mat<#dim x #dim x !graphalg.trop_f64>, %arg1: !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> {

    // Check that graphalg.cast is lowered to a graphalg.apply block mapping i1 to trop_f64
    // CHECK:           %[[APPLY_CAST:.*]] = graphalg.apply %[[ARG1]] : !graphalg.mat<distinct[0]<> x 1 x i1> -> <distinct[0]<> x 1 x !graphalg.trop_f64> {
    // CHECK-NEXT:      ^bb0(%[[SCALAR_ARG:.*]]: i1):
    // CHECK-NEXT:        %[[SCALAR_CAST:.*]] = graphalg.cast_scalar %[[SCALAR_ARG]] : i1 -> !graphalg.trop_f64
    // CHECK-NEXT:        graphalg.apply.return %[[SCALAR_CAST]] : !graphalg.trop_f64
    // CHECK-NEXT:      }
    %0 = graphalg.cast %arg1 : <#dim x 1 x i1> -> <#dim x 1 x !graphalg.trop_f64>

    // Verify cast_dim is eliminated after conversion
    // CHECK-NOT:       graphalg.cast_dim
    %1 = graphalg.cast_dim #dim

    // Verify the for_dim loop is initialized with the apply result
    // CHECK:           %[[FOR_RES:.*]] = graphalg.for_dim range(distinct[0]<>) init(%[[APPLY_CAST]]) : !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64> -> !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64> body {
    // CHECK-NEXT:      ^bb0(%[[BB_ARG0:.*]]: !graphalg.mat<1 x 1 x i64>, %[[ITER_ARG:.*]]: !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64>):
    %2 = graphalg.for_dim range(#dim) init(%0) : !graphalg.mat<#dim x 1 x !graphalg.trop_f64> -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> body {
    ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<#dim x 1 x !graphalg.trop_f64>):

      // Check that vxm is lowered to transpose and mxm
      // CHECK-NEXT:      %[[TRANSPOSE:.*]] = graphalg.transpose %[[ARG0]] : <distinct[0]<> x distinct[0]<> x !graphalg.trop_f64>
      // CHECK-NEXT:      %[[MXM:.*]] = graphalg.mxm %[[TRANSPOSE]], %[[ITER_ARG]] : <distinct[0]<> x distinct[0]<> x !graphalg.trop_f64>, <distinct[0]<> x 1 x !graphalg.trop_f64>
      %3 = graphalg.vxm %arg3, %arg0 : <#dim x 1 x !graphalg.trop_f64>, <#dim x #dim x !graphalg.trop_f64>

      // Check that ewise ADD is lowered to an element-wise apply block performing add
      // CHECK-NEXT:      %[[APPLY_EWISE:.*]] = graphalg.apply %[[ITER_ARG]], %[[MXM]] : !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64>, !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64> -> <distinct[0]<> x 1 x !graphalg.trop_f64> {
      // CHECK-NEXT:      ^bb0(%[[EWISE_ARG0:.*]]: !graphalg.trop_f64, %[[EWISE_ARG1:.*]]: !graphalg.trop_f64):
      // CHECK-NEXT:        %[[ADD:.*]] = graphalg.add %[[EWISE_ARG0]], %[[EWISE_ARG1]] : !graphalg.trop_f64
      // CHECK-NEXT:        graphalg.apply.return %[[ADD]] : !graphalg.trop_f64
      // CHECK-NEXT:      }
      %4 = graphalg.ewise %arg3 ADD %3 : <#dim x 1 x !graphalg.trop_f64>

      // CHECK-NEXT:      graphalg.yield %[[APPLY_EWISE]] : !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64>
      graphalg.yield %4 : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>
    } until {
    }

    // Ensure the loop result correctly dictates the returned value
    // CHECK:           return %[[FOR_RES]] : !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64>
    return %2 : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>
  }
}