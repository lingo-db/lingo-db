// XFAIL: *
// RUN: env LINGODB_EXECUTION_MODE=DEFAULT mlir-db-opt %s -split-input-file -mlir-print-local-scope --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE mlir-db-opt %s -split-input-file -mlir-print-local-scope --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg | FileCheck %s %}

// CHECK: module @"<input>"
module @"<input>" {
  func.func @SSSP(%arg0: !graphalg.mat<distinct[0]<> x distinct[0]<> x !graphalg.trop_f64>, %arg1: !graphalg.mat<distinct[0]<> x 1 x i1>) -> !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64> {
    %0 = graphalg.apply %arg1 : !graphalg.mat<distinct[0]<> x 1 x i1> -> <distinct[0]<> x 1 x !graphalg.trop_f64> {
    ^bb0(%arg2: i1):
      %2 = graphalg.cast_scalar %arg2 : i1 -> !graphalg.trop_f64
      graphalg.apply.return %2 : !graphalg.trop_f64
    }
    %1 = graphalg.for_dim range(distinct[0]<>) init(%0) : !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64> -> !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64> body {
    ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64>):
      %2 = graphalg.transpose %arg0 : <distinct[0]<> x distinct[0]<> x !graphalg.trop_f64>
      %3 = graphalg.mxm %2, %arg3 : <distinct[0]<> x distinct[0]<> x !graphalg.trop_f64>, <distinct[0]<> x 1 x !graphalg.trop_f64>
      %4 = graphalg.apply %arg3, %3 : !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64>, !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64> -> <distinct[0]<> x 1 x !graphalg.trop_f64> {
      ^bb0(%arg4: !graphalg.trop_f64, %arg5: !graphalg.trop_f64):
        %5 = graphalg.add %arg4, %arg5 : !graphalg.trop_f64
        graphalg.apply.return %5 : !graphalg.trop_f64
      }
      graphalg.yield %4 : !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64>
    } until {
    }
    return %1 : !graphalg.mat<distinct[0]<> x 1 x !graphalg.trop_f64>
  }
}