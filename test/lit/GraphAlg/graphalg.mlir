// RUN: env LINGODB_EXECUTION_MODE=DEFAULT mlir-db-opt %s -split-input-file -mlir-print-local-scope --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE mlir-db-opt %s -split-input-file -mlir-print-local-scope --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg | FileCheck %s %}

#dim = #graphalg.dim<distinct[0]<>>
// CHECK: module @"<input>"
module @"<input>" {
  func.func @SSSP(%arg0: !graphalg.mat<#dim x #dim x !graphalg.trop_f64>, %arg1: !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> {
    %0 = graphalg.cast %arg1 : <#dim x 1 x i1> -> <#dim x 1 x !graphalg.trop_f64>
    %1 = graphalg.cast_dim #dim
    %2 = graphalg.for_dim range(#dim) init(%0) : !graphalg.mat<#dim x 1 x !graphalg.trop_f64> -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> body {
    ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<#dim x 1 x !graphalg.trop_f64>):
      %3 = graphalg.vxm %arg3, %arg0 : <#dim x 1 x !graphalg.trop_f64>, <#dim x #dim x !graphalg.trop_f64>
      %4 = graphalg.ewise %arg3 ADD %3 : <#dim x 1 x !graphalg.trop_f64>
      graphalg.yield %4 : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>
    } until {
    }
    return %2 : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>
  }
}