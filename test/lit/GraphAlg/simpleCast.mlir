// RUN: env LINGODB_EXECUTION_MODE=DEFAULT mlir-db-opt %s -split-input-file -mlir-print-local-scope --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE mlir-db-opt %s -split-input-file -mlir-print-local-scope --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg | FileCheck %s %}

#dim = #graphalg.dim<distinct[0]<>>
// CHECK: module @"<input>"
module @"<input>" {
  func.func @SimpleCast(%arg0: !graphalg.mat<#dim x #dim x !graphalg.trop_f64>, %arg1: !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> {
    %0 = graphalg.cast %arg1 : <#dim x 1 x i1> -> <#dim x 1 x !graphalg.trop_f64>
    return %0 : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>
  }
}