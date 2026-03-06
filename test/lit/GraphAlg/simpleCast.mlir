// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

#dim = #graphalg.dim<distinct[0]<>>
module @"<input>" {
  func.func @SimpleCast(%arg0: !graphalg.mat<#dim x #dim x !graphalg.trop_f64>, %arg1: !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> {
    %0 = graphalg.cast %arg1 : <#dim x 1 x i1> -> <#dim x 1 x !graphalg.trop_f64>
    return %0 : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>
  }
}