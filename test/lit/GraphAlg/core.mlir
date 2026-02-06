// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
// RUN: %if baseline-backend %{LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s %}

#dim = #graphalg.dim<distinct[0]<>>
module @"<input>" {
  func.func @SSSP(%arg0: !graphalg.mat<#dim x #dim x !graphalg.trop_f64>, %arg1: !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> {
    %0 = graphalg.apply %arg1 : !graphalg.mat<#dim x 1 x i1> -> <#dim x 1 x !graphalg.trop_f64> {
    ^bb0(%arg2: i1):
      %2 = graphalg.cast_scalar %arg2 : i1 -> !graphalg.trop_f64
      graphalg.apply.return %2 : !graphalg.trop_f64
    }
    %1 = graphalg.for_dim range(#dim) init(%0) : !graphalg.mat<#dim x 1 x !graphalg.trop_f64> -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> body {
    ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<#dim x 1 x !graphalg.trop_f64>):
      %2 = graphalg.transpose %arg0 : <#dim x #dim x !graphalg.trop_f64>
      %3 = graphalg.mxm %2, %arg3 : <#dim x #dim x !graphalg.trop_f64>, <#dim x 1 x !graphalg.trop_f64>
      %4 = graphalg.apply %arg3, %3 : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>, !graphalg.mat<#dim x 1 x !graphalg.trop_f64> -> <#dim x 1 x !graphalg.trop_f64> {
      ^bb0(%arg4: !graphalg.trop_f64, %arg5: !graphalg.trop_f64):
        %5 = graphalg.add %arg4, %arg5 : !graphalg.trop_f64
        graphalg.apply.return %5 : !graphalg.trop_f64
      }
      graphalg.yield %4 : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>
    } until {
    }
    return %1 : !graphalg.mat<#dim x 1 x !graphalg.trop_f64>
  }
}
