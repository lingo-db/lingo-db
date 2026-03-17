#dim = #graphalg.dim<distinct[0]<>>
module {
  func.func private @setDepth(%arg0: !graphalg.mat<1 x 1 x i1>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
    %0 = graphalg.cast %arg0 : <1 x 1 x i1> -> <1 x 1 x i64>
    %1 = graphalg.literal 2 : i64
    %2 = graphalg.ewise %arg1 ADD %1 : <1 x 1 x i64>
    %3 = graphalg.mxm %0, %2 : <1 x 1 x i64>, <1 x 1 x i64>
    return %3 : !graphalg.mat<1 x 1 x i64>
  }
  func.func private @BFS(%arg0: !graphalg.mat<#dim x #dim x i1>, %arg1: !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x i64> {
    %0 = graphalg.cast_dim #dim
    %1 = graphalg.const_mat 0 : i64 -> <#dim x 1 x i64>
    %2 = graphalg.literal 1 : i64
    %3 = graphalg.broadcast %2 : <1 x 1 x i64> -> <#dim x 1 x i64>
    %4 = graphalg.mask %1<%arg1 : <#dim x 1 x i1>> = %3 : <#dim x 1 x i64> {complement = false}
    %5 = graphalg.cast_dim #dim
    %6:3 = graphalg.for_dim range(#dim) init(%4, %arg1, %arg1) : !graphalg.mat<#dim x 1 x i64>, !graphalg.mat<#dim x 1 x i1>, !graphalg.mat<#dim x 1 x i1> -> !graphalg.mat<#dim x 1 x i64>, !graphalg.mat<#dim x 1 x i1>, !graphalg.mat<#dim x 1 x i1> body {
    ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<#dim x 1 x i64>, %arg4: !graphalg.mat<#dim x 1 x i1>, %arg5: !graphalg.mat<#dim x 1 x i1>):
      %7 = graphalg.cast_dim #dim
      %8 = graphalg.const_mat false -> <#dim x 1 x i1>
      %9 = graphalg.vxm %arg4, %arg0 : <#dim x 1 x i1>, <#dim x #dim x i1>
      %10 = graphalg.mask %8<%arg5 : <#dim x 1 x i1>> = %9 : <#dim x 1 x i1> {complement = true}
      %11 = graphalg.apply_binary @setDepth %10, %arg2 : (!graphalg.mat<#dim x 1 x i1>, !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<#dim x 1 x i64>
      %12 = graphalg.ewise %arg3 ADD %11 : <#dim x 1 x i64>
      %13 = graphalg.ewise %arg5 ADD %10 : <#dim x 1 x i1>
      graphalg.yield %12, %10, %13 : !graphalg.mat<#dim x 1 x i64>, !graphalg.mat<#dim x 1 x i1>, !graphalg.mat<#dim x 1 x i1>
    } until {
    ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<#dim x 1 x i64>, %arg4: !graphalg.mat<#dim x 1 x i1>, %arg5: !graphalg.mat<#dim x 1 x i1>):
      %7 = graphalg.nvals %arg4 : <#dim x 1 x i1>
      %8 = graphalg.literal 0 : i64
      %9 = graphalg.ewise %7 EQ %8 : <1 x 1 x i64>
      graphalg.yield %9 : !graphalg.mat<1 x 1 x i1>
    }
    return %6#0 : !graphalg.mat<#dim x 1 x i64>
  }
}