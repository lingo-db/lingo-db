// RUN: mlir-db-opt %s --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg > %t.mlir
// RUN: run-mlir %t.mlir | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

module {
  func.func private @withDamping(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<1 x 1 x f64> {
    %0 = graphalg.cast %arg0 : <1 x 1 x i64> -> <1 x 1 x f64>
    %1 = graphalg.ewise %0 DIV %arg1 : <1 x 1 x f64>
    return %1 : !graphalg.mat<1 x 1 x f64>
  }

  func.func private @PR(%arg0: !graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x 1 x f64> {
    %0 = graphalg.literal 8.500000e-01 : f64
    %1 = graphalg.literal 10 : i64
    %2 = graphalg.cast_dim #dim
    %3 = graphalg.literal 1.000000e+00 : f64
    %4 = graphalg.ewise %3 SUB %0 : <1 x 1 x f64>
    %5 = graphalg.cast %2 : <1 x 1 x i64> -> <1 x 1 x f64>
    %6 = graphalg.ewise %4 DIV %5 : <1 x 1 x f64>
    %7 = graphalg.literal 1.000000e+00 : f64
    %8 = graphalg.cast %arg0 : <#dim x #dim x i1> -> <#dim x #dim x i64>
    %9 = graphalg.reduce %8 : <#dim x #dim x i64> -> <#dim x 1 x i64>
    %10 = graphalg.apply_binary @withDamping %9, %0 : (!graphalg.mat<#dim x 1 x i64>, !graphalg.mat<1 x 1 x f64>) -> !graphalg.mat<#dim x 1 x f64>
    %11 = graphalg.reduce %arg0 : <#dim x #dim x i1> -> <#dim x 1 x i1>
    %12 = graphalg.const_mat false -> <#dim x 1 x i1>
    %13 = graphalg.literal true
    %14 = graphalg.broadcast %13 : <1 x 1 x i1> -> <#dim x 1 x i1>
    %15 = graphalg.mask %12<%11 : <#dim x 1 x i1>> = %14 : <#dim x 1 x i1> {complement = true}
    %16 = graphalg.const_mat 0.000000e+00 : f64 -> <#dim x 1 x f64>
    %17 = graphalg.literal 1.000000e+00 : f64
    %18 = graphalg.cast %2 : <1 x 1 x i64> -> <1 x 1 x f64>
    %19 = graphalg.ewise %17 DIV %18 : <1 x 1 x f64>
    %20 = graphalg.broadcast %19 : <1 x 1 x f64> -> <#dim x 1 x f64>
    %21 = graphalg.literal 0 : i64
    %22 = graphalg.for_const range(%21, %1) : <1 x 1 x i64> init(%20) : !graphalg.mat<#dim x 1 x f64> -> !graphalg.mat<#dim x 1 x f64> body {
    ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<#dim x 1 x f64>):
      %23 = graphalg.const_mat 0.000000e+00 : f64 -> <#dim x 1 x f64>
      %24 = graphalg.mask %23<%15 : <#dim x 1 x i1>> = %arg2 : <#dim x 1 x f64> {complement = false}
      %25 = graphalg.cast %2 : <1 x 1 x i64> -> <1 x 1 x f64>
      %26 = graphalg.ewise %0 DIV %25 : <1 x 1 x f64>
      %27 = graphalg.reduce %24 : <#dim x 1 x f64> -> <1 x 1 x f64>
      %28 = graphalg.mxm %26, %27 : <1 x 1 x f64>, <1 x 1 x f64>
      %29 = graphalg.ewise %arg2 DIV %10 : <#dim x 1 x f64>
      %30 = graphalg.ewise %6 ADD %28 : <1 x 1 x f64>
      %31 = graphalg.broadcast %30 : <1 x 1 x f64> -> <#dim x 1 x f64>
      %32 = graphalg.cast %arg0 : <#dim x #dim x i1> -> <#dim x #dim x f64>
      %33 = graphalg.transpose %32 : <#dim x #dim x f64>
      %34 = graphalg.mxm %33, %29 : <#dim x #dim x f64>, <#dim x 1 x f64>
      %35 = graphalg.ewise %31 ADD %34 : <#dim x 1 x f64>
      graphalg.yield %35 : !graphalg.mat<#dim x 1 x f64>
    } until {
    }
    return %22 : !graphalg.mat<#dim x 1 x f64>
  }

  func.func @main() {
    %result = relalg.query() {
      // 5 Nodes: 0, 1, 2, 3, 4
      %edges_rel = relalg.const_relation columns:[@edges::@src({type=i64}), @edges::@dst({type=i64}), @edges::@val({type=i1})] values: [[0: i64, 1: i64, true],[0: i64, 2: i64, true],
        [1: i64, 2: i64, true],[2: i64, 0: i64, true],[2: i64, 3: i64, true],[3: i64, 4: i64, true],
        [4: i64, 0: i64, true]
      ]
      %graph = builtin.unrealized_conversion_cast %edges_rel : !tuples.tuplestream to !graphalg.mat<#dim x #dim x i1> { cols =[@edges::@src, @edges::@dst, @edges::@val] }

      %pr = func.call @PR(%graph) : (!graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x 1 x f64>

      %result_rel = builtin.unrealized_conversion_cast %pr : !graphalg.mat<#dim x 1 x f64> to !tuples.tuplestream { cols =[@pr::@node, @pr::@val] }
      %materialized = relalg.materialize %result_rel[@pr::@node, @pr::@val] => ["node", "pr"] : !subop.local_table<[node: i64, pr: f64], ["node", "pr"]>

      relalg.query_return %materialized : !subop.local_table<[node: i64, pr: f64],["node", "pr"]>
    } -> !subop.local_table<[node: i64, pr: f64], ["node", "pr"]>

    subop.set_result 0 %result : !subop.local_table<[node: i64, pr: f64],["node", "pr"]>

    // CHECK: |{{.*}}node{{.*}}|{{.*}}pr{{.*}}|
    // CHECK-NEXT: ---
    // CHECK-DAG: |{{ +}}0{{ +}}|{{.*}}0.2770{{.*}}|
    // CHECK-DAG: |{{ +}}1{{ +}}|{{.*}}0.1484{{.*}}|
    // CHECK-DAG: |{{ +}}2{{ +}}|{{.*}}0.2733{{.*}}|
    // CHECK-DAG: |{{ +}}3{{ +}}|{{.*}}0.1460{{.*}}|
    // CHECK-DAG: |{{ +}}4{{ +}}|{{.*}}0.1550{{.*}}|

    return
  }
}