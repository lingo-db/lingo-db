// RUN: mlir-db-opt %s --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg > %t.mlir
// RUN: run-mlir %t.mlir | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

module {
  func.func private @isMax(%arg0: !graphalg.mat<1 x 1 x i64>, %arg1: !graphalg.mat<1 x 1 x !graphalg.trop_max_i64>) -> !graphalg.mat<1 x 1 x i1> {
    %0 = graphalg.cast %arg0 : <1 x 1 x i64> -> <1 x 1 x !graphalg.trop_max_i64>
    %1 = graphalg.ewise %0 EQ %arg1 : <1 x 1 x !graphalg.trop_max_i64>
    %2 = graphalg.literal 0 : i64
    %3 = graphalg.ewise %arg0 NE %2 : <1 x 1 x i64>
    %4 = graphalg.mxm %1, %3 : <1 x 1 x i1>, <1 x 1 x i1>
    return %4 : !graphalg.mat<1 x 1 x i1>
  }

  // CDLP Algorithm (Community Detection Label Propagation)
  func.func private @CDLP(%arg0: !graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x #dim x i1> {
    %0 = graphalg.literal 5 : i64
    %1 = graphalg.cast_dim #dim
    %2 = graphalg.const_mat false -> <#dim x 1 x i1>
    %3 = graphalg.literal true
    %4 = graphalg.broadcast %3 : <1 x 1 x i1> -> <#dim x 1 x i1>
    %5 = graphalg.diag %4 : !graphalg.mat<#dim x 1 x i1>
    %6 = graphalg.literal 0 : i64
    %7 = graphalg.for_const range(%6, %0) : <1 x 1 x i64> init(%5) : !graphalg.mat<#dim x #dim x i1> -> !graphalg.mat<#dim x #dim x i1> body {
    ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<#dim x #dim x i1>):
      %19 = graphalg.cast %arg0 : <#dim x #dim x i1> -> <#dim x #dim x i64>
      %20 = graphalg.cast %arg2 : <#dim x #dim x i1> -> <#dim x #dim x i64>
      %21 = graphalg.mxm %19, %20 : <#dim x #dim x i64>, <#dim x #dim x i64>
      %22 = graphalg.transpose %arg0 : <#dim x #dim x i1>
      %23 = graphalg.cast %22 : <#dim x #dim x i1> -> <#dim x #dim x i64>
      %24 = graphalg.cast %arg2 : <#dim x #dim x i1> -> <#dim x #dim x i64>
      %25 = graphalg.mxm %23, %24 : <#dim x #dim x i64>, <#dim x #dim x i64>
      %26 = graphalg.ewise %21 ADD %25 : <#dim x #dim x i64>
      %27 = graphalg.cast %26 : <#dim x #dim x i64> -> <#dim x #dim x !graphalg.trop_max_i64>
      %28 = graphalg.reduce %27 : <#dim x #dim x !graphalg.trop_max_i64> -> <#dim x 1 x !graphalg.trop_max_i64>
      %29 = graphalg.cast_dim #dim
      %30 = graphalg.const_mat #graphalg.trop_inf : !graphalg.trop_max_i64 -> <#dim x 1 x !graphalg.trop_max_i64>
      %31 = graphalg.literal #graphalg.trop_int<0 : i64> : !graphalg.trop_max_i64
      %32 = graphalg.broadcast %31 : <1 x 1 x !graphalg.trop_max_i64> -> <#dim x 1 x !graphalg.trop_max_i64>
      %33 = graphalg.transpose %32 : <#dim x 1 x !graphalg.trop_max_i64>
      %34 = graphalg.mxm %28, %33 : <#dim x 1 x !graphalg.trop_max_i64>, <1 x #dim x !graphalg.trop_max_i64>
      %35 = graphalg.apply_elementwise @isMax %26, %34 : (!graphalg.mat<#dim x #dim x i64>, !graphalg.mat<#dim x #dim x !graphalg.trop_max_i64>) -> !graphalg.mat<#dim x #dim x i1>
      %36 = graphalg.pick_any %35 : <#dim x #dim x i1>
      graphalg.yield %36 : !graphalg.mat<#dim x #dim x i1>
    } until {
    }
    %8 = graphalg.reduce %arg0 : <#dim x #dim x i1> -> <#dim x 1 x i1>
    %9 = graphalg.transpose %arg0 : <#dim x #dim x i1>
    %10 = graphalg.reduce %9 : <#dim x #dim x i1> -> <#dim x 1 x i1>
    %11 = graphalg.ewise %8 ADD %10 : <#dim x 1 x i1>
    %12 = graphalg.cast_dim #dim
    %13 = graphalg.const_mat false -> <#dim x 1 x i1>
    %14 = graphalg.literal true
    %15 = graphalg.broadcast %14 : <1 x 1 x i1> -> <#dim x 1 x i1>
    %16 = graphalg.mask %13<%11 : <#dim x 1 x i1>> = %15 : <#dim x 1 x i1> {complement = true}
    %17 = graphalg.diag %16 : !graphalg.mat<#dim x 1 x i1>
    %18 = graphalg.ewise %17 ADD %7 : <#dim x #dim x i1>
    return %18 : !graphalg.mat<#dim x #dim x i1>
  }

  func.func @main() {
    %result = relalg.query() {

      // 1. Setup the Graph Relation

      %edges_rel = relalg.const_relation columns:[@edges::@src({type=i64}), @edges::@dst({type=i64}), @edges::@val({type=i1})] values: [[0: i64, 1: i64, true],
        [0: i64, 2: i64, true],
        [0: i64, 6: i64, true],[1: i64, 0: i64, true],[1: i64, 2: i64, true],
        [2: i64, 0: i64, true],
        [2: i64, 1: i64, true],[3: i64, 4: i64, true],[3: i64, 5: i64, true],
        [4: i64, 3: i64, true],
        [4: i64, 5: i64, true],[4: i64, 6: i64, true],[5: i64, 4: i64, true],
        [5: i64, 6: i64, true],
        [6: i64, 4: i64, true],[6: i64, 5: i64, true],[6: i64, 7: i64, true],
        [7: i64, 5: i64, true]
      ]

      %graph = builtin.unrealized_conversion_cast %edges_rel : !tuples.tuplestream to !graphalg.mat<#dim x #dim x i1> { cols =[@edges::@src, @edges::@dst, @edges::@val] }

      // 2. Run CDLP Algorithm

      %cdlp_res = func.call @CDLP(%graph) : (!graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x #dim x i1>

      // 3. Materialize

      %result_rel = builtin.unrealized_conversion_cast %cdlp_res : !graphalg.mat<#dim x #dim x i1> to !tuples.tuplestream { cols =[@edges::@src, @edges::@dst, @edges::@val] }
      %sorted =  relalg.sort %result_rel [(@edges::@src,asc),(@edges::@dst,asc)]
      %materialized = relalg.materialize %sorted [@edges::@src, @edges::@dst, @edges::@val] =>["node", "label", "val"] : !subop.local_table<[node: i64, label: i64, val: i1], ["node", "label", "val"]>
      relalg.query_return %materialized : !subop.local_table<[node: i64, label: i64, val: i1], ["node", "label", "val"]>

    } -> !subop.local_table<[node: i64, label: i64, val: i1],["node", "label", "val"]>

    // Output the result table to stdout!
    subop.set_result 0 %result : !subop.local_table<[node: i64, label: i64, val: i1], ["node", "label", "val"]>

    // Check the nodes and their respective detected community labels!
    // CHECK-LABEL: |                          node  |                         label  |                           val  |
    // CHECK-NEXT: ----
    // CHECK-DAG: | 0 | 0 | true |
    // CHECK-DAG: | 1 | 0 | true |
    // CHECK-DAG: | 2 | 0 | true |
    // CHECK-DAG: | 3 | 4 | true |
    // CHECK-DAG: | 4 | 3 | true |
    // CHECK-DAG: | 5 | 3 | true |
    // CHECK-DAG: | 6 | 3 | true |
    // CHECK-DAG: | 7 | 3 | true |
    return
  }
}