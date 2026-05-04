// RUN: mlir-db-opt %s --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg > %t.mlir
// RUN: run-mlir %t.mlir | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

module {
  // WCC Algorithm (Weakly Connected Components)
  func.func private @WCC(%arg0: !graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x #dim x i1> {
    %0 = graphalg.cast_dim #dim
    %1 = graphalg.const_mat false -> <#dim x 1 x i1>
    %2 = graphalg.literal true
    %3 = graphalg.broadcast %2 : <1 x 1 x i1> -> <#dim x 1 x i1>
    %4 = graphalg.diag %3 : !graphalg.mat<#dim x 1 x i1>
    %5 = graphalg.cast_dim #dim
    %6 = graphalg.for_dim range(#dim) init(%4) : !graphalg.mat<#dim x #dim x i1> -> !graphalg.mat<#dim x #dim x i1> body {
    ^bb0(%arg1: !graphalg.mat<1 x 1 x i64>, %arg2: !graphalg.mat<#dim x #dim x i1>):
      %7 = graphalg.mxm %arg0, %arg2 : <#dim x #dim x i1>, <#dim x #dim x i1>
      %8 = graphalg.ewise %arg2 ADD %7 : <#dim x #dim x i1>
      %9 = graphalg.transpose %arg0 : <#dim x #dim x i1>
      %10 = graphalg.mxm %9, %arg2 : <#dim x #dim x i1>, <#dim x #dim x i1>
      %11 = graphalg.ewise %8 ADD %10 : <#dim x #dim x i1>
      %12 = graphalg.pick_any %11 : <#dim x #dim x i1>
      graphalg.yield %12 : !graphalg.mat<#dim x #dim x i1>
    } until {
    }
    return %6 : !graphalg.mat<#dim x #dim x i1>
  }

  func.func @main() {
    %result = relalg.query() {

      // 1. Setup the Graph Relation
      %edges_rel = relalg.const_relation columns:[@edges::@src({type=i64}), @edges::@dst({type=i64}), @edges::@val({type=i1})] values:[
        [0: i64, 1: i64, true],[0: i64, 2: i64, true],[1: i64, 0: i64, true],
        [1: i64, 2: i64, true],
        [1: i64, 3: i64, true],[3: i64, 1: i64, true],[5: i64, 6: i64, true],
        [5: i64, 7: i64, true],
        [6: i64, 5: i64, true],[8: i64, 2: i64, true]
      ]

      %graph = builtin.unrealized_conversion_cast %edges_rel : !tuples.tuplestream to !graphalg.mat<#dim x #dim x i1> { cols =[@edges::@src, @edges::@dst, @edges::@val] }

      // 2. Run WCC Algorithm

      %wcc_res = func.call @WCC(%graph) : (!graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x #dim x i1>

      // 3. Materialize

      %result_rel = builtin.unrealized_conversion_cast %wcc_res : !graphalg.mat<#dim x #dim x i1> to !tuples.tuplestream { cols =[@edges::@src, @edges::@dst, @edges::@val] }
      %sorted =  relalg.sort %result_rel [(@edges::@src,asc),(@edges::@dst,asc)]
      %materialized = relalg.materialize %sorted[@edges::@src, @edges::@dst, @edges::@val] => ["node", "label", "val"] : !subop.local_table<[node: i64, label: i64, val: i1],["node", "label", "val"]>

      relalg.query_return %materialized : !subop.local_table<[node: i64, label: i64, val: i1],["node", "label", "val"]>

    } -> !subop.local_table<[node: i64, label: i64, val: i1], ["node", "label", "val"]>

    // Output the result table to stdout!
    subop.set_result 0 %result : !subop.local_table<[node: i64, label: i64, val: i1],["node", "label", "val"]>

    // Check the nodes and their respective weakly connected component labels!
    // CHECK-LABEL: |                          node  |                         label  |                           val  |
    // CHECK-NEXT: ----
    // CHECK-DAG: | 0 | 0 | true |
    // CHECK-DAG: | 1 | 0 | true |
    // CHECK-DAG: | 2 | 0 | true |
    // CHECK-DAG: | 3 | 0 | true |
    // CHECK-DAG: | 5 | 5 | true |
    // CHECK-DAG: | 6 | 5 | true |
    // CHECK-DAG: | 7 | 5 | true |
    // CHECK-DAG: | 8 | 0 | true |
    return
  }
}