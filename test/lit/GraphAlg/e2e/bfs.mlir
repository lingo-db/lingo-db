// RUN: mlir-db-opt %s --graphalg-set-dimensions='func=BFS args=10x10,10x1' --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg > %t.mlir
// RUN: run-mlir %t.mlir | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

module {
  // BFS Helper Function (Calculates the depth)
  func.func private @setDepth(%arg0: !graphalg.mat<1 x 1 x i1>, %arg1: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x i64> {
    %0 = graphalg.cast %arg0 : <1 x 1 x i1> -> <1 x 1 x i64>
    %1 = graphalg.literal 2 : i64
    %2 = graphalg.ewise %arg1 ADD %1 : <1 x 1 x i64>
    %3 = graphalg.mxm %0, %2 : <1 x 1 x i64>, <1 x 1 x i64>
    return %3 : !graphalg.mat<1 x 1 x i64>
  }

  // BFS Algorithm
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

  func.func @main() {
    %result = relalg.query() {

      // 1. Setup the Graph Relation

      %edges_rel = relalg.const_relation columns:[@edges::@src({type=i64}), @edges::@dst({type=i64}), @edges::@val({type=i1})] values:[
        [0: i64, 1: i64, true],[0: i64, 2: i64, true], [1: i64, 2: i64, true], [1: i64, 3: i64, true],[1: i64, 4: i64, true],[2: i64, 0: i64, true], [3: i64, 5: i64, true],[3: i64, 6: i64, true],[3: i64, 7: i64, true], [4: i64, 0: i64, true],[4: i64, 1: i64, true],[5: i64, 3: i64, true],
        [5: i64, 7: i64, true],[7: i64, 0: i64, true],[7: i64, 1: i64, true], [7: i64, 2: i64, true],
        [8: i64, 9: i64, true]
      ]

      %graph = builtin.unrealized_conversion_cast %edges_rel : !tuples.tuplestream to !graphalg.mat<#dim x #dim x i1> { cols =[@edges::@src, @edges::@dst, @edges::@val] }


      // 2. Setup the Source Vector (Start at Node 0)

      %source_rel = relalg.const_relation columns:[@source::@id({type=i64}), @source::@val({type=i1})] values: [[0: i64, true]]
      %source = builtin.unrealized_conversion_cast %source_rel : !tuples.tuplestream to !graphalg.mat<#dim x 1 x i1> { cols =[@source::@id, @source::@val] }


      // 3. Run BFS Algorithm

      %bfs_res = func.call @BFS(%graph, %source) : (!graphalg.mat<#dim x #dim x i1>, !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x i64>


      // 4. Materialize

      %result_rel = builtin.unrealized_conversion_cast %bfs_res : !graphalg.mat<#dim x 1 x i64> to !tuples.tuplestream { cols =[@edges::@dst, @result::@depth] }

      %materialized = relalg.materialize %result_rel[@edges::@dst, @result::@depth] => ["dst", "val"] : !subop.local_table<[dst: i64, val: i64], ["dst", "val"]>

      relalg.query_return %materialized : !subop.local_table<[dst: i64, val: i64], ["dst", "val"]>

    } -> !subop.local_table<[dst: i64, val: i64], ["dst", "val"]>

    subop.set_result 0 %result : !subop.local_table<[dst: i64, val: i64], ["dst", "val"]>

    // Check that all nodes reached from source (0) have correctly calculated depths!
    // CHECK-LABEL: |                           dst  |                           val  |
    // CHECK-NEXT: ----

    // CHECK-DAG: | 0 | 1 |
    // CHECK-DAG: | 1 | 2 |
    // CHECK-DAG: | 2 | 2 |
    // CHECK-DAG: | 3 | 3 |
    // CHECK-DAG: | 4 | 3 |
    // CHECK-DAG: | 5 | 4 |
    // CHECK-DAG: | 6 | 4 |
    // CHECK-DAG: | 7 | 4 |
    return
  }
}