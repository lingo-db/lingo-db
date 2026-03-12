// RUN: mlir-db-opt %s --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg > %t.mlir
// RUN: run-mlir %t.mlir | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

module {
  func.func @main() {
    %result = relalg.query() {

      // -----------------------------------------------------------------
      // 1. Setup the Graph Relation
      // Path: 1->2->3->4->5 and path 6->7 (with self-edges)
      // -----------------------------------------------------------------
      %edges_rel = relalg.const_relation columns:[@edges::@src({type=i64}), @edges::@dst({type=i64}), @edges::@val({type=i1})] values: [[1: i64, 2: i64, true], [2: i64, 3: i64, true],[3: i64, 4: i64, true],[4: i64, 5: i64, true], [6: i64, 7: i64, true],
          [1: i64, 1: i64, true],[2: i64, 2: i64, true],[3: i64, 3: i64, true],[4: i64, 4: i64, true],[5: i64, 5: i64, true],[6: i64, 6: i64, true],[7: i64, 7: i64, true]
      ]
      %graph = builtin.unrealized_conversion_cast %edges_rel : !tuples.tuplestream to !graphalg.mat<#dim x #dim x i1> { cols =[@edges::@dst, @edges::@src, @edges::@val] }

      // -----------------------------------------------------------------
      // 2. Setup the Source Vector (Start at Node 1)
      // -----------------------------------------------------------------
      %source_rel = relalg.const_relation columns:[@source::@id({type=i64}), @source::@val({type=i1})] values: [[1: i64, true]]
      %source = builtin.unrealized_conversion_cast %source_rel : !tuples.tuplestream to !graphalg.mat<#dim x 1 x i1> { cols =[@source::@id, @source::@val] }

      // -----------------------------------------------------------------
      // 3. Run Reachability Algorithm (Unrolled 4 Hops)
      // -----------------------------------------------------------------
      %reach1 = graphalg.mxm %graph, %source : <#dim x #dim x i1>, <#dim x 1 x i1>
      %reach2 = graphalg.mxm %graph, %reach1 : <#dim x #dim x i1>, <#dim x 1 x i1>
      %reach3 = graphalg.mxm %graph, %reach2 : <#dim x #dim x i1>, <#dim x 1 x i1>
      %reach4 = graphalg.mxm %graph, %reach3 : <#dim x #dim x i1>, <#dim x 1 x i1>

      // -----------------------------------------------------------------
      // 4. Materialize
      // -----------------------------------------------------------------
      %result_rel = builtin.unrealized_conversion_cast %reach4 : !graphalg.mat<#dim x 1 x i1> to !tuples.tuplestream { cols =[@edges::@dst, @edges::@val] }

      %materialized = relalg.materialize %result_rel [@edges::@dst] => ["dst"] : !subop.local_table<[dst: i64], ["dst"]>

      relalg.query_return %materialized : !subop.local_table<[dst: i64], ["dst"]>

    } -> !subop.local_table<[dst: i64], ["dst"]>

    // =================================================================
    // Output the result table to stdout!
    // =================================================================
    subop.set_result 0 %result : !subop.local_table<[dst: i64],["dst"]>

    // Check that all 5 nodes were successfully reached!
    // CHECK-LABEL: |                           dst  |
    // CHECK-NEXT: ----------------------------------

    // 2. Assert 1-5 exist (order-independent)
    // CHECK-DAG: |                             1  |
    // CHECK-DAG: |                             2  |
    // CHECK-DAG: |                             3  |
    // CHECK-DAG: |                             4  |
    // CHECK-DAG: |                             5  |

    // CHECK-NOT: |                             6  |
    // CHECK-NOT: |                             7  |
    // CHECK-NOT: |                             8  |
    return
  }
}