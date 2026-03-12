// RUN: mlir-db-opt %s --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg > %t.mlir
// RUN: run-mlir %t.mlir | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

module {
  // =================================================================
  // SSSP Algorithm (Single-Source Shortest Path)
  // =================================================================
  func.func private @SSSP(%arg0: !graphalg.mat<#dim x #dim x !graphalg.trop_f64>, %arg1: !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64> {
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

  func.func @main() {
    %result = relalg.query() {

      // -----------------------------------------------------------------
      // 1. Setup the Graph Relation
      // Path 1->2 (1.5), 2->3 (2.0), 3->4 (1.0), 4->5 (3.0), and 1->3 (5.0)
      // -----------------------------------------------------------------
      %edges_rel = relalg.const_relation columns:[@edges::@src({type=i64}), @edges::@dst({type=i64}), @edges::@val({type=f64})] values: [[1: i64, 2: i64, 1.5 : f64],[2: i64, 3: i64, 2.0 : f64],[3: i64, 4: i64, 1.0 : f64],[4: i64, 5: i64, 3.0 : f64],[1: i64, 3: i64, 5.0 : f64]
      ]

      // We map dim0 to @src and dim1 to @dst so that vxm (v * A) properly propagates distances forward.
      %graph = builtin.unrealized_conversion_cast %edges_rel : !tuples.tuplestream to !graphalg.mat<#dim x #dim x !graphalg.trop_f64> { cols =[@edges::@src, @edges::@dst, @edges::@val] }

      // -----------------------------------------------------------------
      // 2. Setup the Source Vector (Start at Node 1)
      // -----------------------------------------------------------------
      %source_rel = relalg.const_relation columns:[@source::@id({type=i64}), @source::@val({type=i1})] values: [[1: i64, true]]
      %source = builtin.unrealized_conversion_cast %source_rel : !tuples.tuplestream to !graphalg.mat<#dim x 1 x i1> { cols =[@source::@id, @source::@val] }

      // -----------------------------------------------------------------
      // 3. Run SSSP Algorithm
      // -----------------------------------------------------------------
      %sssp_res = func.call @SSSP(%graph, %source) : (!graphalg.mat<#dim x #dim x !graphalg.trop_f64>, !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x !graphalg.trop_f64>

      // -----------------------------------------------------------------
      // 4. Materialize
      // -----------------------------------------------------------------
      %result_rel = builtin.unrealized_conversion_cast %sssp_res : !graphalg.mat<#dim x 1 x !graphalg.trop_f64> to !tuples.tuplestream { cols =[@edges::@dst, @edges::@val] }

      %materialized = relalg.materialize %result_rel [@edges::@dst, @edges::@val] => ["dst", "val"] : !subop.local_table<[dst: i64, val: f64], ["dst", "val"]>

      relalg.query_return %materialized : !subop.local_table<[dst: i64, val: f64], ["dst", "val"]>

    } -> !subop.local_table<[dst: i64, val: f64], ["dst", "val"]>

    // =================================================================
    // Output the result table to stdout!
    // =================================================================
    subop.set_result 0 %result : !subop.local_table<[dst: i64, val: f64], ["dst", "val"]>

    // Check that all 5 nodes were successfully reached with the correct minimum distances!
    // CHECK-LABEL: |                           dst  |                           val  |
    // CHECK-NEXT: --------------------------------------------------------------------

    // CHECK-DAG: |{{[ ]*}}1{{[ ]*}}|{{[ ]*}}0{{(\.0*)?}}{{[ ]*}}|
    // CHECK-DAG: |{{[ ]*}}2{{[ ]*}}|{{[ ]*}}1\.5{{(0*)?}}{{[ ]*}}|
    // CHECK-DAG: |{{[ ]*}}3{{[ ]*}}|{{[ ]*}}3\.5{{(0*)?}}{{[ ]*}}|
    // CHECK-DAG: |{{[ ]*}}4{{[ ]*}}|{{[ ]*}}4\.5{{(0*)?}}{{[ ]*}}|
    // CHECK-DAG: |{{[ ]*}}5{{[ ]*}}|{{[ ]*}}7\.5{{(0*)?}}{{[ ]*}}|

    return
  }
}