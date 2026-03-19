// RUN: mlir-db-opt %s --graphalg-set-dimensions='func=Reachability args=8x8,8x1'  --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg > %t.mlir
// RUN: run-mlir %t.mlir | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>

module {
  func.func private @Reachability(%arg0: !graphalg.mat<#dim x #dim x i1>, %arg1: !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x i1> {
    %0 = graphalg.for_dim range(#dim) init(%arg1) : !graphalg.mat<#dim x 1 x i1> -> !graphalg.mat<#dim x 1 x i1> body {
    ^bb0(%arg2: !graphalg.mat<1 x 1 x i64>, %arg3: !graphalg.mat<#dim x 1 x i1>):
      %1 = graphalg.mxm %arg0, %arg3 : <#dim x #dim x i1>, <#dim x 1 x i1>
      graphalg.yield %1 : !graphalg.mat<#dim x 1 x i1>
    } until {
    }
    return %0 : !graphalg.mat<#dim x 1 x i1>
  }

  func.func @main() {
    %result = relalg.query() {
      %edges_rel = relalg.const_relation columns:[@edges::@src({type=i64}), @edges::@dst({type=i64}), @edges::@val({type=i1})] values: [[1: i64, 2: i64, true], [2: i64, 3: i64, true],[3: i64, 4: i64, true],[4: i64, 5: i64, true], [6: i64, 7: i64, true],[1: i64, 1: i64, true],[2: i64, 2: i64, true],[3: i64, 3: i64, true],[4: i64, 4: i64, true],[5: i64, 5: i64, true],[6: i64, 6: i64, true],[7: i64, 7: i64, true]
      ]
      %graph = builtin.unrealized_conversion_cast %edges_rel : !tuples.tuplestream to !graphalg.mat<#dim x #dim x i1> { cols =[@edges::@dst, @edges::@src, @edges::@val] }

      %source_rel = relalg.const_relation columns:[@source::@id({type=i64}), @source::@val({type=i1})] values: [[1: i64, true]]
      %source = builtin.unrealized_conversion_cast %source_rel : !tuples.tuplestream to !graphalg.mat<#dim x 1 x i1> { cols =[@source::@id, @source::@val] }

      %reach = func.call @Reachability(%graph, %source) : (!graphalg.mat<#dim x #dim x i1>, !graphalg.mat<#dim x 1 x i1>) -> !graphalg.mat<#dim x 1 x i1>

      %result_rel = builtin.unrealized_conversion_cast %reach : !graphalg.mat<#dim x 1 x i1> to !tuples.tuplestream { cols =[@edges::@dst, @edges::@val] }
      %materialized = relalg.materialize %result_rel [@edges::@dst] => ["dst"] : !subop.local_table<[dst: i64], ["dst"]>

      relalg.query_return %materialized : !subop.local_table<[dst: i64], ["dst"]>
    } -> !subop.local_table<[dst: i64], ["dst"]>

    subop.set_result 0 %result : !subop.local_table<[dst: i64],["dst"]>

    // CHECK-LABEL: |                           dst  |
    // CHECK-NEXT: ---
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