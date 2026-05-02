// RUN: mlir-db-opt %s --lower-graphalg-to-graphalg-core --lower-graphalg-core-to-relalg > %t.mlir
// RUN: run-mlir %t.mlir | FileCheck %s

#dim = #graphalg.dim<distinct[0]<>>
module {
  func.func private @combUndir(%arg0: !graphalg.mat<1 x 1 x i64>) -> !graphalg.mat<1 x 1 x f64> {
    %0 = graphalg.literal 1 : i64
    %1 = graphalg.ewise %arg0 SUB %0 : <1 x 1 x i64>
    %2 = graphalg.mxm %arg0, %1 : <1 x 1 x i64>, <1 x 1 x i64>
    %3 = graphalg.cast %2 : <1 x 1 x i64> -> <1 x 1 x f64>
    %4 = graphalg.literal 2.000000e+00 : f64
    %5 = graphalg.ewise %3 DIV %4 : <1 x 1 x f64>
    return %5 : !graphalg.mat<1 x 1 x f64>
  }
  func.func private @LCCUndir(%arg0: !graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x 1 x f64> {
    %0 = graphalg.triu %arg0 : <#dim x #dim x i1>
    %1 = graphalg.cast %arg0 : <#dim x #dim x i1> -> <#dim x #dim x i64>
    %2 = graphalg.reduce %1 : <#dim x #dim x i64> -> <#dim x 1 x i64>
    %3 = graphalg.apply_unary @combUndir %2 : <#dim x 1 x i64> -> <#dim x 1 x f64>
    %4 = graphalg.cast_dim #dim
    %5 = graphalg.cast_dim #dim
    %6 = graphalg.const_mat 0 : i64 -> <#dim x #dim x i64>
    %7 = graphalg.cast %arg0 : <#dim x #dim x i1> -> <#dim x #dim x i64>
    %8 = graphalg.transpose %0 : <#dim x #dim x i1>
    %9 = graphalg.cast %8 : <#dim x #dim x i1> -> <#dim x #dim x i64>
    %10 = graphalg.mxm %7, %9 : <#dim x #dim x i64>, <#dim x #dim x i64>
    %11 = graphalg.mask %6<%arg0 : <#dim x #dim x i1>> = %10 : <#dim x #dim x i64> {complement = false}
    %12 = graphalg.reduce %11 : <#dim x #dim x i64> -> <#dim x 1 x i64>
    %13 = graphalg.cast %12 : <#dim x 1 x i64> -> <#dim x 1 x f64>
    %14 = graphalg.cast %3 : <#dim x 1 x f64> -> <#dim x 1 x f64>
    %15 = graphalg.ewise %13 DIV %14 : <#dim x 1 x f64>
    return %15 : !graphalg.mat<#dim x 1 x f64>
  }
  func.func @main() {
      %result = relalg.query() {
        %edges_rel = relalg.const_relation columns:[@edges::@src({type=i64}), @edges::@dst({type=i64})] values: [
          [1: i64, 2: i64], [2: i64, 1: i64],
          [1: i64, 3: i64], [3: i64, 1: i64],
          [2: i64, 3: i64], [3: i64, 2: i64],
          [4: i64, 5: i64], [5: i64, 4: i64],
          [4: i64, 6: i64], [6: i64, 4: i64],
          [5: i64, 6: i64], [6: i64, 5: i64],
          [7: i64, 8: i64], [8: i64, 7: i64],
          [7: i64, 9: i64], [9: i64, 7: i64],
          [7: i64, 10: i64], [10: i64, 7: i64],
          [8: i64, 9: i64], [9: i64, 8: i64],
          [8: i64, 10: i64], [10: i64, 8: i64],
          [9: i64, 10: i64], [10: i64, 9: i64]
        ]
        %edges_with_val = relalg.map %edges_rel computes : [@edges::@edge_present({type=i1})] (%arg: !tuples.tuple) {
          %true = arith.constant true
          tuples.return %true : i1
        }
        %graph = builtin.unrealized_conversion_cast %edges_with_val : !tuples.tuplestream to !graphalg.mat<#dim x #dim x i1> { cols =[@edges::@src, @edges::@dst, @edges::@edge_present] }
        %lcc_res = func.call @LCCUndir(%graph) : (!graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x 1 x f64>
        %result_rel = builtin.unrealized_conversion_cast %lcc_res : !graphalg.mat<#dim x 1 x f64> to !tuples.tuplestream { cols =[@edges::@dst, @edges::@val] }
        %sorted = relalg.sort %result_rel [(@edges::@dst,asc)]
        %materialized = relalg.materialize %sorted [@edges::@dst, @edges::@val] => ["dst", "val"] : !subop.local_table<[dst: i64, val: f64], ["dst", "val"]>
        relalg.query_return %materialized : !subop.local_table<[dst: i64, val: f64], ["dst", "val"]>
      } -> !subop.local_table<[dst: i64, val: f64], ["dst", "val"]>

      subop.set_result 0 %result : !subop.local_table<[dst: i64, val: f64], ["dst", "val"]>
      // CHECK-LABEL: | dst | val |
      // CHECK-NEXT: ----
      // CHECK-DAG: | 1 | 1
      // CHECK-DAG: | 2 | 1
      // CHECK-DAG: | 3 | 1
      // CHECK-DAG: | 4 | 1
      // CHECK-DAG: | 5 | 1
      // CHECK-DAG: | 6 | 1
      // CHECK-DAG: | 7 | 1
      // CHECK-DAG: | 8 | 1
      // CHECK-DAG: | 9 | 1
      // CHECK-DAG: | 10 | 1
      return
    }
  }
