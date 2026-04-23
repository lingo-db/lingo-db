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
      %edges_rel = relalg.const_relation columns:[@edges::@src({type=i64}), @edges::@dst({type=i64}), @edges::@val({type=i1})] values: [[0: i64, 18: i64, true],[0: i64, 20: i64, true], [0: i64, 21: i64, true],[0: i64, 26: i64, true],[0: i64, 30: i64, true], [0: i64, 36: i64, true],[0: i64, 44: i64, true],[0: i64, 47: i64, true], [1: i64, 2: i64, true], [1: i64, 19: i64, true],[1: i64, 38: i64, true],[1: i64, 45: i64, true], [2: i64, 5: i64, true],[2: i64, 9: i64, true],[2: i64, 31: i64, true], [2: i64, 40: i64, true],[2: i64, 44: i64, true],[3: i64, 14: i64, true], [4: i64, 14: i64, true], [4: i64, 15: i64, true],[4: i64, 17: i64, true],[4: i64, 27: i64, true], [4: i64, 46: i64, true],[5: i64, 48: i64, true],[6: i64, 5: i64, true], [6: i64, 26: i64, true],[6: i64, 42: i64, true],[6: i64, 45: i64, true], [7: i64, 4: i64, true], [7: i64, 20: i64, true],[7: i64, 28: i64, true],[7: i64, 29: i64, true], [7: i64, 31: i64, true],[7: i64, 42: i64, true],[8: i64, 15: i64, true], [8: i64, 17: i64, true],[8: i64, 20: i64, true],[8: i64, 27: i64, true],[8: i64, 29: i64, true], [8: i64, 34: i64, true],[8: i64, 39: i64, true],[9: i64, 8: i64, true], [9: i64, 12: i64, true],[9: i64, 27: i64, true],[9: i64, 28: i64, true], [9: i64, 32: i64, true], [10: i64, 2: i64, true],[10: i64, 38: i64, true],[11: i64, 46: i64, true], [11: i64, 49: i64, true],[12: i64, 6: i64, true],[12: i64, 11: i64, true],[12: i64, 16: i64, true], [12: i64, 31: i64, true],[12: i64, 47: i64, true],[13: i64, 3: i64, true],[13: i64, 19: i64, true], [13: i64, 20: i64, true],[13: i64, 34: i64, true],[13: i64, 37: i64, true],[13: i64, 39: i64, true], [14: i64, 7: i64, true],[14: i64, 23: i64, true],[14: i64, 30: i64, true],[14: i64, 34: i64, true], [14: i64, 43: i64, true],[16: i64, 4: i64, true],[16: i64, 8: i64, true], [16: i64, 10: i64, true], [16: i64, 15: i64, true],[16: i64, 25: i64, true],[16: i64, 36: i64, true], [17: i64, 0: i64, true],[17: i64, 11: i64, true],[17: i64, 27: i64, true],[17: i64, 29: i64, true], [17: i64, 43: i64, true],[17: i64, 44: i64, true],[17: i64, 46: i64, true],[17: i64, 49: i64, true], [18: i64, 9: i64, true],[18: i64, 10: i64, true],[18: i64, 12: i64, true],[18: i64, 26: i64, true], [18: i64, 37: i64, true],[19: i64, 14: i64, true],[19: i64, 24: i64, true],[20: i64, 21: i64, true], [20: i64, 26: i64, true],[20: i64, 30: i64, true],[20: i64, 31: i64, true],[20: i64, 39: i64, true], [21: i64, 18: i64, true],[21: i64, 25: i64, true],[21: i64, 26: i64, true],[21: i64, 30: i64, true], [22: i64, 21: i64, true],[22: i64, 34: i64, true],[22: i64, 35: i64, true],[22: i64, 37: i64, true], [22: i64, 39: i64, true],[22: i64, 45: i64, true],[22: i64, 46: i64, true],[23: i64, 8: i64, true], [23: i64, 12: i64, true],[23: i64, 14: i64, true],[23: i64, 33: i64, true],[23: i64, 35: i64, true], [23: i64, 49: i64, true],[24: i64, 7: i64, true],[24: i64, 23: i64, true],[24: i64, 29: i64, true], [24: i64, 33: i64, true],[24: i64, 40: i64, true],[24: i64, 46: i64, true],[25: i64, 6: i64, true], [25: i64, 30: i64, true],[25: i64, 36: i64, true],[25: i64, 39: i64, true],[25: i64, 43: i64, true], [25: i64, 46: i64, true],[26: i64, 30: i64, true],[26: i64, 32: i64, true],[26: i64, 42: i64, true], [27: i64, 7: i64, true],[27: i64, 31: i64, true],[27: i64, 41: i64, true], [27: i64, 44: i64, true], [28: i64, 0: i64, true],[28: i64, 1: i64, true],[28: i64, 11: i64, true], [28: i64, 13: i64, true],[28: i64, 15: i64, true],[28: i64, 18: i64, true],[28: i64, 19: i64, true], [28: i64, 35: i64, true],[29: i64, 8: i64, true],[29: i64, 23: i64, true],[29: i64, 33: i64, true], [29: i64, 43: i64, true],[30: i64, 10: i64, true],[30: i64, 16: i64, true],[30: i64, 31: i64, true], [30: i64, 38: i64, true],[30: i64, 45: i64, true],[30: i64, 46: i64, true],[31: i64, 1: i64, true], [31: i64, 27: i64, true],[31: i64, 28: i64, true],[31: i64, 29: i64, true],[31: i64, 30: i64, true], [32: i64, 6: i64, true],[32: i64, 7: i64, true],[32: i64, 8: i64, true], [32: i64, 9: i64, true],[32: i64, 31: i64, true],[32: i64, 33: i64, true],[32: i64, 36: i64, true], [33: i64, 25: i64, true],[33: i64, 47: i64, true],[34: i64, 2: i64, true],[34: i64, 9: i64, true], [34: i64, 16: i64, true],[34: i64, 23: i64, true],[34: i64, 25: i64, true],[34: i64, 27: i64, true], [34: i64, 32: i64, true],[34: i64, 40: i64, true],[35: i64, 19: i64, true],[35: i64, 20: i64, true], [35: i64, 28: i64, true],[35: i64, 31: i64, true],[35: i64, 45: i64, true],[36: i64, 0: i64, true], [36: i64, 4: i64, true],[36: i64, 8: i64, true],[36: i64, 12: i64, true], [36: i64, 22: i64, true],[36: i64, 23: i64, true],[37: i64, 1: i64, true],[37: i64, 21: i64, true], [37: i64, 49: i64, true],[38: i64, 5: i64, true],[38: i64, 7: i64, true],[38: i64, 19: i64, true], [38: i64, 27: i64, true],[38: i64, 29: i64, true],[38: i64, 46: i64, true],[38: i64, 47: i64, true], [39: i64, 4: i64, true],[39: i64, 6: i64, true],[39: i64, 7: i64, true], [39: i64, 10: i64, true],[39: i64, 32: i64, true],[39: i64, 33: i64, true],[39: i64, 36: i64, true], [39: i64, 48: i64, true],[40: i64, 23: i64, true],[40: i64, 42: i64, true],[42: i64, 0: i64, true], [42: i64, 1: i64, true],[42: i64, 10: i64, true],[42: i64, 14: i64, true],[42: i64, 16: i64, true], [42: i64, 28: i64, true],[42: i64, 37: i64, true],[42: i64, 46: i64, true],[43: i64, 10: i64, true], [43: i64, 12: i64, true],[43: i64, 14: i64, true],[44: i64, 4: i64, true],[44: i64, 10: i64, true], [44: i64, 11: i64, true],[44: i64, 20: i64, true],[44: i64, 23: i64, true],[45: i64, 22: i64, true], [45: i64, 23: i64, true],[45: i64, 25: i64, true],[45: i64, 30: i64, true],[45: i64, 35: i64, true], [45: i64, 40: i64, true],[46: i64, 7: i64, true],[46: i64, 13: i64, true],[46: i64, 15: i64, true], [46: i64, 27: i64, true],[46: i64, 28: i64, true],[46: i64, 33: i64, true],[46: i64, 34: i64, true], [46: i64, 39: i64, true],[46: i64, 41: i64, true],[46: i64, 45: i64, true],[46: i64, 49: i64, true], [47: i64, 7: i64, true],[47: i64, 18: i64, true],[47: i64, 29: i64, true], [47: i64, 34: i64, true], [47: i64, 37: i64, true],[47: i64, 42: i64, true],[47: i64, 49: i64, true], [48: i64, 6: i64, true],[48: i64, 7: i64, true],[48: i64, 16: i64, true],[48: i64, 17: i64, true], [49: i64, 3: i64, true],[49: i64, 27: i64, true],[49: i64, 46: i64, true]]
      %graph = builtin.unrealized_conversion_cast %edges_rel : !tuples.tuplestream to !graphalg.mat<#dim x #dim x i1> { cols =[@edges::@src, @edges::@dst, @edges::@val] }

      %pr = func.call @PR(%graph) : (!graphalg.mat<#dim x #dim x i1>) -> !graphalg.mat<#dim x 1 x f64>

      %result_rel = builtin.unrealized_conversion_cast %pr : !graphalg.mat<#dim x 1 x f64> to !tuples.tuplestream { cols =[@pr::@node, @pr::@val] }
      %materialized = relalg.materialize %result_rel[@pr::@node, @pr::@val] => ["node", "pr"] : !subop.local_table<[node: i64, pr: f64], ["node", "pr"]>

      relalg.query_return %materialized : !subop.local_table<[node: i64, pr: f64], ["node", "pr"]>
    } -> !subop.local_table<[node: i64, pr: f64], ["node", "pr"]>

    subop.set_result 0 %result : !subop.local_table<[node: i64, pr: f64],["node", "pr"]>

    // CHECK: |{{.*}}node{{.*}}|{{.*}}pr{{.*}}|
    // CHECK-NEXT: ---
    // CHECK-DAG: |{{ +}}0{{ +}}|{{.*}}0.012305{{.*}}|
    // CHECK-DAG: |{{ +}}1{{ +}}|{{.*}}0.018516{{.*}}|
    // CHECK-DAG: |{{ +}}2{{ +}}|{{.*}}0.020895{{.*}}|
    // CHECK-DAG: |{{ +}}3{{ +}}|{{.*}}0.011765{{.*}}|
    // CHECK-DAG: |{{ +}}4{{ +}}|{{.*}}0.018121{{.*}}|
    // CHECK-DAG: |{{ +}}5{{ +}}|{{.*}}0.013705{{.*}}|
    // CHECK-DAG: |{{ +}}6{{ +}}|{{.*}}0.017670{{.*}}|
    // CHECK-DAG: |{{ +}}7{{ +}}|{{.*}}0.034001{{.*}}|
    // CHECK-DAG: |{{ +}}8{{ +}}|{{.*}}0.022537{{.*}}|
    // CHECK-DAG: |{{ +}}9{{ +}}|{{.*}}0.013155{{.*}}|
    // CHECK-DAG: |{{ +}}10{{ +}}|{{.*}}0.026543{{.*}}|
    // CHECK-DAG: |{{ +}}11{{ +}}|{{.*}}0.013863{{.*}}|
    // CHECK-DAG: |{{ +}}12{{ +}}|{{.*}}0.020308{{.*}}|
    // CHECK-DAG: |{{ +}}13{{ +}}|{{.*}}0.009025{{.*}}|
    // CHECK-DAG: |{{ +}}14{{ +}}|{{.*}}0.036728{{.*}}|
    // CHECK-DAG: |{{ +}}15{{ +}}|{{.*}}0.017720{{.*}}|
    // CHECK-DAG: |{{ +}}16{{ +}}|{{.*}}0.020311{{.*}}|
    // CHECK-DAG: |{{ +}}17{{ +}}|{{.*}}0.012985{{.*}}|
    // CHECK-DAG: |{{ +}}18{{ +}}|{{.*}}0.012671{{.*}}|
    // CHECK-DAG: |{{ +}}19{{ +}}|{{.*}}0.016796{{.*}}|
    // CHECK-DAG: |{{ +}}20{{ +}}|{{.*}}0.019116{{.*}}|
    // CHECK-DAG: |{{ +}}21{{ +}}|{{.*}}0.012898{{.*}}|
    // CHECK-DAG: |{{ +}}22{{ +}}|{{.*}}0.008824{{.*}}|
    // CHECK-DAG: |{{ +}}23{{ +}}|{{.*}}0.032901{{.*}}|
    // CHECK-DAG: |{{ +}}24{{ +}}|{{.*}}0.010670{{.*}}|
    // CHECK-DAG: |{{ +}}25{{ +}}|{{.*}}0.023695{{.*}}|
    // CHECK-DAG: |{{ +}}26{{ +}}|{{.*}}0.016739{{.*}}|
    // CHECK-DAG: |{{ +}}27{{ +}}|{{.*}}0.033753{{.*}}|
    // CHECK-DAG: |{{ +}}28{{ +}}|{{.*}}0.024650{{.*}}|
    // CHECK-DAG: |{{ +}}29{{ +}}|{{.*}}0.025260{{.*}}|
    // CHECK-DAG: |{{ +}}30{{ +}}|{{.*}}0.034319{{.*}}|
    // CHECK-DAG: |{{ +}}31{{ +}}|{{.*}}0.034973{{.*}}|
    // CHECK-DAG: |{{ +}}32{{ +}}|{{.*}}0.014581{{.*}}|
    // CHECK-DAG: |{{ +}}33{{ +}}|{{.*}}0.021640{{.*}}|
    // CHECK-DAG: |{{ +}}34{{ +}}|{{.*}}0.020208{{.*}}|
    // CHECK-DAG: |{{ +}}35{{ +}}|{{.*}}0.015084{{.*}}|
    // CHECK-DAG: |{{ +}}36{{ +}}|{{.*}}0.014767{{.*}}|
    // CHECK-DAG: |{{ +}}37{{ +}}|{{.*}}0.013190{{.*}}|
    // CHECK-DAG: |{{ +}}38{{ +}}|{{.*}}0.023610{{.*}}|
    // CHECK-DAG: |{{ +}}39{{ +}}|{{.*}}0.018100{{.*}}|
    // CHECK-DAG: |{{ +}}40{{ +}}|{{.*}}0.013943{{.*}}|
    // CHECK-DAG: |{{ +}}41{{ +}}|{{.*}}0.013578{{.*}}|
    // CHECK-DAG: |{{ +}}42{{ +}}|{{.*}}0.025243{{.*}}|
    // CHECK-DAG: |{{ +}}43{{ +}}|{{.*}}0.019880{{.*}}|
    // CHECK-DAG: |{{ +}}44{{ +}}|{{.*}}0.016944{{.*}}|
    // CHECK-DAG: |{{ +}}45{{ +}}|{{.*}}0.022593{{.*}}|
    // CHECK-DAG: |{{ +}}46{{ +}}|{{.*}}0.037191{{.*}}|
    // CHECK-DAG: |{{ +}}47{{ +}}|{{.*}}0.020356{{.*}}|
    // CHECK-DAG: |{{ +}}48{{ +}}|{{.*}}0.017103{{.*}}|
    // CHECK-DAG: |{{ +}}49{{ +}}|{{.*}}0.024548{{.*}}|

    return
  }
}