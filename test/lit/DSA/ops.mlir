// RUN: mlir-db-opt -allow-unregistered-dialect %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  | FileCheck %s
// CHECK:   %{{.*}} = dsa.for %{{.*}} in %{{.*}} : !util.buffer<i8>  iter_args(%{{.*}} = %c0_i8) -> (i8){
func.func @test(%arg0: !util.buffer<i8>) -> i8 {
  %c0_i8 = arith.constant 0 : i8
  %0 = dsa.for %arg1 in %arg0 : !util.buffer<i8>  iter_args(%arg2 = %c0_i8) -> (i8){
    dsa.yield %arg2 : i8
  }
  return %0 : i8
}
