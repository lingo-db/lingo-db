// RUN: mlir-db-opt %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  --relalg-extract-nested-operators | FileCheck %s
module  {
  func @query() {
    //CHECK: %{{.*}} = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
    //CHECK: %{{.*}} = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
    //CHECK: %{{.*}} = relalg.const_relation @constrel3  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
    //CHECK: %{{.*}} = relalg.selection %2 (%arg0: !relalg.tuple) {
    //CHECK-DAG: %{{.*}} = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
    //CHECK-DAG: %{{.*}} = relalg.getattr %arg0 @constrel3::@attr1 : !db.int<32>
    //CHECK-DAG: %{{.*}} = relalg.getattr %arg0 @constrel2::@attr1 : !db.int<32>
    //CHECK: relalg.return
    //CHECK: %{{.*}} = relalg.selection %1 (%arg0: !relalg.tuple) {
	//CHECK: relalg.return
	//CHECK: %{{.*}} = relalg.selection %0 (%arg0: !relalg.tuple) {
	//CHECK: %{{.*}} = relalg.exists
	//CHECK: relalg.return
    %0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
    %1 = relalg.selection %0 (%arg0: !relalg.tuple) {
      %2 = relalg.const_relation @constrel2  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
      %3 = relalg.selection %2 (%arg1: !relalg.tuple) {
        %5 = relalg.const_relation @constrel3  attributes: [@attr1({type = !db.int<32>})] values: [1, 2]
        %6 = relalg.getattr %arg1 @constrel2::@attr1 : !db.int<32>
        %7 = relalg.selection %5 (%arg2: !relalg.tuple) {
          %13 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
          %14 = relalg.getattr %arg2 @constrel3::@attr1 : !db.int<32>
          %15 = db.compare eq %13 : !db.int<32>, %14 : !db.int<32>
          %16 = db.compare eq %6 : !db.int<32>, %14 : !db.int<32>
          %17 = db.and %15:!db.bool,%16:!db.bool
          relalg.return %17 : !db.bool
        }
        %8 = relalg.getattr %arg0 @constrel::@attr1 : !db.int<32>
        %9 = relalg.getattr %arg1 @constrel2::@attr1 : !db.int<32>
        %10 = db.compare eq %8 : !db.int<32>, %9 : !db.int<32>
        %11 = relalg.exists %7
        %12 = db.and %10:!db.bool,%11:!db.bool
        relalg.return %12 : !db.bool
      }
      %4 = relalg.exists %3
      relalg.return %4 : !db.bool
    }
    return
  }
}
