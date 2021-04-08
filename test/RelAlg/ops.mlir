// RUN: mlir-db-opt -allow-unregistered-dialect %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  | FileCheck %s
module{
// CHECK: %0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["MAIL", "SHIP"]
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["MAIL", "SHIP"]
}
