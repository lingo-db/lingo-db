// RUN: mlir-db-opt -allow-unregistered-dialect %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  | FileCheck %s
module{
// CHECK: %0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["MAIL", "SHIP"]
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["MAIL", "SHIP"]
}

// -----
module{
%0 = relalg.const_relation @constrel  attributes: [@attr1({type = !db.string})] values: ["A", "B"]
//CHECK: %1 = relalg.selection %0 (%arg0: !relalg.tuple) {
%1 = relalg.selection %0 (%arg0: !relalg.tuple) {
    //CHECK:    %2 = db.constant( "true" ) : !db.bool
	%2 = db.constant( "true" ) : !db.bool
	//CHECK:    relalg.return %2 : !db.bool
	relalg.return %2 : !db.bool
}
}