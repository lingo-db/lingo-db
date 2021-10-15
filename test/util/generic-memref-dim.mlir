 // RUN: db-run %s | FileCheck %s
 module {
    func private @dump_int(i1,index) -> () attributes {llvm.emit_c_interface}
	func @main () {
        %false = arith.constant 0 : i1
        %c2 = arith.constant 100 : index

        %generic_memref=util.alloc(%c2) : !util.generic_memref<?x!db.int<32,nullable>>
        %size = util.dim %generic_memref : !util.generic_memref<?x!db.int<32,nullable>>
        //CHECK: int(100)
        call @dump_int(%false,%size) : (i1,index) -> ()

		return
	}
 }