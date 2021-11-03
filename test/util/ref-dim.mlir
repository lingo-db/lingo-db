 // RUN: db-run %s | FileCheck %s
 module {
    func private @rt_dump_index(index) -> ()
	func @main () {
        %c2 = arith.constant 100 : index

        %generic_memref=util.alloc(%c2) : !util.ref<?x!db.int<32,nullable>>
        %size = util.dim %generic_memref : !util.ref<?x!db.int<32,nullable>>
        //CHECK: index(100)
        call @rt_dump_index(%size) : (index) -> ()

		return
	}
 }