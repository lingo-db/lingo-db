 // RUN: db-run %s | FileCheck %s
 module {
    func private @rt_dump_index(index) -> ()
	func @main () {
        %c2 = arith.constant 100 : index

        %generic_memref=util.alloc(%c2) : !util.ref<?x!db.nullable<!db.int<32>>>
        %size = util.dim %generic_memref : !util.ref<?x!db.nullable<!db.int<32>>>
        //CHECK: index(100)
        call @rt_dump_index(%size) : (index) -> ()

		return
	}
 }