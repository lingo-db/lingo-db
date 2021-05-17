 // RUN: db-run %s | FileCheck %s
 module {
 	func @main () {
        %c1 = constant 1 : index
        %c10 = constant 10 : index
        %false = db.constant ( 0 ) : !db.bool
        %true = db.constant ( 1 ) : !db.bool
		%5 = util.pack %true, %false : !db.bool, !db.bool -> tuple<!db.bool, !db.bool>

        %memref=memref.alloc(%c10) : memref<?xi8>
        %generic_memref= util.to_generic_memref %memref : memref<?xi8> -> !util.generic_memref<tuple<!db.bool, !db.bool>>

        util.store %5:tuple<!db.bool, !db.bool>,%generic_memref[%c1] :!util.generic_memref<tuple<!db.bool, !db.bool>>
        %memberref=util.member_ref %generic_memref[%c1][1] :!util.generic_memref<tuple<!db.bool, !db.bool>> -> !util.generic_memref<!db.bool>
        util.store %true:!db.bool,%memberref[] :!util.generic_memref<!db.bool>
        %res1=util.load %generic_memref[%c1] :!util.generic_memref<tuple<!db.bool, !db.bool>> -> tuple<!db.bool, !db.bool>
		%6,%7 = util.unpack %res1 : tuple<!db.bool, !db.bool> -> !db.bool, !db.bool
		//CHECK: bool(true)
		//CHECK: bool(true)
		db.dump %6 : !db.bool
		db.dump %7 : !db.bool
		return
	}
 }