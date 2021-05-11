 // RUN: db-run %s %S/../../resources/data/uni | FileCheck %s
 module {
	//CHECK: int(24002)
    //CHECK: int(18)
    //CHECK: string("Xenokrates")
	func @main (%execution_context: memref<i8>) {
			%str_const = db.constant ( "---------------" ) :!db.string
	 		%0 = db.tablescan "studenten" ["matrnr","semester","name"] %execution_context: memref<i8> : !db.iterable<!db.iterable<tuple<!db.int<64>,!db.int<64>,!db.string>,table_row_iterator>,table_chunk_iterator>
			db.for %table_chunk in %0 : !db.iterable<!db.iterable<tuple<!db.int<64>,!db.int<64>,!db.string>,table_row_iterator>,table_chunk_iterator>{
				db.for %table_row in %table_chunk : !db.iterable<tuple<!db.int<64>,!db.int<64>,!db.string>,table_row_iterator>{
					%3,%4,%5 = util.unpack %table_row : tuple<!db.int<64>, !db.int<64>,!db.string> -> !db.int<64>, !db.int<64>,!db.string
					db.dump %3 : !db.int<64>
					db.dump %4 : !db.int<64>
					db.dump %5 : !db.string
					db.dump %str_const : !db.string
				}
			}
		return
	}
 }