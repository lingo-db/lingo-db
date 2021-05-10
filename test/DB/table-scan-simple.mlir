 // RUN: db-run %s %S/../../resources/data/uni | FileCheck %s
 module {
	//CHECK: int(26120)
    //CHECK: int(5001)
	func @main (%execution_context: memref<i8>) {
			%str_const = db.constant ( "---------------" ) :!db.string
	 		%0 = db.tablescan "hoeren" ["matrnr","vorlnr"] %execution_context: memref<i8> : !db.iterable<!db.iterable<tuple<!db.int<64>,!db.int<64>>,table_row_iterator>,table_chunk_iterator>
			db.for %table_chunk in %0 : !db.iterable<!db.iterable<tuple<!db.int<64>,!db.int<64>>,table_row_iterator>,table_chunk_iterator>{
				db.for %table_row in %table_chunk : !db.iterable<tuple<!db.int<64>,!db.int<64>>,table_row_iterator>{
					%3,%4 = util.unpack %table_row : tuple<!db.int<64>, !db.int<64>> -> !db.int<64>, !db.int<64>
					db.dump %3 : !db.int<64>
					db.dump %4 : !db.int<64>
					db.dump %str_const : !db.string
				}
			}
		return
	}
 }