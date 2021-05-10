 // RUN: db-run %s %S/../../resources/data/uni | FileCheck %s
 module {
 	memref.global "private" @hoeren : memref<6xi8> = dense<[104,111,101,114,101,110]>
 	memref.global "private" @matrnr : memref<6xi8> = dense<[109, 97, 116, 114, 110, 114]>
	memref.global "private" @vorlnr : memref<6xi8> = dense<[118, 111, 114, 108, 110, 114]>

	func private @get_table(memref<i8>, memref<i8>, i32) -> memref<i8>
	func private @get_column_id(memref<i8>, memref<i8>, i32) -> index
	func private @table_chunk_iterator_init(memref<i8>) -> memref<i8>
	func private @table_chunk_iterator_next(memref<i8>) -> memref<i8>
	func private @table_chunk_iterator_curr(memref<i8>) -> memref<i8>
	func private @table_chunk_iterator_valid(memref<i8>) -> i1
	func private @table_chunk_iterator_free(memref<i8>) -> ()
	func private @table_chunk_num_rows(memref<i8>) -> i64
	func private @dumpInt(i1, i64) -> ()
	func private @table_chunk_get_column_buffer(memref<i8>,index,index) -> memref<i8>
	func private @table_column_get_int_64(memref<i8>, index,index) -> !db.int<64>
	func private @table_chunk_get_column_offset(memref<i8>,index) -> index
	//CHECK: int(26120)
    //CHECK: int(5001)
 	func @main (%execution_context: memref<i8>) {
 	    %c0 = constant 0 : index
 	    %c1 = constant 1 : index

 	    %false = constant false
		%str_const = db.constant ( "---------------" ) :!db.string
		%0 = memref.get_global @hoeren : memref<6xi8>
		%1 = memref.get_global @matrnr : memref<6xi8>
		%2 = memref.get_global @vorlnr : memref<6xi8>
		%00 = memref.cast %0 : memref<6xi8> to memref<?xi8>
		%01 = memref.cast %1 : memref<6xi8> to memref<?xi8>
		%02 = memref.cast %2 : memref<6xi8> to memref<?xi8>
		%10 = memref.dim %00, %c0 : memref<?xi8>
		%11 = memref.dim %01, %c0 : memref<?xi8>
		%12 = memref.dim %02, %c0 : memref<?xi8>
		%20 = index_cast %10 : index to i32
		%21 = index_cast %11 : index to i32
		%22 = index_cast %12 : index to i32
		%30 = memref.reinterpret_cast %00 to offset: [0], sizes: [], strides: [] : memref<?xi8> to memref<i8>
		%31 = memref.reinterpret_cast %01 to offset: [0], sizes: [], strides: [] : memref<?xi8> to memref<i8>
		%32 = memref.reinterpret_cast %02 to offset: [0], sizes: [], strides: [] : memref<?xi8> to memref<i8>

		%hoeren_table = call  @get_table(%execution_context, %30, %20) : (memref<i8>, memref<i8>, i32) -> memref<i8>
		%matrnr_id =  call @get_column_id(%hoeren_table,%31,%21) : (memref<i8>, memref<i8>, i32) -> index
		%vorlnr_id =  call @get_column_id(%hoeren_table,%32,%22) : (memref<i8>, memref<i8>, i32) -> index
		%table_chunk_iterator = call @table_chunk_iterator_init(%hoeren_table) : (memref<i8>) -> memref<i8>
   		%res = scf.while (%arg1 = %table_chunk_iterator) : (memref<i8>) -> (memref<i8>) {
      		%condition = call @table_chunk_iterator_valid(%arg1) : (memref<i8>) -> i1
      		scf.condition(%condition) %arg1 : memref<i8>
   		} do {
    		^bb0(%arg2: memref<i8>):
    			%curr_chunk =  call @table_chunk_iterator_curr(%arg2) : (memref<i8>) -> memref<i8>
    			%num_rows = call @table_chunk_num_rows(%curr_chunk) : (memref<i8>) -> i64
    			%matrnr_col_chunk = call @table_chunk_get_column_buffer(%curr_chunk,%matrnr_id,%c1) : (memref<i8>,index,index) -> memref<i8>
    			%matrnr_col_offset = call @table_chunk_get_column_offset(%curr_chunk,%matrnr_id) : (memref<i8>,index) -> index
    			%vorlnr_col_chunk = call @table_chunk_get_column_buffer(%curr_chunk,%vorlnr_id,%c1) : (memref<i8>,index,index) -> memref<i8>
    			%vorlnr_col_offset = call @table_chunk_get_column_offset(%curr_chunk,%matrnr_id) : (memref<i8>,index) -> index
				%num_rows_as_idx = index_cast %num_rows : i64 to index

				scf.for %iv = %c0 to %num_rows_as_idx step %c1 {
		 			%curr_matrnr = call @table_column_get_int_64(%matrnr_col_chunk,%matrnr_col_offset,%iv) :(memref<i8>, index,index) -> !db.int<64>
		  			%curr_vorlnr = call @table_column_get_int_64(%vorlnr_col_chunk,%vorlnr_col_offset,%iv) :(memref<i8>, index,index) -> !db.int<64>
		  			db.dump %curr_matrnr : !db.int<64>
		  			db.dump %curr_vorlnr : !db.int<64>
		  			db.dump %str_const : !db.string
				}
				%next = call @table_chunk_iterator_next (%arg2) : (memref<i8>) -> memref<i8>
      			scf.yield %next: memref<i8>
    	}
		call @table_chunk_iterator_free (%table_chunk_iterator): (memref<i8>) -> ()

		return
 	}
 }