 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 module {
     //CHECK: int(42)
     //CHECK: string("---------------")
     //CHECK: int(999)
     //CHECK: string("---------------")
    func private @arrow_type(i32) -> memref<i8> attributes {llvm.emit_c_interface}
    func private @arrow_schema_create_builder() -> memref<i8> attributes {llvm.emit_c_interface}
    func private @arrow_schema_add_field(memref<i8>,memref<i8>,!db.string) -> () attributes {llvm.emit_c_interface}
    func private @arrow_schema_build(memref<i8>) -> memref<i8> attributes {llvm.emit_c_interface}
    func private @arrow_table_build(memref<i8>,memref<i8>) -> !db.table attributes {llvm.emit_c_interface}
	func private @empty_resizeable_buffer() -> memref<i8>  attributes {llvm.emit_c_interface}
	func private @create_resizeable_buffer(i64) -> memref<i8>  attributes {llvm.emit_c_interface}
	func private @create_column_data(memref<i8>,memref<i8>,memref<i8>,memref<i8>,i64) -> memref<i8>  attributes {llvm.emit_c_interface}
	func private @create_batch_builder(memref<i8>,i64) -> memref<i8>  attributes {llvm.emit_c_interface}
	func private @batch_builder_add_column_data(memref<i8>,memref<i8>) -> ()  attributes {llvm.emit_c_interface}
	func private @batch_builder_build(memref<i8>) -> memref<i8>  attributes {llvm.emit_c_interface}
	func private @resizeable_buffer_get(memref<i8>) -> !util.generic_memref<?x!db.int<32>>  attributes {llvm.emit_c_interface}

	func @main (%execution_context: memref<i8>) {
        %c7 = constant 7 : i32
        %c2 = constant 2 : i64
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %val1 = db.constant (42) : !db.int<32>
        %val2 = db.constant (999) : !db.int<32>

	    %arrow_i32_type = call @arrow_type(%c7) : (i32) -> memref<i8>
	    %schema_builder = call @arrow_schema_create_builder() : () -> memref<i8>
	    %column_name = db.constant ("testcolumn") : !db.string
	    call @arrow_schema_add_field (%schema_builder,%arrow_i32_type,%column_name): (memref<i8>,memref<i8>,!db.string) -> ()
        %schema = call @arrow_schema_build(%schema_builder):(memref<i8>) -> memref<i8>

        %bitmap_buffer=call @empty_resizeable_buffer():()->memref<i8>
        %val_buffer=call @create_resizeable_buffer(%c2):(i64)->memref<i8>
        %var_buffer=call @empty_resizeable_buffer():()->memref<i8>
        %raw_buffer=call @resizeable_buffer_get(%val_buffer):(memref<i8>) -> !util.generic_memref<?x!db.int<32>>
        util.store %val1:!db.int<32>,%raw_buffer[%c0] :!util.generic_memref<?x!db.int<32>>
        util.store %val2:!db.int<32>,%raw_buffer[%c1] :!util.generic_memref<?x!db.int<32>>

        //todo: insert real data ;)
        %column_data= call @create_column_data(%arrow_i32_type,%bitmap_buffer,%val_buffer,%var_buffer,%c2) : (memref<i8>,memref<i8>,memref<i8>,memref<i8>,i64) -> memref<i8>
        %batch_builder = call @create_batch_builder(%schema,%c2) : (memref<i8>,i64) -> memref<i8>
        call @batch_builder_add_column_data(%batch_builder,%column_data) : (memref<i8>,memref<i8>) -> ()
        %batch = call @batch_builder_build(%batch_builder) : (memref<i8>)->memref<i8>
        %table_name = db.constant ("testtable") : !db.string
        %table = call @arrow_table_build(%schema,%batch) :(memref<i8>,memref<i8>) -> !db.table


        %str_const = db.constant ( "---------------" ) :!db.string
         %0 = db.tablescan %table ["testcolumn"] : !db.iterable<!db.iterable<tuple<!db.int<32>>,table_row_iterator>,table_chunk_iterator>

         db.for %table_chunk in %0 : !db.iterable<!db.iterable<tuple<!db.int<32>>,table_row_iterator>,table_chunk_iterator>{
            db.for %table_row in %table_chunk : !db.iterable<tuple<!db.int<32>>,table_row_iterator>{
                %1 = util.unpack %table_row : tuple<!db.int<32>> -> !db.int<32>

                db.dump %1 : !db.int<32>
                db.dump %str_const : !db.string
            }
        }
		return
	}
 }