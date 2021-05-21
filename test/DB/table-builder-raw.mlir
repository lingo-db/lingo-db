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
    func private @arrow_create_table_builder(memref<i8>) -> memref<i8> attributes {llvm.emit_c_interface}
    func private @table_builder_finish_row(memref<i8>) -> () attributes {llvm.emit_c_interface}
    func private @table_builder_build(memref<i8>) -> !db.table attributes {llvm.emit_c_interface}
    func private @table_builder_add_int_32(memref<i8>,i32,i1,i32) -> () attributes {llvm.emit_c_interface}

	func @main (%execution_context: memref<i8>) {
        %c7 = constant 7 : i32
        %c2 = constant 2 : i64
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %val1 = constant 42 : i32
        %val2 = constant 999 : i32
        %false = constant 0: i1
        %zero = constant 0: i32

	    %arrow_i32_type = call @arrow_type(%c7) : (i32) -> memref<i8>
	    %schema_builder = call @arrow_schema_create_builder() : () -> memref<i8>
	    %column_name = db.constant ("testcolumn") : !db.string
	    call @arrow_schema_add_field (%schema_builder,%arrow_i32_type,%column_name): (memref<i8>,memref<i8>,!db.string) -> ()
        %schema = call @arrow_schema_build(%schema_builder):(memref<i8>) -> memref<i8>
        %table_builder=call @arrow_create_table_builder(%schema) : (memref<i8>) -> memref<i8>
        call @table_builder_add_int_32(%table_builder,%zero,%false,%val1):(memref<i8>,i32,i1,i32) -> ()
        call @table_builder_finish_row(%table_builder):(memref<i8>) -> ()
        call @table_builder_add_int_32(%table_builder,%zero,%false,%val2):(memref<i8>,i32,i1,i32) -> ()
        call @table_builder_finish_row(%table_builder):(memref<i8>) -> ()
        %table=call @table_builder_build(%table_builder) : (memref<i8>) -> !db.table

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