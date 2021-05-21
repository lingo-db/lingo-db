 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 module {
//CHECK: int(42)
//CHECK: string("str")
//CHECK: bool(true)
//CHECK: string("---------------")
//CHECK: int(42)
//CHECK: string("str")
//CHECK: bool(true)
//CHECK: string("---------------")
//CHECK: int(42)
//CHECK: string("str")
//CHECK: bool(false)
//CHECK: string("---------------")
//CHECK: int(42)
//CHECK: string("str2")
//CHECK: bool(false)
//CHECK: string("---------------")
	func @main (%execution_context: memref<i8>) {
	    %str_const = db.constant ( "---------------" ) :!db.string
        %val1 = db.constant (42) : !db.int<32>
        %val2 = db.constant ("str") : !db.string
        %val21 = db.constant ("str2") : !db.string
        %val3 = db.constant (1) : !db.bool
        %val31 = db.constant (0) : !db.bool

        %row1 = util.pack %val1,%val2,%val3 : !db.int<32>, !db.string, !db.bool -> tuple<!db.int<32>,!db.string,!db.bool>
        %row2 = util.pack %val1,%val2,%val3 : !db.int<32>, !db.string, !db.bool -> tuple<!db.int<32>,!db.string,!db.bool>
        %row3 = util.pack %val1,%val2,%val31 : !db.int<32>, !db.string, !db.bool -> tuple<!db.int<32>,!db.string,!db.bool>
        %row4 = util.pack %val1,%val21,%val31 : !db.int<32>, !db.string, !db.bool -> tuple<!db.int<32>,!db.string,!db.bool>

        %table_builder0=db.create_table_builder ["c1","c2","c3"] : !db.table_builder<tuple<!db.int<32>,!db.string,!db.bool>>

        %table_builder1=db.builder_merge %table_builder0 : !db.table_builder<tuple<!db.int<32>,!db.string,!db.bool>>, %row1 : tuple<!db.int<32>,!db.string,!db.bool> -> !db.table_builder<tuple<!db.int<32>,!db.string,!db.bool>>
        %table_builder2=db.builder_merge %table_builder1 : !db.table_builder<tuple<!db.int<32>,!db.string,!db.bool>>, %row2 : tuple<!db.int<32>,!db.string,!db.bool> -> !db.table_builder<tuple<!db.int<32>,!db.string,!db.bool>>
        %table_builder3=db.builder_merge %table_builder2 : !db.table_builder<tuple<!db.int<32>,!db.string,!db.bool>>, %row3 : tuple<!db.int<32>,!db.string,!db.bool> -> !db.table_builder<tuple<!db.int<32>,!db.string,!db.bool>>
        %table_builder4=db.builder_merge %table_builder3 : !db.table_builder<tuple<!db.int<32>,!db.string,!db.bool>>, %row4 : tuple<!db.int<32>,!db.string,!db.bool> -> !db.table_builder<tuple<!db.int<32>,!db.string,!db.bool>>

        %table=db.builder_build %table_builder4 : !db.table_builder<tuple<!db.int<32>,!db.string,!db.bool>> -> !db.table
        %0 = db.tablescan %table ["c1","c2","c3"] : !db.iterable<!db.iterable<tuple<!db.int<32>,!db.string,!db.bool>,table_row_iterator>,table_chunk_iterator>
        db.for %table_chunk in %0 : !db.iterable<!db.iterable<tuple<!db.int<32>,!db.string,!db.bool>,table_row_iterator>,table_chunk_iterator>{
           db.for %table_row in %table_chunk : !db.iterable<tuple<!db.int<32>,!db.string,!db.bool>,table_row_iterator>{
               %1,%2,%3 = util.unpack %table_row : tuple<!db.int<32>,!db.string,!db.bool> -> !db.int<32>,!db.string,!db.bool
               db.dump %1 : !db.int<32>
               db.dump %2 : !db.string
               db.dump %3 : !db.bool
               db.dump %str_const : !db.string
           }
        }
		return
	}
 }