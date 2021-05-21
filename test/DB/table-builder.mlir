 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 module {
  //CHECK: int(42)
  //CHECK: int(999)
  //CHECK: string("---------------")
  //CHECK: int(999)
  //CHECK: int(42)
  //CHECK: string("---------------")
  //CHECK: int(999)
  //CHECK: int(999)
  //CHECK: string("---------------")
  //CHECK: int(42)
  //CHECK: int(42)
  //CHECK: string("---------------")
	func @main (%execution_context: memref<i8>) {
	    %str_const = db.constant ( "---------------" ) :!db.string
        %val1 = db.constant (42) : !db.int<32>
        %val2 = db.constant (999) : !db.int<32>
        %row1 = util.pack %val1,%val2 : !db.int<32>, !db.int<32> -> tuple<!db.int<32>, !db.int<32>>
        %row2 = util.pack %val2,%val1 : !db.int<32>, !db.int<32> -> tuple<!db.int<32>, !db.int<32>>
        %row3 = util.pack %val2,%val2 : !db.int<32>, !db.int<32> -> tuple<!db.int<32>, !db.int<32>>
        %row4 = util.pack %val1,%val1 : !db.int<32>, !db.int<32> -> tuple<!db.int<32>, !db.int<32>>

        %table_builder0=db.create_table_builder ["c1","c2"] : !db.table_builder<tuple<!db.int<32>,!db.int<32>>>

        %table_builder1=db.builder_merge %table_builder0 : !db.table_builder<tuple<!db.int<32>,!db.int<32>>>, %row1 : tuple<!db.int<32>, !db.int<32>> -> !db.table_builder<tuple<!db.int<32>,!db.int<32>>>
        %table_builder2=db.builder_merge %table_builder1 : !db.table_builder<tuple<!db.int<32>,!db.int<32>>>, %row2 : tuple<!db.int<32>, !db.int<32>> -> !db.table_builder<tuple<!db.int<32>,!db.int<32>>>
        %table_builder3=db.builder_merge %table_builder2 : !db.table_builder<tuple<!db.int<32>,!db.int<32>>>, %row3 : tuple<!db.int<32>, !db.int<32>> -> !db.table_builder<tuple<!db.int<32>,!db.int<32>>>
        %table_builder4=db.builder_merge %table_builder3 : !db.table_builder<tuple<!db.int<32>,!db.int<32>>>, %row4 : tuple<!db.int<32>, !db.int<32>> -> !db.table_builder<tuple<!db.int<32>,!db.int<32>>>

        %table=db.builder_build %table_builder4 : !db.table_builder<tuple<!db.int<32>,!db.int<32>>> -> !db.table
        %0 = db.tablescan %table ["c1","c2"] : !db.iterable<!db.iterable<tuple<!db.int<32>,!db.int<32>>,table_row_iterator>,table_chunk_iterator>
        db.for %table_chunk in %0 : !db.iterable<!db.iterable<tuple<!db.int<32>,!db.int<32>>,table_row_iterator>,table_chunk_iterator>{
           db.for %table_row in %table_chunk : !db.iterable<tuple<!db.int<32>,!db.int<32>>,table_row_iterator>{
               %1,%2 = util.unpack %table_row : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
               db.dump %1 : !db.int<32>
               db.dump %2 : !db.int<32>
               db.dump %str_const : !db.string
           }
        }
		return
	}
 }