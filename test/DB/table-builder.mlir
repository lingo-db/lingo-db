 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 !test_table_tuple=type tuple<!db.string<nullable>,!db.bool<nullable>,!db.int<32,nullable>,!db.int<64,nullable>,!db.float<32,nullable>,!db.float<64,nullable>,!db.date<day,nullable>,!db.date<millisecond,nullable>,!db.decimal<5,2,nullable>>
 module {
    //CHECK: string("str")
    //CHECK: bool(true)
    //CHECK: int(1)
    //CHECK: int(1)
    //CHECK: float(1.1)
    //CHECK: float(1.1)
    //CHECK: date(1996-01-02)
    //CHECK: date(1996-01-02)
    //CHECK: decimal(1.10)
    //CHECK: string("---------------")
    //CHECK: string(NULL)
    //CHECK: bool(NULL)
    //CHECK: int(NULL)
    //CHECK: int(NULL)
    //CHECK: float(NULL)
    //CHECK: float(NULL)
    //CHECK: date(NULL)
    //CHECK: date(NULL)
    //CHECK: decimal(NULL)
    //CHECK: string("---------------")
	func @main (%execution_context: memref<i8>) {
	    %str_const = db.constant ( "---------------" ) :!db.string
        %table_builder=db.create_table_builder ["str!","bool!","int32!","int64!","float32!","float64!","date32!","date64!","decimal!"] : !db.table_builder<!test_table_tuple>
        %table=db.get_table "test" %execution_context: memref<i8>
	 	%0 = db.tablescan %table ["str","bool","int32","int64","float32","float64","date32","date64","decimal"] : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator>
        %final_builder=db.for %table_chunk in %0 : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator> iter_args(%builder = %table_builder) -> (!db.table_builder<!test_table_tuple>){
            %builder_3 = db.for %table_row in %table_chunk : !db.iterable<!test_table_tuple,table_row_iterator> iter_args(%builder_2 = %builder) -> (!db.table_builder<!test_table_tuple>){
               %curr_table_builder=db.builder_merge %builder_2 : !db.table_builder<!test_table_tuple>, %table_row : !test_table_tuple -> !db.table_builder<!test_table_tuple>
                db.yield %curr_table_builder : !db.table_builder<!test_table_tuple>
            }
            db.yield %builder_3 : !db.table_builder<!test_table_tuple>
        }
        %table_mat=db.builder_build %final_builder : !db.table_builder<!test_table_tuple> -> !db.table
        %00 = db.tablescan %table_mat ["str!","bool!","int32!","int64!","float32!","float64!","date32!","date64!","decimal!"] : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator>
        db.for %table_chunk in %00 : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator>{
            db.for %table_row in %table_chunk : !db.iterable<!test_table_tuple,table_row_iterator>{
                %1,%2,%3,%4,%5,%6,%7,%8,%9 = util.unpack %table_row : !test_table_tuple -> !db.string<nullable>,!db.bool<nullable>,!db.int<32,nullable>,!db.int<64,nullable>,!db.float<32,nullable>,!db.float<64,nullable>,!db.date<day,nullable>,!db.date<millisecond,nullable>,!db.decimal<5,2,nullable>
                db.dump %1 : !db.string<nullable>
                db.dump %2 : !db.bool<nullable>
			    db.dump %3 : !db.int<32,nullable>
			    db.dump %4 : !db.int<64,nullable>
                db.dump %5 : !db.float<32,nullable>
                db.dump %6 : !db.float<64,nullable>
                db.dump %7 : !db.date<day,nullable>
                db.dump %8 : !db.date<millisecond,nullable>
                db.dump %9 : !db.decimal<5,2,nullable>
                db.dump %str_const : !db.string
            }
        }
		return
	}
 }