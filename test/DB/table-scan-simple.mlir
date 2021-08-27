 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 !test_table_tuple=type tuple<!db.string<nullable>,!db.float<32,nullable>,!db.float<64,nullable>,!db.decimal<5,2,nullable>,!db.int<32,nullable>,!db.int<64,nullable>,!db.bool<nullable>,!db.date<day,nullable>,!db.date<millisecond,nullable>>
 module {
    //CHECK: string("str")
    //CHECK: float(1.1)
    //CHECK: float(1.1)
    //CHECK: decimal(1.10)
    //CHECK: int(1)
    //CHECK: int(1)
    //CHECK: bool(true)
    //CHECK: date(1996-01-02)
    //CHECK: date(1996-01-02)
    //CHECK: string("---------------")
    //CHECK: string(NULL)
    //CHECK: float(NULL)
    //CHECK: float(NULL)
    //CHECK: decimal(NULL)
    //CHECK: int(NULL)
    //CHECK: int(NULL)
    //CHECK: bool(NULL)
    //CHECK: date(NULL)
    //CHECK: date(NULL)
    //CHECK: string("---------------")
	func @main (%execution_context: !util.generic_memref<i8>) {
			%str_const = db.constant ( "---------------" ) :!db.string
	 		%table=db.get_table "test"
            %0 = db.tablescan %table ["str","float32","float64","decimal","int32","int64","bool","date32","date64"] : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator>
            db.for %table_chunk in %0 : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator>{
				db.for %table_row in %table_chunk : !db.iterable<!test_table_tuple,table_row_iterator>{
					%1,%2,%3,%4,%5,%6,%7,%8,%9 = util.unpack %table_row : !test_table_tuple -> !db.string<nullable>,!db.float<32,nullable>,!db.float<64,nullable>,!db.decimal<5,2,nullable>,!db.int<32,nullable>,!db.int<64,nullable>,!db.bool<nullable>,!db.date<day,nullable>,!db.date<millisecond,nullable>
					db.dump %1 : !db.string<nullable>
					db.dump %2 : !db.float<32,nullable>
					db.dump %3 : !db.float<64,nullable>
					db.dump %4 : !db.decimal<5,2,nullable>
					db.dump %5 : !db.int<32,nullable>
					db.dump %6 : !db.int<64,nullable>
					db.dump %7 : !db.bool<nullable>
					db.dump %8 : !db.date<day,nullable>
					db.dump %9 : !db.date<millisecond,nullable>

					db.dump %str_const : !db.string
				}
			}
		return
	}
 }