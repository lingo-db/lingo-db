 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 !test_table_tuple=type tuple<!db.nullable<!db.string>,!db.nullable<!db.float<32>>,!db.nullable<!db.float<64>>,!db.nullable<!db.decimal<5,2>>,!db.nullable<!db.int<32>>,!db.nullable<!db.int<64>>,!db.nullable<!db.bool>,!db.nullable<!db.date<day>>,!db.nullable<!db.date<millisecond>>>
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
	func @main () {
			%str_const = db.constant ( "---------------" ) :!db.string
            %0 = db.scan_source "{ \"table\": \"test\", \"columns\": [\"str\",\"float32\",\"float64\",\"decimal\",\"int32\",\"int64\",\"bool\",\"date32\",\"date64\"] }" : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator>
            db.for %table_chunk in %0 : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator>{
				db.for %table_row in %table_chunk : !db.iterable<!test_table_tuple,table_row_iterator>{
					%1,%2,%3,%4,%5,%6,%7,%8,%9 = util.unpack %table_row : !test_table_tuple -> !db.nullable<!db.string>,!db.nullable<!db.float<32>>,!db.nullable<!db.float<64>>,!db.nullable<!db.decimal<5,2>>,!db.nullable<!db.int<32>>,!db.nullable<!db.int<64>>,!db.nullable<!db.bool>,!db.nullable<!db.date<day>>,!db.nullable<!db.date<millisecond>>
					db.dump %1 : !db.nullable<!db.string>
					db.dump %2 : !db.nullable<!db.float<32>>
					db.dump %3 : !db.nullable<!db.float<64>>
					db.dump %4 : !db.nullable<!db.decimal<5,2>>
					db.dump %5 : !db.nullable<!db.int<32>>
					db.dump %6 : !db.nullable<!db.int<64>>
					db.dump %7 : !db.nullable<!db.bool>
					db.dump %8 : !db.nullable<!db.date<day>>
					db.dump %9 : !db.nullable<!db.date<millisecond>>

					db.dump %str_const : !db.string
				}
			}
		return
	}
 }