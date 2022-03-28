 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 !test_table_tuple=type tuple<!db.nullable<!db.string>,!db.nullable<f32>,!db.nullable<f64>,!db.nullable<!db.decimal<5,2>>,!db.nullable<i32>,!db.nullable<i64>,!db.nullable<i1>,!db.nullable<!db.date<day>>,!db.nullable<!db.date<millisecond>>>
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
            %0 = db.scan_source "{ \"table\": \"test\", \"columns\": [\"str\",\"float32\",\"float64\",\"decimal\",\"int32\",\"int64\",\"bool\",\"date32\",\"date64\"] }" : !db.iterable<!db.record_batch<!test_table_tuple>,table_chunk_iterator>
            db.for %record_batch in %0 : !db.iterable<!db.record_batch<!test_table_tuple>,table_chunk_iterator>{
				db.for %row in %record_batch : !db.record_batch<!test_table_tuple>{
				    %1 = db.at %row[0] : !db.record<!test_table_tuple> -> !db.nullable<!db.string>
				    %2 = db.at %row[1] : !db.record<!test_table_tuple> -> !db.nullable<f32>
				    %3 = db.at %row[2] : !db.record<!test_table_tuple> -> !db.nullable<f64>
				    %4 = db.at %row[3] : !db.record<!test_table_tuple> -> !db.nullable<!db.decimal<5,2>>
				    %5 = db.at %row[4] : !db.record<!test_table_tuple> -> !db.nullable<i32>
				    %6 = db.at %row[5] : !db.record<!test_table_tuple> -> !db.nullable<i64>
				    %7 = db.at %row[6] : !db.record<!test_table_tuple> -> !db.nullable<i1>
				    %8 = db.at %row[7] : !db.record<!test_table_tuple> -> !db.nullable<!db.date<day>>
				    %9 = db.at %row[8] : !db.record<!test_table_tuple> ->  !db.nullable<!db.date<millisecond>>

            		db.runtime_call "DumpValue" (%1) : (!db.nullable<!db.string>) -> ()
					db.runtime_call "DumpValue" (%2) : (!db.nullable<f32>) -> ()
					db.runtime_call "DumpValue" (%3) : (!db.nullable<f64>) -> ()
					db.runtime_call "DumpValue" (%4) : (!db.nullable<!db.decimal<5,2>>) -> ()
					db.runtime_call "DumpValue" (%5) : (!db.nullable<i32>) -> ()
					db.runtime_call "DumpValue" (%6) : (!db.nullable<i64>) -> ()
					db.runtime_call "DumpValue" (%7) : (!db.nullable<i1>) -> ()
					db.runtime_call "DumpValue" (%8) : (!db.nullable<!db.date<day>>) -> ()
					db.runtime_call "DumpValue" (%9) : (!db.nullable<!db.date<millisecond>>) -> ()

					db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
				}
			}
		return
	}
 }