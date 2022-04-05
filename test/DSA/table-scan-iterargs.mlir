 // RUN: db-run-query %s %S/../../resources/data/test | FileCheck %s
 !test_table_tuple=type tuple<!db.string,f32,f64,!db.decimal<5,2>,i32,i64,i1,!db.date<day>,!db.date<millisecond>>
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
            %0 = dsa.scan_source "{ \"table\": \"test\", \"columns\": [\"str\",\"float32\",\"float64\",\"decimal\",\"int32\",\"int64\",\"bool\",\"date32\",\"date64\"] }" : !dsa.iterable<!dsa.record_batch<!test_table_tuple>,table_chunk_iterator>
            %count_0 = db.constant (0) : i32
            %one = db.constant (1) : i32
            %total_count = dsa.for %record_batch in %0 : !dsa.iterable<!dsa.record_batch<!test_table_tuple>,table_chunk_iterator>  iter_args(%count_iter = %count_0) -> (i32){
                %count = dsa.for %row in %record_batch : !dsa.record_batch<!test_table_tuple>  iter_args(%count_iter_2 = %count_iter) -> (i32) {
				    %10:2 = dsa.at %row[0] : !dsa.record<!test_table_tuple> -> !db.string, i1
				    %20:2 = dsa.at %row[1] : !dsa.record<!test_table_tuple> -> f32, i1
				    %30:2 = dsa.at %row[2] : !dsa.record<!test_table_tuple> -> f64, i1
				    %40:2 = dsa.at %row[3] : !dsa.record<!test_table_tuple> -> !db.decimal<5,2>, i1
				    %50:2 = dsa.at %row[4] : !dsa.record<!test_table_tuple> -> i32, i1
				    %60:2 = dsa.at %row[5] : !dsa.record<!test_table_tuple> -> i64, i1
				    %70:2 = dsa.at %row[6] : !dsa.record<!test_table_tuple> -> i1, i1
				    %80:2 = dsa.at %row[7] : !dsa.record<!test_table_tuple> -> !db.date<day>, i1
				    %90:2 = dsa.at %row[8] : !dsa.record<!test_table_tuple> ->  !db.date<millisecond>, i1
				    %100= db.not %10#1 : i1
				    %200= db.not %20#1 : i1
				    %300= db.not %30#1 : i1
				    %400= db.not %40#1 : i1
				    %500= db.not %50#1 : i1
				    %600= db.not %60#1 : i1
				    %700= db.not %70#1 : i1
				    %800= db.not %80#1 : i1
				    %900= db.not %90#1 : i1
                    %1 = db.as_nullable %10#0 :!db.string, %100 -> !db.nullable<!db.string>
                    %2 = db.as_nullable %20#0 : f32, %200 -> !db.nullable<f32>
                    %3 = db.as_nullable %30#0 : f64, %300 -> !db.nullable<f64>
                    %4 = db.as_nullable %40#0 : !db.decimal<5,2> , %400 -> !db.nullable<!db.decimal<5,2>>
                    %5 = db.as_nullable %50#0 : i32, %500 -> !db.nullable<i32>
                    %6 = db.as_nullable %60#0 : i64, %600 -> !db.nullable<i64>
                    %7 = db.as_nullable %70#0 : i1, %700 -> !db.nullable<i1>
                    %8 = db.as_nullable %80#0 :!db.date<day>, %800 -> !db.nullable<!db.date<day>>
                    %9 = db.as_nullable %90#0 :!db.date<millisecond>, %900 -> !db.nullable<!db.date<millisecond>>
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
					%curr_count=db.add %count_iter_2 : i32, %one : i32
					dsa.yield %curr_count : i32
				}
				%curr_count=db.add %count_iter : i32, %count : i32
                dsa.yield %curr_count : i32
			}
			db.runtime_call "DumpValue" (%total_count) : (i32) -> ()
		return
	}
 }