 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 !test_table_tuple=type tuple<!db.string,f32,f64,!db.decimal<5,2>,i32,i64,i1,!db.date<day>,!db.date<millisecond>>
!test_table_tuple2=type tuple<!db.nullable<!db.string>,!db.nullable<f32>,!db.nullable<f64>,!db.nullable<!db.decimal<5,2>>,!db.nullable<i32>,!db.nullable<i64>,!db.nullable<i1>,!db.nullable<!db.date<day>>,!db.nullable<!db.date<millisecond>>>

 module {
    //CHECK: |                          str!  |                      float32!  |                      float64!  |                      decimal!  |                        int32!  |                        int64! |                          bool  |                       date32!  |                       date64!  |
    //CHECK: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //CHECK: |                         "str"  |                           1.1  |                           1.1  |                          1.10  |                             1  |                             1  |                          true  |                    1996-01-02  |                    1996-01-02  |
    //CHECK: |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |
	func @main () -> !db.table {
	    %str_const = db.constant ( "---------------" ) :!db.string
        %table_builder=db.create_table_builder ["str!","float32!","float64!","decimal!","int32!","int64!","bool","date32!","date64!"] : !db.table_builder<!test_table_tuple2>
        %0 = db.scan_source "{ \"table\": \"test\", \"columns\": [\"str\",\"float32\",\"float64\",\"decimal\",\"int32\",\"int64\",\"bool\",\"date32\",\"date64\"] }" : !db.iterable<!db.record_batch<!test_table_tuple>,table_chunk_iterator>
        db.for %record_batch in %0 : !db.iterable<!db.record_batch<!test_table_tuple>,table_chunk_iterator>{
            db.for %row in %record_batch : !db.record_batch<!test_table_tuple>{
                %10:2 = db.at %row[0] : !db.record<!test_table_tuple> -> !db.string, i1
                %20:2 = db.at %row[1] : !db.record<!test_table_tuple> -> f32, i1
                %30:2 = db.at %row[2] : !db.record<!test_table_tuple> -> f64, i1
                %40:2 = db.at %row[3] : !db.record<!test_table_tuple> -> !db.decimal<5,2>, i1
                %50:2 = db.at %row[4] : !db.record<!test_table_tuple> -> i32, i1
                %60:2 = db.at %row[5] : !db.record<!test_table_tuple> -> i64, i1
                %70:2 = db.at %row[6] : !db.record<!test_table_tuple> -> i1, i1
                %80:2 = db.at %row[7] : !db.record<!test_table_tuple> -> !db.date<day>, i1
                %90:2 = db.at %row[8] : !db.record<!test_table_tuple> ->  !db.date<millisecond>, i1
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

                %packed = util.pack %1,%2,%3,%4,%5,%6,%7,%8,%9 : !db.nullable<!db.string>,!db.nullable<f32>,!db.nullable<f64>,!db.nullable<!db.decimal<5,2>>,!db.nullable<i32>,!db.nullable<i64>,!db.nullable<i1>,!db.nullable<!db.date<day>>,!db.nullable<!db.date<millisecond>> -> !test_table_tuple2
                db.add_row %table_builder : !db.table_builder<!test_table_tuple2>, %packed : !test_table_tuple2
                db.yield
            }
            db.yield
        }
        %table_mat=db.finalize_table %table_builder : !db.table_builder<!test_table_tuple2>
		return %table_mat : !db.table
	}
 }