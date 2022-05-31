 // RUN: run-mlir %s %S/../../../resources/data/test | FileCheck %s
 !test_table_tuple=type tuple<!db.string,f32,f64,!db.decimal<5,2>,i32,i64,i1,!db.date<day>,!db.date<millisecond>>

 module {
    //CHECK: |                          str!  |                      float32!  |                      float64!  |                      decimal!  |                        int32!  |                        int64! |                          bool  |                       date32!  |                       date64!  |
    //CHECK: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //CHECK: |                         "str"  |                           1.1  |                           1.1  |                          1.10  |                             1  |                             1  |                          true  |                    1996-01-02  |                    1996-01-02  |
    //CHECK: |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |
	func.func @main () -> !dsa.table {
	    %str_const = db.constant ( "---------------" ) :!db.string
        %table_builder=dsa.create_ds ("str!:string; float32!:float[32]; float64!:float[64]; decimal!:decimal[5,2]; int32!:int[32]; int64!:int[64]; bool:bool; date32!:date[32]; date64!:date[64]") -> !dsa.table_builder<!test_table_tuple>
        %0 = dsa.scan_source "{ \"table\": \"test\", \"columns\": [\"str\",\"float32\",\"float64\",\"decimal\",\"int32\",\"int64\",\"bool\",\"date32\",\"date64\"] }" : !dsa.iterable<!dsa.record_batch<!test_table_tuple>,table_chunk_iterator>
        dsa.for %record_batch in %0 : !dsa.iterable<!dsa.record_batch<!test_table_tuple>,table_chunk_iterator>{
            dsa.for %row in %record_batch : !dsa.record_batch<!test_table_tuple>{
                %10:2 = dsa.at %row[0] : !dsa.record<!test_table_tuple> -> !db.string, i1
                %20:2 = dsa.at %row[1] : !dsa.record<!test_table_tuple> -> f32, i1
                %30:2 = dsa.at %row[2] : !dsa.record<!test_table_tuple> -> f64, i1
                %40:2 = dsa.at %row[3] : !dsa.record<!test_table_tuple> -> !db.decimal<5,2>, i1
                %50:2 = dsa.at %row[4] : !dsa.record<!test_table_tuple> -> i32, i1
                %60:2 = dsa.at %row[5] : !dsa.record<!test_table_tuple> -> i64, i1
                %70:2 = dsa.at %row[6] : !dsa.record<!test_table_tuple> -> i1, i1
                %80:2 = dsa.at %row[7] : !dsa.record<!test_table_tuple> -> !db.date<day>, i1
                %90:2 = dsa.at %row[8] : !dsa.record<!test_table_tuple> ->  !db.date<millisecond>, i1
                dsa.ds_append %table_builder : !dsa.table_builder<!test_table_tuple>, %10#0 :!db.string, %10#1
                dsa.ds_append %table_builder : !dsa.table_builder<!test_table_tuple>, %20#0 : f32, %20#1
                dsa.ds_append %table_builder : !dsa.table_builder<!test_table_tuple>, %30#0 : f64, %30#1
                dsa.ds_append %table_builder : !dsa.table_builder<!test_table_tuple>, %40#0 : !db.decimal<5,2> , %40#1
                dsa.ds_append %table_builder : !dsa.table_builder<!test_table_tuple>, %50#0 : i32, %50#1
                dsa.ds_append %table_builder : !dsa.table_builder<!test_table_tuple>, %60#0 : i64, %60#1
                dsa.ds_append %table_builder : !dsa.table_builder<!test_table_tuple>, %70#0 : i1, %70#1
                dsa.ds_append %table_builder : !dsa.table_builder<!test_table_tuple>, %80#0 :!db.date<day>, %80#1
                dsa.ds_append %table_builder : !dsa.table_builder<!test_table_tuple>, %90#0 :!db.date<millisecond>, %90#1

                dsa.next_row %table_builder : !dsa.table_builder<!test_table_tuple>
            }
        }
        %table_mat=dsa.finalize %table_builder : !dsa.table_builder<!test_table_tuple> -> !dsa.table
		return %table_mat : !dsa.table
	}
 }