 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 !test_table_tuple=type tuple<!db.nullable<!db.string>,!db.nullable<f32>,!db.nullable<f64>,!db.nullable<!db.decimal<5,2>>,!db.nullable<i32>,!db.nullable<i64>,!db.nullable<i1>,!db.nullable<!db.date<day>>,!db.nullable<!db.date<millisecond>>>
 module {
    //CHECK: |                          str!  |                      float32!  |                      float64!  |                      decimal!  |                        int32!  |                        int64! |                          bool  |                       date32!  |                       date64!  |
    //CHECK: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //CHECK: |                         "str"  |                           1.1  |                           1.1  |                          1.10  |                             1  |                             1  |                          true  |                    1996-01-02  |                    1996-01-02  |
    //CHECK: |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |
	func @main () -> !db.table {
	    %str_const = db.constant ( "---------------" ) :!db.string
        %table_builder=db.create_table_builder ["str!","float32!","float64!","decimal!","int32!","int64!","bool","date32!","date64!"] : !db.table_builder<!test_table_tuple>
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
                %packed = util.pack %1,%2,%3,%4,%5,%6,%7,%8,%9 : !db.nullable<!db.string>,!db.nullable<f32>,!db.nullable<f64>,!db.nullable<!db.decimal<5,2>>,!db.nullable<i32>,!db.nullable<i64>,!db.nullable<i1>,!db.nullable<!db.date<day>>,!db.nullable<!db.date<millisecond>> -> !test_table_tuple
                db.add_row %table_builder : !db.table_builder<!test_table_tuple>, %packed : !test_table_tuple
                db.yield
            }
            db.yield
        }
        %table_mat=db.finalize_table %table_builder : !db.table_builder<!test_table_tuple>
		return %table_mat : !db.table
	}
 }