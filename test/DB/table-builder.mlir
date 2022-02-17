 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 !test_table_tuple=type tuple<!db.string<nullable>,!db.bool<nullable>,!db.int<32,nullable>,!db.int<64,nullable>,!db.float<32,nullable>,!db.float<64,nullable>,!db.date<day,nullable>,!db.date<millisecond,nullable>,!db.decimal<5,2,nullable>>
 module {
    //CHECK: |                          str!  |                         bool!  |                        int32!  |                        int64!  |                      float32!  |                      float64!  |                       date32!  |                       date64!  |                      decimal!  |
    //CHECK: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    //CHECK: |                         "str"  |                          true  |                   -1717986918  |                           110  |                         1e-45  |                        5e-324  |                    1970-01-02  |                    1970-01-01  |                 8205408000.00  |
    //CHECK: |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |                          null  |
	func @main () -> !db.table {
	    %str_const = db.constant ( "---------------" ) :!db.string
        %table_builder=db.create_table_builder ["str!","bool!","int32!","int64!","float32!","float64!","date32!","date64!","decimal!"] : !db.table_builder<!test_table_tuple>
        %0 = db.scan_source "{ \"table\": \"test\", \"columns\": [\"str\",\"float32\",\"float64\",\"decimal\",\"int32\",\"int64\",\"bool\",\"date32\",\"date64\"] }" : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator>
        %final_builder=db.for %table_chunk in %0 : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator> iter_args(%builder = %table_builder) -> (!db.table_builder<!test_table_tuple>){
            %builder_3 = db.for %table_row in %table_chunk : !db.iterable<!test_table_tuple,table_row_iterator> iter_args(%builder_2 = %builder) -> (!db.table_builder<!test_table_tuple>){
               %curr_table_builder=db.builder_merge %builder_2 : !db.table_builder<!test_table_tuple>, %table_row : !test_table_tuple
                db.yield %curr_table_builder : !db.table_builder<!test_table_tuple>
            }
            db.yield %builder_3 : !db.table_builder<!test_table_tuple>
        }
        %table_mat=db.builder_build %final_builder : !db.table_builder<!test_table_tuple> -> !db.table
		return %table_mat : !db.table
	}
 }