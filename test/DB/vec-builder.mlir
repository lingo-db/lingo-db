 // RUN: db-run %s %S/../../resources/data/test | FileCheck %s
 !test_table_tuple=type tuple<!db.nullable<!db.string>,!db.nullable<i1>,!db.nullable<i32>,!db.nullable<i64>,!db.nullable<f32>,!db.nullable<f64>,!db.nullable<!db.date<day>>,!db.nullable<!db.date<millisecond>>,!db.nullable<!db.decimal<5,2>>>
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
	func @main () {
	    %str_const = db.constant ( "---------------" ) :!db.string
        %vector_builder=db.create_vector_builder : !db.vector_builder<!test_table_tuple>
        %0 = db.scan_source "{ \"table\": \"test\", \"columns\": [\"str\",\"bool\",\"int32\",\"int64\",\"float32\",\"float64\",\"date32\",\"date64\",\"decimal\"] }" : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator>
        %final_builder=db.for %table_chunk in %0 : !db.iterable<!db.iterable<!test_table_tuple,table_row_iterator>,table_chunk_iterator> iter_args(%builder = %vector_builder) -> (!db.vector_builder<!test_table_tuple>){
            %builder_3 = db.for %table_row in %table_chunk : !db.iterable<!test_table_tuple,table_row_iterator> iter_args(%builder_2 = %builder) -> (!db.vector_builder<!test_table_tuple>){
               %curr_vector_builder=db.builder_merge %builder_2 : !db.vector_builder<!test_table_tuple>, %table_row : !test_table_tuple
                db.yield %curr_vector_builder : !db.vector_builder<!test_table_tuple>
            }
            db.yield %builder_3 : !db.vector_builder<!test_table_tuple>
        }
        %vector=db.builder_build %final_builder : !db.vector_builder<!test_table_tuple> -> !db.vector<!test_table_tuple>
        db.for %row in %vector : !db.vector<!test_table_tuple> {
            %1,%2,%3,%4,%5,%6,%7,%8,%9 = util.unpack %row : !test_table_tuple -> !db.nullable<!db.string>,!db.nullable<i1>,!db.nullable<i32>,!db.nullable<i64>,!db.nullable<f32>,!db.nullable<f64>,!db.nullable<!db.date<day>>,!db.nullable<!db.date<millisecond>>,!db.nullable<!db.decimal<5,2>>
            db.runtime_call "DumpValue" (%1) : (!db.nullable<!db.string>) -> ()
            db.runtime_call "DumpValue" (%2) : (!db.nullable<i1>) -> ()
            db.runtime_call "DumpValue" (%3) : (!db.nullable<i32>) -> ()
            db.runtime_call "DumpValue" (%4) : (!db.nullable<i64>) -> ()
            db.runtime_call "DumpValue" (%5) : (!db.nullable<f32>) -> ()
            db.runtime_call "DumpValue" (%6) : (!db.nullable<f64>) -> ()
            db.runtime_call "DumpValue" (%7) : (!db.nullable<!db.date<day>>) -> ()
            db.runtime_call "DumpValue" (%8) : (!db.nullable<!db.date<millisecond>>) -> ()
            db.runtime_call "DumpValue" (%9) : (!db.nullable<!db.decimal<5,2>>) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
        }
		return
	}
 }