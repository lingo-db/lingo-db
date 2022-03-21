 // RUN: db-run %s | FileCheck %s
 !test_tuple_type=type tuple<!db.string,i32>
 !test_tuple_raw=type tuple<tuple<i64,i64>,i32>
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: string("---------------")
//CHECK: string("strb")
//CHECK: int(2)
//CHECK: string("---------------")
//CHECK: string("strc")
//CHECK: int(3)
//CHECK: string("---------------")
//CHECK: string("strd")
//CHECK: int(1)
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("strd")
//CHECK: int(1)
//CHECK: string("---------------")
//CHECK: string("strc")
//CHECK: int(3)
//CHECK: string("---------------")
//CHECK: string("strb")
//CHECK: int(2)
//CHECK: string("---------------")
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: string("---------------")

 module {

	func @main () {
         %str_const = db.constant ( "---------------" ) :!db.string

         %str1=db.constant ( "stra" ) :!db.string
         %str2=db.constant ( "strb" ) :!db.string
         %str3=db.constant ( "strc" ) :!db.string
         %str4=db.constant ( "strd" ) :!db.string
         %int1=db.constant ( 4 ) : i32
         %int2=db.constant ( 2 ) : i32
         %int3=db.constant ( 3 ) : i32
         %int4=db.constant ( 1 ) : i32

        %row1 = util.pack %str1, %int1 : !db.string,i32 -> tuple<!db.string,i32>
        %row2 = util.pack %str2, %int2 : !db.string,i32 -> tuple<!db.string,i32>
        %row3 = util.pack %str3, %int3 : !db.string,i32 -> tuple<!db.string,i32>
        %row4 = util.pack %str4, %int4 : !db.string,i32 -> tuple<!db.string,i32>

        %vector_builder=db.create_vector_builder : !db.vector_builder<!test_tuple_type>
        %builder1=db.builder_merge %vector_builder : !db.vector_builder<!test_tuple_type>, %row1 : !test_tuple_type
        %builder2=db.builder_merge %builder1 : !db.vector_builder<!test_tuple_type>, %row2 : !test_tuple_type
        %builder3=db.builder_merge %builder2 : !db.vector_builder<!test_tuple_type>, %row3 : !test_tuple_type
        %builder4=db.builder_merge %builder3 : !db.vector_builder<!test_tuple_type>, %row4 : !test_tuple_type
        %vector=db.builder_build %builder4 : !db.vector_builder<!test_tuple_type> -> !db.vector<!test_tuple_type>

        db.for %row in %vector : !db.vector<!test_tuple_type> {
            %1,%2 = util.unpack %row : !test_tuple_type -> !db.string,i32
            db.runtime_call "DumpValue" (%1) : (!db.string) -> ()
            db.runtime_call "DumpValue" (%2) : (i32) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
        }

        db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
        db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()

        db.sort %vector : !db.vector<!test_tuple_type> (%left,%right) {
           %left1,%left2 = util.unpack %left : !test_tuple_type -> !db.string,i32
           %right1,%right2 = util.unpack %right : !test_tuple_type -> !db.string,i32
           %lt = db.compare gte %left1 : !db.string, %right1 : !db.string
           db.yield %lt : i1
        }

        db.for %row in %vector : !db.vector<!test_tuple_type> {
            %1,%2 = util.unpack %row : !test_tuple_type -> !db.string,i32
            db.runtime_call "DumpValue" (%1) : (!db.string) -> ()
            db.runtime_call "DumpValue" (%2) : (i32) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
        }
        return
	}
 }
