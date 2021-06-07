 // RUN: db-run %s | FileCheck %s
 !test_tuple_type=type tuple<!db.string,!db.int<32>>
 !test_tuple_raw=type tuple<tuple<i64,i64>,!db.int<32>>
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

	func @main (%execution_context: memref<i8>) {
         %str_const = db.constant ( "---------------" ) :!db.string

         %str1=db.constant ( "stra" ) :!db.string
         %str2=db.constant ( "strb" ) :!db.string
         %str3=db.constant ( "strc" ) :!db.string
         %str4=db.constant ( "strd" ) :!db.string
         %int1=db.constant ( 4 ) : !db.int<32>
         %int2=db.constant ( 2 ) : !db.int<32>
         %int3=db.constant ( 3 ) : !db.int<32>
         %int4=db.constant ( 1 ) : !db.int<32>

        %row1 = util.pack %str1, %int1 : !db.string,!db.int<32> -> tuple<!db.string,!db.int<32>>
        %row2 = util.pack %str2, %int2 : !db.string,!db.int<32> -> tuple<!db.string,!db.int<32>>
        %row3 = util.pack %str3, %int3 : !db.string,!db.int<32> -> tuple<!db.string,!db.int<32>>
        %row4 = util.pack %str4, %int4 : !db.string,!db.int<32> -> tuple<!db.string,!db.int<32>>

        %vector_builder=db.create_vector_builder : !db.vector_builder<!test_tuple_type>
        %builder1=db.builder_merge %vector_builder : !db.vector_builder<!test_tuple_type>, %row1 : !test_tuple_type -> !db.vector_builder<!test_tuple_type>
        %builder2=db.builder_merge %builder1 : !db.vector_builder<!test_tuple_type>, %row2 : !test_tuple_type -> !db.vector_builder<!test_tuple_type>
        %builder3=db.builder_merge %builder2 : !db.vector_builder<!test_tuple_type>, %row3 : !test_tuple_type -> !db.vector_builder<!test_tuple_type>
        %builder4=db.builder_merge %builder3 : !db.vector_builder<!test_tuple_type>, %row4 : !test_tuple_type -> !db.vector_builder<!test_tuple_type>
        %vector=db.builder_build %builder4 : !db.vector_builder<!test_tuple_type> -> !db.vector<!test_tuple_type>

        db.for %row in %vector : !db.vector<!test_tuple_type> {
            %1,%2 = util.unpack %row : !test_tuple_type -> !db.string,!db.int<32>
            db.dump %1 : !db.string
            db.dump %2 : !db.int<32>
            db.dump %str_const : !db.string
        }

        db.dump %str_const : !db.string
        db.dump %str_const : !db.string

        %sorted_vec=db.sort %vector : !db.vector<!test_tuple_type> (%left,%right) {
           %left1,%left2 = util.unpack %left : !test_tuple_type -> !db.string,!db.int<32>
           %right1,%right2 = util.unpack %right : !test_tuple_type -> !db.string,!db.int<32>
           %lt = db.compare gte %left1 : !db.string, %right1 : !db.string
           db.yield %lt : !db.bool
        }

        db.for %row in %sorted_vec : !db.vector<!test_tuple_type> {
            %1,%2 = util.unpack %row : !test_tuple_type -> !db.string,!db.int<32>
            db.dump %1 : !db.string
            db.dump %2 : !db.int<32>
            db.dump %str_const : !db.string
        }
        return
	}
 }