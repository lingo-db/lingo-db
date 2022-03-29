// RUN: db-run %s | FileCheck %s
!test_tuple_type=type tuple<!db.string,i32>
!test_tuple_raw=type tuple<tuple<i64,i64>,i32>
//CHECK: string("stra")
//CHECK: int(1)
//CHECK: bool(false)
//CHECK: string("---------------")
//CHECK: string("strb")
//CHECK: int(2)
//CHECK: bool(false)
//CHECK: string("---------------")
//CHECK: string("strc")
//CHECK: int(3)
//CHECK: bool(true)
//CHECK: string("---------------")

module {

    func @main () {
          %str_const = db.constant ( "---------------" ) :!db.string

          %str1=db.constant ( "stra" ) :!db.string
          %str2=db.constant ( "strb" ) :!db.string
          %str3=db.constant ( "strc" ) :!db.string
          %str4=db.constant ( "strd" ) :!db.string
          %int1=db.constant ( 1 ) : i32
          %int2=db.constant ( 2 ) : i32
          %int3=db.constant ( 3 ) : i32
          %int4=db.constant ( 4 ) : i32

         %row1 = util.pack %str1, %int1 : !db.string,i32 -> tuple<!db.string,i32>
         %row2 = util.pack %str2, %int2 : !db.string,i32 -> tuple<!db.string,i32>
         %row3 = util.pack %str3, %int3 : !db.string,i32 -> tuple<!db.string,i32>
         %row4 = util.pack %str4, %int4 : !db.string,i32 -> tuple<!db.string,i32>

       	    %vector = dsa.create_ds !dsa.vector<!test_tuple_type>
           dsa.ds_append %vector : !dsa.vector<!test_tuple_type> , %row1 : !test_tuple_type
           dsa.ds_append %vector : !dsa.vector<!test_tuple_type> , %row2 : !test_tuple_type
           dsa.ds_append %vector : !dsa.vector<!test_tuple_type> , %row3 : !test_tuple_type
           dsa.ds_append %vector : !dsa.vector<!test_tuple_type> , %row4 : !test_tuple_type
         %flag = dsa.createflag
         dsa.for %row in %vector : !dsa.vector<!test_tuple_type> until %flag {
             %1,%2 = util.unpack %row : !test_tuple_type -> !db.string,i32
             db.runtime_call "DumpValue" (%1) : (!db.string) -> ()
             db.runtime_call "DumpValue" (%2) : (i32) -> ()
             %cmp = db.compare gte %2 : i32, %int3 : i32
             db.runtime_call "DumpValue" (%cmp) : (i1) -> ()
             dsa.setflag %flag, %cmp
             db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
         }
         return
    }
}
