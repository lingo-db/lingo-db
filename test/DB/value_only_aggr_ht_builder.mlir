 // RUN: db-run %s | FileCheck %s
 !entry_type=type tuple<tuple<>,tuple<i32,i32>>
//CHECK: int(10)
//CHECK: int(24)
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
         %zero=db.constant ( 0 ) : i32
         %one=db.constant ( 1 ) : i32

        %empty_key = util.undef_tuple : tuple<>

        %val1 = util.pack %int1, %int1 : i32,i32 -> tuple<i32,i32>
        %val2 = util.pack %int2, %int2 : i32,i32 -> tuple<i32,i32>
        %val3 = util.pack %int3, %int3 : i32,i32 -> tuple<i32,i32>
        %val4 = util.pack %int4, %int4 : i32,i32 -> tuple<i32,i32>

        %initial = util.pack %zero, %one : i32,i32 -> tuple<i32,i32>
        %ht = dsa.create_ds %initial : tuple<i32,i32> ->  !dsa.aggr_ht<tuple<>,tuple<i32,i32>>
        dsa.ht_insert %ht :  !dsa.aggr_ht<tuple<>,tuple<i32,i32>>, %empty_key : tuple<>, %val1 : tuple<i32,i32> reduce: (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
            %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
            %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
            %add1 = db.add %curr1 : i32,%new1 : i32
            %mul1 = db.mul %curr2 : i32,%new2 : i32
            %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
            dsa.yield %updated_tuple : tuple<i32, i32>
         }
         dsa.ht_insert %ht :  !dsa.aggr_ht<tuple<>,tuple<i32,i32>>, %empty_key : tuple<>, %val2 : tuple<i32,i32> reduce: (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
             %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
             %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
             %add1 = db.add %curr1 : i32,%new1 : i32
             %mul1 = db.mul %curr2 : i32,%new2 : i32
             %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
             dsa.yield %updated_tuple : tuple<i32, i32>
          }
         dsa.ht_insert %ht :  !dsa.aggr_ht<tuple<>,tuple<i32,i32>>, %empty_key : tuple<>, %val3 : tuple<i32,i32> reduce: (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
             %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
             %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
             %add1 = db.add %curr1 : i32,%new1 : i32
             %mul1 = db.mul %curr2 : i32,%new2 : i32
             %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
             dsa.yield %updated_tuple : tuple<i32, i32>
          }
          dsa.ht_insert %ht :  !dsa.aggr_ht<tuple<>,tuple<i32,i32>>, %empty_key : tuple<>, %val4 : tuple<i32,i32> reduce: (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
              %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
              %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
              %add1 = db.add %curr1 : i32,%new1 : i32
              %mul1 = db.mul %curr2 : i32,%new2 : i32
              %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
              dsa.yield %updated_tuple : tuple<i32, i32>
           }
        dsa.for %entry in %ht : !dsa.aggr_ht<tuple<>,tuple<i32,i32>> {
            %key,%val = util.unpack %entry : tuple<tuple<>,tuple<i32,i32>> -> tuple<>,tuple<i32,i32>
            %v1,%v2 = util.unpack %val : tuple<i32,i32> -> i32,i32

            db.runtime_call "DumpValue" (%v1) : (i32) -> ()
            db.runtime_call "DumpValue" (%v2) : (i32) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
        }
        return
	}
 }
