 // RUN: db-run-query %s | FileCheck %s
 !entry_type=type tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(6)
//CHECK: int(8)
//CHECK: string("---------------")
//CHECK: string("strc")
//CHECK: int(3)
//CHECK: int(3)
//CHECK: int(3)
//CHECK: string("---------------")
//CHECK: string("strd")
//CHECK: int(1)
//CHECK: int(1)
//CHECK: int(1)
//CHECK: string("---------------")
 module {

	func @main () {
         %str_const = db.constant ( "---------------" ) :!db.string

         %str1c=db.constant ( "stra" ) :!db.string
         %str2c=db.constant ( "strb" ) :!db.string
         %str3c=db.constant ( "strc" ) :!db.string
         %str4c=db.constant ( "strd" ) :!db.string
         %str1 = db.as_nullable %str1c  : !db.string -> !db.nullable<!db.string>
         %str2 = db.as_nullable %str2c  : !db.string -> !db.nullable<!db.string>
         %str3 = db.as_nullable %str3c  : !db.string -> !db.nullable<!db.string>
         %str4 = db.as_nullable %str4c  : !db.string -> !db.nullable<!db.string>
         %int1=db.constant ( 4 ) : i32
         %int2=db.constant ( 2 ) : i32
         %int3=db.constant ( 3 ) : i32
         %int4=db.constant ( 1 ) : i32
         %zero=db.constant ( 0 ) : i32
         %one=db.constant ( 1 ) : i32

        %key1 = util.pack %str1, %int1 : !db.nullable<!db.string>,i32 -> tuple<!db.nullable<!db.string>,i32>
        %key2 = util.pack %str1, %int1 : !db.nullable<!db.string>,i32 -> tuple<!db.nullable<!db.string>,i32>
        %key3 = util.pack %str3, %int3 : !db.nullable<!db.string>,i32 -> tuple<!db.nullable<!db.string>,i32>
        %key4 = util.pack %str4, %int4 : !db.nullable<!db.string>,i32 -> tuple<!db.nullable<!db.string>,i32>

        %val1 = util.pack %int1, %int1 : i32,i32 -> tuple<i32,i32>
        %val2 = util.pack %int2, %int2 : i32,i32 -> tuple<i32,i32>
        %val3 = util.pack %int3, %int3 : i32,i32 -> tuple<i32,i32>
        %val4 = util.pack %int4, %int4 : i32,i32 -> tuple<i32,i32>



        %initial = util.pack %zero, %one : i32,i32 -> tuple<i32,i32>

        %ht = dsa.create_ds %initial : tuple<i32,i32> -> !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>
        dsa.ht_insert %ht : !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>, %key1 : tuple<!db.nullable<!db.string>,i32>, %val1 : tuple<i32,i32>  hash: (%key : tuple<!db.nullable<!db.string>,i32>){
            %h = db.hash %key : tuple<!db.nullable<!db.string>,i32>
            dsa.yield %h : index
         }  eq: (%left : tuple<!db.nullable<!db.string>,i32>,%right : tuple<!db.nullable<!db.string>,i32>){
            %l1,%l2 = util.unpack %left : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %r1,%r2 = util.unpack %right : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %cmp1 = db.compare "eq" %l1 : !db.nullable<!db.string>, %r1 : !db.nullable<!db.string>
            %cmp2 = db.compare "eq" %l2 : i32, %r2 : i32
            %anded= db.and %cmp1, %cmp2 : !db.nullable<i1>, i1
            %t = db.derive_truth %anded : !db.nullable<i1>
            dsa.yield %t : i1
        } reduce: (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
            %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
            %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
            %add1 = db.add %curr1 : i32,%new1 : i32
            %mul1 = db.mul %curr2 : i32,%new2 : i32
            %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
            dsa.yield %updated_tuple : tuple<i32, i32>
         }
         dsa.ht_insert %ht : !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>, %key2 : tuple<!db.nullable<!db.string>,i32>, %val2 : tuple<i32,i32>  hash: (%key : tuple<!db.nullable<!db.string>,i32>){
            %h = db.hash %key : tuple<!db.nullable<!db.string>,i32>
            dsa.yield %h : index
         }  eq: (%left : tuple<!db.nullable<!db.string>,i32>,%right : tuple<!db.nullable<!db.string>,i32>){
             %l1,%l2 = util.unpack %left : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
             %r1,%r2 = util.unpack %right : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
             %cmp1 = db.compare "eq" %l1 : !db.nullable<!db.string>, %r1 : !db.nullable<!db.string>
             %cmp2 = db.compare "eq" %l2 : i32, %r2 : i32
             %anded= db.and %cmp1, %cmp2 : !db.nullable<i1>, i1
             %t = db.derive_truth %anded : !db.nullable<i1>
             dsa.yield %t : i1
         } reduce: (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
             %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
             %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
             %add1 = db.add %curr1 : i32,%new1 : i32
             %mul1 = db.mul %curr2 : i32,%new2 : i32
             %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
             dsa.yield %updated_tuple : tuple<i32, i32>
          }
         dsa.ht_insert %ht : !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>, %key3 : tuple<!db.nullable<!db.string>,i32>, %val3 : tuple<i32,i32> hash: (%key : tuple<!db.nullable<!db.string>,i32>){
            %h = db.hash %key : tuple<!db.nullable<!db.string>,i32>
            dsa.yield %h : index
         } eq: (%left : tuple<!db.nullable<!db.string>,i32>,%right : tuple<!db.nullable<!db.string>,i32>){
             %l1,%l2 = util.unpack %left : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
             %r1,%r2 = util.unpack %right : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
             %cmp1 = db.compare "eq" %l1 : !db.nullable<!db.string>, %r1 : !db.nullable<!db.string>
             %cmp2 = db.compare "eq" %l2 : i32, %r2 : i32
             %anded= db.and %cmp1, %cmp2 : !db.nullable<i1>, i1
             %t = db.derive_truth %anded : !db.nullable<i1>
             dsa.yield %t : i1
         } reduce: (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
             %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
             %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
             %add1 = db.add %curr1 : i32,%new1 : i32
             %mul1 = db.mul %curr2 : i32,%new2 : i32
             %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
             dsa.yield %updated_tuple : tuple<i32, i32>
          }
          dsa.ht_insert %ht : !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>, %key4 : tuple<!db.nullable<!db.string>,i32>, %val4 : tuple<i32,i32>  hash: (%key : tuple<!db.nullable<!db.string>,i32>){
            %h = db.hash %key : tuple<!db.nullable<!db.string>,i32>
            dsa.yield %h : index
         }  eq: (%left : tuple<!db.nullable<!db.string>,i32>,%right : tuple<!db.nullable<!db.string>,i32>){
              %l1,%l2 = util.unpack %left : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
              %r1,%r2 = util.unpack %right : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
              %cmp1 = db.compare "eq" %l1 : !db.nullable<!db.string>, %r1 : !db.nullable<!db.string>
              %cmp2 = db.compare "eq" %l2 : i32, %r2 : i32
              %anded= db.and %cmp1, %cmp2 : !db.nullable<i1>, i1
              %t = db.derive_truth %anded : !db.nullable<i1>
              dsa.yield %t : i1
          } reduce: (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
              %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
              %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
              %add1 = db.add %curr1 : i32,%new1 : i32
              %mul1 = db.mul %curr2 : i32,%new2 : i32
              %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
              dsa.yield %updated_tuple : tuple<i32, i32>
           }
        db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
        dsa.for %entry in %ht : !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>> {
            %key,%val = util.unpack %entry : tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>> -> tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>
            %k1,%k2 = util.unpack %key : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %v1,%v2 = util.unpack %val : tuple<i32,i32> -> i32,i32
            db.runtime_call "DumpValue" (%k1) : (!db.nullable<!db.string>) -> ()
            db.runtime_call "DumpValue" (%k2) : (i32) -> ()
            db.runtime_call "DumpValue" (%v1) : (i32) -> ()
            db.runtime_call "DumpValue" (%v2) : (i32) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
        }
        return
	}
 }
