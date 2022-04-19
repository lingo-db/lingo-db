 // RUN: run-mlir %s | FileCheck %s
 !entry_type=type tuple<tuple<!db.string,i32>,tuple<i32,i32>>
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(2)
//CHECK: int(2)
//CHECK: string("---------------")
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(4)
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
         %zero=db.constant ( 0 ) : i32
         %one=db.constant ( 1 ) : i32

        %key1 = util.pack %str1, %int1 : !db.string,i32 -> tuple<!db.string,i32>
        %key2 = util.pack %str1, %int1 : !db.string,i32 -> tuple<!db.string,i32>
        %key3 = util.pack %str3, %int3 : !db.string,i32 -> tuple<!db.string,i32>
        %key4 = util.pack %str4, %int4 : !db.string,i32 -> tuple<!db.string,i32>

        %val1 = util.pack %int1, %int1 : i32,i32 -> tuple<i32,i32>
        %val2 = util.pack %int2, %int2 : i32,i32 -> tuple<i32,i32>
        %val3 = util.pack %int3, %int3 : i32,i32 -> tuple<i32,i32>
        %val4 = util.pack %int4, %int4 : i32,i32 -> tuple<i32,i32>



        %ht= dsa.create_ds !dsa.join_ht<tuple<!db.string,i32>,tuple<i32,i32>>
        dsa.ht_insert %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i32,i32>>, %key1 : tuple<!db.string,i32> , %val1 : tuple<i32,i32>  hash: (%key : tuple<!db.string,i32>){
                                                                                                                                                          %h = db.hash %key : tuple<!db.string,i32>
                                                                                                                                                          dsa.yield %h : index
                                                                                                                                                       }
        dsa.ht_insert %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i32,i32>>, %key2 : tuple<!db.string,i32> , %val2 : tuple<i32,i32> hash: (%key : tuple<!db.string,i32>){
                                                                                                                                                         %h = db.hash %key : tuple<!db.string,i32>
                                                                                                                                                         dsa.yield %h : index
                                                                                                                                                      }
        dsa.ht_insert %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i32,i32>>, %key3 : tuple<!db.string,i32> , %val3 : tuple<i32,i32> hash: (%key : tuple<!db.string,i32>){
                                                                                                                                                         %h = db.hash %key : tuple<!db.string,i32>
                                                                                                                                                         dsa.yield %h : index
                                                                                                                                                      }
        dsa.ht_insert %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i32,i32>>, %key4 : tuple<!db.string,i32> , %val4 : tuple<i32,i32> hash: (%key : tuple<!db.string,i32>){
                                                                                                                                                         %h = db.hash %key : tuple<!db.string,i32>
                                                                                                                                                         dsa.yield %h : index
                                                                                                                                                      }
        dsa.finalize %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i32,i32>>
        %hash = db.hash %key1  : tuple<!db.string,i32>
        %matches = dsa.lookup %ht :  !dsa.join_ht<tuple<!db.string,i32>,tuple<i32,i32>>, %hash : index -> !dsa.iterable<tuple<tuple<!db.string,i32>,tuple<i32,i32>>,join_ht_iterator>
        dsa.for %entry in %matches : !dsa.iterable<tuple<tuple<!db.string,i32>,tuple<i32,i32>>,join_ht_iterator> {
            %key,%val = util.unpack %entry : tuple<tuple<!db.string,i32>,tuple<i32,i32>> -> tuple<!db.string,i32>,tuple<i32,i32>
            %k1,%k2 = util.unpack %key : tuple<!db.string,i32> -> !db.string,i32
            %v1,%v2 = util.unpack %val : tuple<i32,i32> -> i32,i32
            db.runtime_call "DumpValue" (%k1) : (!db.string) -> ()
            db.runtime_call "DumpValue" (%k2) : (i32) -> ()
            db.runtime_call "DumpValue" (%v1) : (i32) -> ()
            db.runtime_call "DumpValue" (%v2) : (i32) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
        }
        return
	}
 }
