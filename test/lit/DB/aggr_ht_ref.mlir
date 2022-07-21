 // RUN: run-mlir %s | FileCheck %s
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

	func.func @main () {
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

        %hash1 = db.hash %key1 : tuple<!db.nullable<!db.string>,i32>
        %hash2 = db.hash %key2 : tuple<!db.nullable<!db.string>,i32>
        %hash3 = db.hash %key3 : tuple<!db.nullable<!db.string>,i32>
        %hash4 = db.hash %key4 : tuple<!db.nullable<!db.string>,i32>
        %val1 = util.pack %int1, %int1 : i32,i32 -> tuple<i32,i32>
        %val2 = util.pack %int2, %int2 : i32,i32 -> tuple<i32,i32>
        %val3 = util.pack %int3, %int3 : i32,i32 -> tuple<i32,i32>
        %val4 = util.pack %int4, %int4 : i32,i32 -> tuple<i32,i32>



        %initial = util.pack %zero, %one : i32,i32 -> tuple<i32,i32>

        %ht = dsa.create_ds !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>
        %ref1 = dsa.ht_get_ref_or_insert %ht : !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>, %hash1, %key1 : tuple<!db.nullable<!db.string>,i32>
         eq: (%left : tuple<!db.nullable<!db.string>,i32>,%right : tuple<!db.nullable<!db.string>,i32>){
            %l1,%l2 = util.unpack %left : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %r1,%r2 = util.unpack %right : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %cmp1 = db.compare "eq" %l1 : !db.nullable<!db.string>, %r1 : !db.nullable<!db.string>
            %cmp2 = db.compare "eq" %l2 : i32, %r2 : i32
            %anded= db.and %cmp1, %cmp2 : !db.nullable<i1>, i1
            %t = db.derive_truth %anded : !db.nullable<i1>
            dsa.yield %t : i1
        } initial: {
             %zero0=db.constant ( 0 ) : i32
             %one0=db.constant ( 1 ) : i32
            %initial0 = util.pack %zero0, %one0 : i32,i32 -> tuple<i32,i32>
            dsa.yield %initial0 : tuple<i32,i32>
        }
        %val_ref1 = util.tupleelementptr %ref1[1] : !util.ref<tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>> ->  !util.ref<tuple<i32,i32>>
        %curr_val1 = util.load %val_ref1[] : !util.ref<tuple<i32,i32>> -> tuple<i32,i32>
        %curr_val11, %curr_val12 = util.unpack %curr_val1 : tuple<i32,i32> -> i32, i32
        %add1 = db.add %curr_val11 : i32,%int1 : i32
        %mul1 = db.mul %curr_val12 : i32,%int1 : i32
        %updated_val1 = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
        util.store %updated_val1 : tuple<i32, i32> , %val_ref1[] : !util.ref<tuple<i32,i32>>

        %ref2 = dsa.ht_get_ref_or_insert %ht : !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>, %hash2, %key2 : tuple<!db.nullable<!db.string>,i32>
         eq: (%left : tuple<!db.nullable<!db.string>,i32>,%right : tuple<!db.nullable<!db.string>,i32>){
            %l1,%l2 = util.unpack %left : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %r1,%r2 = util.unpack %right : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %cmp1 = db.compare "eq" %l1 : !db.nullable<!db.string>, %r1 : !db.nullable<!db.string>
            %cmp2 = db.compare "eq" %l2 : i32, %r2 : i32
            %anded= db.and %cmp1, %cmp2 : !db.nullable<i1>, i1
            %t = db.derive_truth %anded : !db.nullable<i1>
            dsa.yield %t : i1
        } initial: {
             %zero0=db.constant ( 0 ) : i32
             %one0=db.constant ( 1 ) : i32
            %initial0 = util.pack %zero0, %one0 : i32,i32 -> tuple<i32,i32>
            dsa.yield %initial0 : tuple<i32,i32>
        }
        %val_ref2 = util.tupleelementptr %ref2[1] : !util.ref<tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>> ->  !util.ref<tuple<i32,i32>>
        %curr_val2 = util.load %val_ref2[] : !util.ref<tuple<i32,i32>> -> tuple<i32,i32>
        %curr_val21, %curr_val22 = util.unpack %curr_val2 : tuple<i32,i32> -> i32, i32
        %add2 = db.add %curr_val21 : i32,%int2 : i32
        %mul2 = db.mul %curr_val22 : i32,%int2 : i32
        %updated_val2 = util.pack %add2, %mul2 : i32, i32 -> tuple<i32, i32>
        util.store %updated_val2 : tuple<i32, i32> , %val_ref2[] : !util.ref<tuple<i32,i32>>



        %ref3 = dsa.ht_get_ref_or_insert %ht : !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>, %hash3, %key3 : tuple<!db.nullable<!db.string>,i32>
         eq: (%left : tuple<!db.nullable<!db.string>,i32>,%right : tuple<!db.nullable<!db.string>,i32>){
            %l1,%l2 = util.unpack %left : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %r1,%r2 = util.unpack %right : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %cmp1 = db.compare "eq" %l1 : !db.nullable<!db.string>, %r1 : !db.nullable<!db.string>
            %cmp2 = db.compare "eq" %l2 : i32, %r2 : i32
            %anded= db.and %cmp1, %cmp2 : !db.nullable<i1>, i1
            %t = db.derive_truth %anded : !db.nullable<i1>
            dsa.yield %t : i1
        } initial: {
             %zero0=db.constant ( 0 ) : i32
             %one0=db.constant ( 1 ) : i32
            %initial0 = util.pack %zero0, %one0 : i32,i32 -> tuple<i32,i32>
            dsa.yield %initial0 : tuple<i32,i32>
        }
        %val_ref3 = util.tupleelementptr %ref3[1] : !util.ref<tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>> ->  !util.ref<tuple<i32,i32>>
        %curr_val3 = util.load %val_ref3[] : !util.ref<tuple<i32,i32>> -> tuple<i32,i32>
        %curr_val31, %curr_val32 = util.unpack %curr_val3 : tuple<i32,i32> -> i32, i32
        %add3 = db.add %curr_val31 : i32,%int3 : i32
        %mul3 = db.mul %curr_val32 : i32,%int3 : i32
        %updated_val3 = util.pack %add3, %mul3 : i32, i32 -> tuple<i32, i32>
        util.store %updated_val3 : tuple<i32, i32> , %val_ref3[] : !util.ref<tuple<i32,i32>>

        %ref4 = dsa.ht_get_ref_or_insert %ht : !dsa.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>, %hash4, %key4 : tuple<!db.nullable<!db.string>,i32>
         eq: (%left : tuple<!db.nullable<!db.string>,i32>,%right : tuple<!db.nullable<!db.string>,i32>){
            %l1,%l2 = util.unpack %left : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %r1,%r2 = util.unpack %right : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %cmp1 = db.compare "eq" %l1 : !db.nullable<!db.string>, %r1 : !db.nullable<!db.string>
            %cmp2 = db.compare "eq" %l2 : i32, %r2 : i32
            %anded= db.and %cmp1, %cmp2 : !db.nullable<i1>, i1
            %t = db.derive_truth %anded : !db.nullable<i1>
            dsa.yield %t : i1
        } initial: {
             %zero0=db.constant ( 0 ) : i32
             %one0=db.constant ( 1 ) : i32
            %initial0 = util.pack %zero0, %one0 : i32,i32 -> tuple<i32,i32>
            dsa.yield %initial0 : tuple<i32,i32>
        }
        %val_ref4 = util.tupleelementptr %ref4[1] : !util.ref<tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>> ->  !util.ref<tuple<i32,i32>>
        %curr_val4 = util.load %val_ref4[] : !util.ref<tuple<i32,i32>> -> tuple<i32,i32>
        %curr_val41, %curr_val42 = util.unpack %curr_val4 : tuple<i32,i32> -> i32, i32
        %add4 = db.add %curr_val41 : i32,%int4 : i32
        %mul4 = db.mul %curr_val42 : i32,%int4 : i32
        %updated_val4 = util.pack %add4, %mul4 : i32, i32 -> tuple<i32, i32>
        util.store %updated_val4 : tuple<i32, i32> , %val_ref4[] : !util.ref<tuple<i32,i32>>



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
