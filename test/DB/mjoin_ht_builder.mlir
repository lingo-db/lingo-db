 // RUN: db-run %s | FileCheck %s
 !entry_type=type tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(2)
//CHECK: int(2)
//CHECK: string("---------------")
//CHECK: int(0)
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(4)
//CHECK: int(4)
//CHECK: string("---------------")
//CHECK: int(0)
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(2)
//CHECK: int(2)
//CHECK: int(1)
//CHECK: string("---------------")
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(4)
//CHECK: int(4)
//CHECK: int(1)
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(4)
//CHECK: int(4)
//CHECK: int(1)
//CHECK: string("---------------")
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(2)
//CHECK: int(2)
//CHECK: int(1)
//CHECK: string("---------------")
//CHECK: string("strc")
//CHECK: int(3)
//CHECK: int(3)
//CHECK: int(3)
//CHECK: int(0)
//CHECK: string("---------------")
//CHECK: string("strj")
//CHECK: int(1)
//CHECK: int(1)
//CHECK: int(1)
//CHECK: int(0)
//CHECK: string("---------------")
 module {

	func @main (%execution_context:  !util.generic_memref<i8>) {
         %str_const = db.constant ( "---------------" ) :!db.string

         %str1=db.constant ( "stra" ) :!db.string
         %str2=db.constant ( "strb" ) :!db.string
         %str3=db.constant ( "strc" ) :!db.string
         %str4=db.constant ( "strj" ) :!db.string
         %int1=db.constant ( 4 ) : !db.int<32>
         %int2=db.constant ( 2 ) : !db.int<32>
         %int3=db.constant ( 3 ) : !db.int<32>
         %int4=db.constant ( 1 ) : !db.int<32>
         %zero=db.constant ( 0 ) : !db.int<32>
         %one=db.constant ( 1 ) : !db.int<32>

        %key1 = util.pack %str1, %int1 : !db.string,!db.int<32> -> tuple<!db.string,!db.int<32>>
        %key2 = util.pack %str1, %int1 : !db.string,!db.int<32> -> tuple<!db.string,!db.int<32>>
        %key3 = util.pack %str3, %int3 : !db.string,!db.int<32> -> tuple<!db.string,!db.int<32>>
        %key4 = util.pack %str4, %int4 : !db.string,!db.int<32> -> tuple<!db.string,!db.int<32>>

        %val1 = util.pack %int1, %int1 : !db.int<32>,!db.int<32> -> tuple<!db.int<32>,!db.int<32>>
        %val2 = util.pack %int2, %int2 : !db.int<32>,!db.int<32> -> tuple<!db.int<32>,!db.int<32>>
        %val3 = util.pack %int3, %int3 : !db.int<32>,!db.int<32> -> tuple<!db.int<32>,!db.int<32>>
        %val4 = util.pack %int4, %int4 : !db.int<32>,!db.int<32> -> tuple<!db.int<32>,!db.int<32>>

        %entry1 = util.pack %key1,%val1 :tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>> -> tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %entry2 = util.pack %key2,%val2 :tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>> -> tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %entry3 = util.pack %key3,%val3 :tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>> -> tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %entry4 = util.pack %key4,%val4 :tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>> -> tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>


        %mjoin_ht_builder= db.create_mjoin_ht_builder : !db.mjoin_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %builder1= db.builder_merge %mjoin_ht_builder : !db.mjoin_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>, %entry1 : !entry_type -> !db.mjoin_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %builder2= db.builder_merge %builder1 : !db.mjoin_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>, %entry2 : !entry_type -> !db.mjoin_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %builder3= db.builder_merge %builder2 : !db.mjoin_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>, %entry3 : !entry_type -> !db.mjoin_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %builder4= db.builder_merge %builder3 : !db.mjoin_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>, %entry4 : !entry_type -> !db.mjoin_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %ht  = db.builder_build %builder4 : !db.mjoin_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>> -> !db.mjoin_ht<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %matches = db.lookup %ht :  !db.mjoin_ht<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>, %key1  : tuple<!db.string,!db.int<32>> -> !db.iterable<tuple<tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,memref<i8>>,mjoin_ht_iterator>
        db.for %entry in %matches : !db.iterable<tuple<tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,memref<i8>>,mjoin_ht_iterator> {
            %payload,%marker = util.unpack %entry : tuple<tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,memref<i8>> -> tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,memref<i8>
            %key,%val = util.unpack %payload : tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>> -> tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>
            %k1,%k2 = util.unpack %key : tuple<!db.string,!db.int<32>> -> !db.string,!db.int<32>
            %v1,%v2 = util.unpack %val : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
            db.dump %k1 : !db.string
            db.dump %k2 : !db.int<32>
            db.dump %v1 : !db.int<32>
            db.dump %v2 : !db.int<32>
            db.dump %str_const : !db.string
            %one_i8 = constant 1 : i8
            %marker_val=atomic_rmw "assign" %one_i8 , %marker[] : (i8, memref<i8>) -> i8
            %db_marker_val = db.type_cast %marker_val : i8 -> !db.int<8>
            db.dump %db_marker_val : !db.int<8>
        }
            db.dump %str_const : !db.string
            db.dump %str_const : !db.string

        db.for %entry in %matches : !db.iterable<tuple<tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,memref<i8>>,mjoin_ht_iterator> {
            %payload,%marker = util.unpack %entry : tuple<tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,memref<i8>> -> tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,memref<i8>
            %key,%val = util.unpack %payload : tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>> -> tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>
            %k1,%k2 = util.unpack %key : tuple<!db.string,!db.int<32>> -> !db.string,!db.int<32>
            %v1,%v2 = util.unpack %val : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
            db.dump %k1 : !db.string
            db.dump %k2 : !db.int<32>
            db.dump %v1 : !db.int<32>
            db.dump %v2 : !db.int<32>
            %one_i8 = constant 1 : i8
            %marker_val=atomic_rmw "assign" %one_i8 , %marker[] : (i8, memref<i8>) -> i8
            %db_marker_val = db.type_cast %marker_val : i8 -> !db.int<8>
            db.dump %db_marker_val : !db.int<8>
            db.dump %str_const : !db.string
        }
            db.dump %str_const : !db.string
            db.dump %str_const : !db.string
            db.dump %str_const : !db.string
            db.dump %str_const : !db.string

        db.for %entry in %ht : !db.mjoin_ht<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>> {
            %payload,%marker = util.unpack %entry : tuple<tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,memref<i8>> -> tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,memref<i8>
            %key,%val = util.unpack %payload : tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>> -> tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>
            %k1,%k2 = util.unpack %key : tuple<!db.string,!db.int<32>> -> !db.string,!db.int<32>
            %v1,%v2 = util.unpack %val : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
            db.dump %k1 : !db.string
            db.dump %k2 : !db.int<32>
            db.dump %v1 : !db.int<32>
            db.dump %v2 : !db.int<32>
            %one_i8 = constant 1 : i8
            %marker_val=atomic_rmw "assign" %one_i8 , %marker[] : (i8, memref<i8>) -> i8
            %db_marker_val = db.type_cast %marker_val : i8 -> !db.int<8>
            db.dump %db_marker_val : !db.int<8>
            db.dump %str_const : !db.string
        }
        return
	}
 }
