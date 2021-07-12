 // RUN: db-run %s | FileCheck %s
 !entry_type=type tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
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


        %join_ht_builder= db.create_join_ht_builder : !db.join_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %builder1= db.builder_merge %join_ht_builder : !db.join_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>, %entry1 : !entry_type -> !db.join_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %builder2= db.builder_merge %builder1 : !db.join_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>, %entry2 : !entry_type -> !db.join_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %builder3= db.builder_merge %builder2 : !db.join_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>, %entry3 : !entry_type -> !db.join_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %builder4= db.builder_merge %builder3 : !db.join_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>, %entry4 : !entry_type -> !db.join_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %ht  = db.builder_build %builder4 : !db.join_ht_builder<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>> -> !db.join_ht<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %matches = db.lookup %ht :  !db.join_ht<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>, %key1  : tuple<!db.string,!db.int<32>> -> !db.iterable<tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,join_ht_iterator>
        db.for %entry in %matches : !db.iterable<tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>,join_ht_iterator> {
            %key,%val = util.unpack %entry : tuple<tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>> -> tuple<!db.string,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>
            %k1,%k2 = util.unpack %key : tuple<!db.string,!db.int<32>> -> !db.string,!db.int<32>
            %v1,%v2 = util.unpack %val : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
            db.dump %k1 : !db.string
            db.dump %k2 : !db.int<32>
            db.dump %v1 : !db.int<32>
            db.dump %v2 : !db.int<32>
            db.dump %str_const : !db.string
        }
        return
	}
 }
