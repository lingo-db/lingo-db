 // RUN: db-run %s | FileCheck %s
 !entry_type=type tuple<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
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
         %str1 = db.cast %str1c  : !db.string -> !db.string<nullable>
         %str2 = db.cast %str2c  : !db.string -> !db.string<nullable>
         %str3 = db.cast %str3c  : !db.string -> !db.string<nullable>
         %str4 = db.cast %str4c  : !db.string -> !db.string<nullable>
         %int1=db.constant ( 4 ) : !db.int<32>
         %int2=db.constant ( 2 ) : !db.int<32>
         %int3=db.constant ( 3 ) : !db.int<32>
         %int4=db.constant ( 1 ) : !db.int<32>
         %zero=db.constant ( 0 ) : !db.int<32>
         %one=db.constant ( 1 ) : !db.int<32>

        %key1 = util.pack %str1, %int1 : !db.string<nullable>,!db.int<32> -> tuple<!db.string<nullable>,!db.int<32>>
        %key2 = util.pack %str1, %int1 : !db.string<nullable>,!db.int<32> -> tuple<!db.string<nullable>,!db.int<32>>
        %key3 = util.pack %str3, %int3 : !db.string<nullable>,!db.int<32> -> tuple<!db.string<nullable>,!db.int<32>>
        %key4 = util.pack %str4, %int4 : !db.string<nullable>,!db.int<32> -> tuple<!db.string<nullable>,!db.int<32>>

        %val1 = util.pack %int1, %int1 : !db.int<32>,!db.int<32> -> tuple<!db.int<32>,!db.int<32>>
        %val2 = util.pack %int2, %int2 : !db.int<32>,!db.int<32> -> tuple<!db.int<32>,!db.int<32>>
        %val3 = util.pack %int3, %int3 : !db.int<32>,!db.int<32> -> tuple<!db.int<32>,!db.int<32>>
        %val4 = util.pack %int4, %int4 : !db.int<32>,!db.int<32> -> tuple<!db.int<32>,!db.int<32>>

        %entry1 = util.pack %key1,%val1 :tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>> -> tuple<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %entry2 = util.pack %key2,%val2 :tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>> -> tuple<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %entry3 = util.pack %key3,%val3 :tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>> -> tuple<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        %entry4 = util.pack %key4,%val4 :tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>> -> tuple<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>


        %initial = util.pack %zero, %one : !db.int<32>,!db.int<32> -> tuple<!db.int<32>,!db.int<32>>

        %aggr_ht_builder= db.create_aggr_ht_builder %initial : tuple<!db.int<32>,!db.int<32>>-> !db.aggr_ht_builder<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>, tuple<!db.int<32>,!db.int<32>>>
        %builder1= db.builder_merge %aggr_ht_builder : !db.aggr_ht_builder<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>, tuple<!db.int<32>,!db.int<32>>>, %entry1 : !entry_type (%curr:tuple<!db.int<32>,!db.int<32>>, %new:tuple<!db.int<32>,!db.int<32>>) {
                                                                                                                                                                                                                     %curr1,%curr2 = util.unpack %curr : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
                                                                                                                                                                                                                     %new1,%new2 = util.unpack %new : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
                                                                                                                                                                                                                     %add1 = db.add %curr1 : !db.int<32>,%new1 : !db.int<32>
                                                                                                                                                                                                                     %mul1 = db.mul %curr2 : !db.int<32>,%new2 : !db.int<32>
                                                                                                                                                                                                          		   %updated_tuple = util.pack %add1, %mul1 : !db.int<32>, !db.int<32> -> tuple<!db.int<32>, !db.int<32>>
                                                                                                                                                                                                                     db.yield %updated_tuple : tuple<!db.int<32>, !db.int<32>>
                                                                                                                                                                                                                  }
        db.dump %str_const : !db.string
        %builder2= db.builder_merge %builder1 : !db.aggr_ht_builder<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>, tuple<!db.int<32>,!db.int<32>>>, %entry2 : !entry_type (%curr:tuple<!db.int<32>,!db.int<32>>, %new:tuple<!db.int<32>,!db.int<32>>) {
                                                                                                                                                                                                              %curr1,%curr2 = util.unpack %curr : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
                                                                                                                                                                                                              %new1,%new2 = util.unpack %new : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
                                                                                                                                                                                                              %add1 = db.add %curr1 : !db.int<32>,%new1 : !db.int<32>
                                                                                                                                                                                                              %mul1 = db.mul %curr2 : !db.int<32>,%new2 : !db.int<32>
                                                                                                                                                                                                   		   %updated_tuple = util.pack %add1, %mul1 : !db.int<32>, !db.int<32> -> tuple<!db.int<32>, !db.int<32>>
                                                                                                                                                                                                              db.yield %updated_tuple : tuple<!db.int<32>, !db.int<32>>
                                                                                                                                                                                                           }
        db.dump %str_const : !db.string
        %builder3= db.builder_merge %builder2 : !db.aggr_ht_builder<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>, tuple<!db.int<32>,!db.int<32>>>, %entry3 : !entry_type (%curr:tuple<!db.int<32>,!db.int<32>>, %new:tuple<!db.int<32>,!db.int<32>>) {
                                                                                                                                                                                                              %curr1,%curr2 = util.unpack %curr : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
                                                                                                                                                                                                              %new1,%new2 = util.unpack %new : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
                                                                                                                                                                                                              %add1 = db.add %curr1 : !db.int<32>,%new1 : !db.int<32>
                                                                                                                                                                                                              %mul1 = db.mul %curr2 : !db.int<32>,%new2 : !db.int<32>
                                                                                                                                                                                                   		   %updated_tuple = util.pack %add1, %mul1 : !db.int<32>, !db.int<32> -> tuple<!db.int<32>, !db.int<32>>
                                                                                                                                                                                                              db.yield %updated_tuple : tuple<!db.int<32>, !db.int<32>>
                                                                                                                                                                                                           }
        db.dump %str_const : !db.string
        %builder4= db.builder_merge %builder3 : !db.aggr_ht_builder<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>, tuple<!db.int<32>,!db.int<32>>>, %entry4 : !entry_type (%curr:tuple<!db.int<32>,!db.int<32>>, %new:tuple<!db.int<32>,!db.int<32>>) {
                                                                                                                                                                                                              %curr1,%curr2 = util.unpack %curr : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
                                                                                                                                                                                                              %new1,%new2 = util.unpack %new : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
                                                                                                                                                                                                              %add1 = db.add %curr1 : !db.int<32>,%new1 : !db.int<32>
                                                                                                                                                                                                              %mul1 = db.mul %curr2 : !db.int<32>,%new2 : !db.int<32>
                                                                                                                                                                                                   		   %updated_tuple = util.pack %add1, %mul1 : !db.int<32>, !db.int<32> -> tuple<!db.int<32>, !db.int<32>>
                                                                                                                                                                                                              db.yield %updated_tuple : tuple<!db.int<32>, !db.int<32>>
                                                                                                                                                                                                           }
        db.dump %str_const : !db.string
        %ht  = db.builder_build %builder4 : !db.aggr_ht_builder<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>, tuple<!db.int<32>,!db.int<32>>> -> !db.aggr_ht<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>>
        db.for %entry in %ht : !db.aggr_ht<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>> {
            %key,%val = util.unpack %entry : tuple<tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>> -> tuple<!db.string<nullable>,!db.int<32>>,tuple<!db.int<32>,!db.int<32>>
            %k1,%k2 = util.unpack %key : tuple<!db.string<nullable>,!db.int<32>> -> !db.string<nullable>,!db.int<32>
            %v1,%v2 = util.unpack %val : tuple<!db.int<32>,!db.int<32>> -> !db.int<32>,!db.int<32>
            db.dump %k1 : !db.string<nullable>
            db.dump %k2 : !db.int<32>
            db.dump %v1 : !db.int<32>
            db.dump %v2 : !db.int<32>
            db.dump %str_const : !db.string
        }
        return
	}
 }
