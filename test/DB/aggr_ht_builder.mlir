 // RUN: db-run %s | FileCheck %s
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
         %str1 = db.cast %str1c  : !db.string -> !db.nullable<!db.string>
         %str2 = db.cast %str2c  : !db.string -> !db.nullable<!db.string>
         %str3 = db.cast %str3c  : !db.string -> !db.nullable<!db.string>
         %str4 = db.cast %str4c  : !db.string -> !db.nullable<!db.string>
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

        %entry1 = util.pack %key1,%val1 :tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32> -> tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>
        %entry2 = util.pack %key2,%val2 :tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32> -> tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>
        %entry3 = util.pack %key3,%val3 :tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32> -> tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>
        %entry4 = util.pack %key4,%val4 :tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32> -> tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>


        %initial = util.pack %zero, %one : i32,i32 -> tuple<i32,i32>

        %aggr_ht_builder= db.create_aggr_ht_builder %initial : tuple<i32,i32>-> !db.aggr_ht_builder<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>, tuple<i32,i32>>
        %builder1= db.builder_merge %aggr_ht_builder : !db.aggr_ht_builder<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>, tuple<i32,i32>>, %entry1 : !entry_type (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
                                                                                                                                                                                                                     %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
                                                                                                                                                                                                                     %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
                                                                                                                                                                                                                     %add1 = db.add %curr1 : i32,%new1 : i32
                                                                                                                                                                                                                     %mul1 = db.mul %curr2 : i32,%new2 : i32
                                                                                                                                                                                                          		   %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
                                                                                                                                                                                                                     db.yield %updated_tuple : tuple<i32, i32>
                                                                                                                                                                                                                  }
        db.dump %str_const : !db.string
        %builder2= db.builder_merge %builder1 : !db.aggr_ht_builder<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>, tuple<i32,i32>>, %entry2 : !entry_type (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
                                                                                                                                                                                                              %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
                                                                                                                                                                                                              %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
                                                                                                                                                                                                              %add1 = db.add %curr1 : i32,%new1 : i32
                                                                                                                                                                                                              %mul1 = db.mul %curr2 : i32,%new2 : i32
                                                                                                                                                                                                   		   %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
                                                                                                                                                                                                              db.yield %updated_tuple : tuple<i32, i32>
                                                                                                                                                                                                           }
        db.dump %str_const : !db.string
        %builder3= db.builder_merge %builder2 : !db.aggr_ht_builder<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>, tuple<i32,i32>>, %entry3 : !entry_type (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
                                                                                                                                                                                                              %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
                                                                                                                                                                                                              %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
                                                                                                                                                                                                              %add1 = db.add %curr1 : i32,%new1 : i32
                                                                                                                                                                                                              %mul1 = db.mul %curr2 : i32,%new2 : i32
                                                                                                                                                                                                   		   %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
                                                                                                                                                                                                              db.yield %updated_tuple : tuple<i32, i32>
                                                                                                                                                                                                           }
        db.dump %str_const : !db.string
        %builder4= db.builder_merge %builder3 : !db.aggr_ht_builder<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>, tuple<i32,i32>>, %entry4 : !entry_type (%curr:tuple<i32,i32>, %new:tuple<i32,i32>) {
                                                                                                                                                                                                              %curr1,%curr2 = util.unpack %curr : tuple<i32,i32> -> i32,i32
                                                                                                                                                                                                              %new1,%new2 = util.unpack %new : tuple<i32,i32> -> i32,i32
                                                                                                                                                                                                              %add1 = db.add %curr1 : i32,%new1 : i32
                                                                                                                                                                                                              %mul1 = db.mul %curr2 : i32,%new2 : i32
                                                                                                                                                                                                   		   %updated_tuple = util.pack %add1, %mul1 : i32, i32 -> tuple<i32, i32>
                                                                                                                                                                                                              db.yield %updated_tuple : tuple<i32, i32>
                                                                                                                                                                                                           }
        db.dump %str_const : !db.string
        %ht  = db.builder_build %builder4 : !db.aggr_ht_builder<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>, tuple<i32,i32>> -> !db.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>>
        db.for %entry in %ht : !db.aggr_ht<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>> {
            %key,%val = util.unpack %entry : tuple<tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>> -> tuple<!db.nullable<!db.string>,i32>,tuple<i32,i32>
            %k1,%k2 = util.unpack %key : tuple<!db.nullable<!db.string>,i32> -> !db.nullable<!db.string>,i32
            %v1,%v2 = util.unpack %val : tuple<i32,i32> -> i32,i32
            db.dump %k1 : !db.nullable<!db.string>
            db.dump %k2 : i32
            db.dump %v1 : i32
            db.dump %v2 : i32
            db.dump %str_const : !db.string
        }
        return
	}
 }
