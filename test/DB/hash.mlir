// RUN: db-run %s | FileCheck %s

module {
    func @main () {
         %int32_const = db.constant ( 10 ) : i32

         %int64_const = db.constant ( 10 ) : i64
         %bool_true_const = db.constant ( 1 ) : i1
         %decimal2_const = db.constant ( "100.01" ):!db.decimal<15,2>
         %date_const = db.constant ( "2020-06-11") : !db.date<day>
         %timestamp_const = db.constant ( "2020-06-11 12:30:00" ) :!db.timestamp<second>
         %str_const = db.constant ( "hello world!" ) :!db.string

         %hash1 = db.hash %int32_const : i32
         %hash2 = db.hash %int64_const : i64
         %hash3 = db.hash %bool_true_const : i1
         %hash4 = db.hash %decimal2_const : !db.decimal<15,2>
         %hash5 = db.hash %date_const : !db.date<day>
         %hash6 = db.hash %timestamp_const : !db.timestamp<second>
         %hash7 = db.hash %str_const : !db.string
         %tuple = util.pack %int32_const, %int64_const, %bool_true_const, %decimal2_const, %date_const, %timestamp_const, %str_const
          : i32,i64,i1,!db.decimal<15,2>,!db.date<day>,!db.timestamp<second>,!db.string -> tuple<i32, i64,i1,!db.decimal<15,2>,!db.date<day>,!db.timestamp<second>,!db.string>
         %tuple_hash = db.hash %tuple : tuple<i32, i64,i1,!db.decimal<15,2>,!db.date<day>,!db.timestamp<second>,!db.string>


//CHECK: index(9003023063795233148)
//CHECK: index(9003023063795233148)
//CHECK: index(14576801547736533962)
//CHECK: index(5768746606534069840)
//CHECK: index(17331534688241223152)
//CHECK: index(12374225058675341995)
//CHECK: index(11221175996223801097)
//CHECK: index(14428801650687329114)

         db.dump_index %hash1
         db.dump_index %hash2
         db.dump_index %hash3
         db.dump_index %hash4
         db.dump_index %hash5
         db.dump_index %hash6
         db.dump_index %hash7
         db.dump_index %tuple_hash
        return
    }
}