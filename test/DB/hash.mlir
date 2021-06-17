// RUN: db-run %s | FileCheck %s

module {
    func @main () {
         %int32_const = db.constant ( 10 ) : !db.int<32>

         %int64_const = db.constant ( 10 ) : !db.int<64>
         %bool_true_const = db.constant ( 1 ) : !db.bool
         %decimal2_const = db.constant ( "100.01" ):!db.decimal<15,2>
         %date_const = db.constant ( "2020-06-11") : !db.date<day>
         %timestamp_const = db.constant ( "2020-06-11 12:30:00" ) :!db.timestamp<second>
         %float64_const = db.constant ( 100.0000000001 ) :!db.float<64>
         %str_const = db.constant ( "hello world!" ) :!db.string

         %hash1 = db.hash %int32_const : !db.int<32>
         %hash2 = db.hash %int64_const : !db.int<64>
         %hash3 = db.hash %bool_true_const : !db.bool
         %hash4 = db.hash %decimal2_const : !db.decimal<15,2>
         %hash5 = db.hash %date_const : !db.date<day>
         %hash6 = db.hash %timestamp_const : !db.timestamp<second>
         %hash7 = db.hash %float64_const : !db.float<64>
         %hash8 = db.hash %str_const : !db.string
         %tuple = util.pack %int32_const, %int64_const, %bool_true_const, %decimal2_const, %date_const, %timestamp_const, %float64_const, %str_const
          : !db.int<32>,!db.int<64>,!db.bool,!db.decimal<15,2>,!db.date<day>,!db.timestamp<second>,!db.float<64>,!db.string -> tuple<!db.int<32>, !db.int<64>,!db.bool,!db.decimal<15,2>,!db.date<day>,!db.timestamp<second>,!db.float<64>,!db.string>
         %tuple_hash = db.hash %tuple : tuple<!db.int<32>, !db.int<64>,!db.bool,!db.decimal<15,2>,!db.date<day>,!db.timestamp<second>,!db.float<64>,!db.string>


//CHECK: index(10071948712206390799)
//CHECK: index(14456516954291514641)
//CHECK: index(5161771770385946686)
//CHECK: index(4684687102070716739)
//CHECK: index(18197025341656604621)
//CHECK: index(7793543308923603326)
//CHECK: index(621161557985691062)
//CHECK: index(4577883861590986596)
//CHECK: index(16303516795538966018)

         db.dump_index %hash1
         db.dump_index %hash2
         db.dump_index %hash3
         db.dump_index %hash4
         db.dump_index %hash5
         db.dump_index %hash6
         db.dump_index %hash7
         db.dump_index %hash8
         db.dump_index %tuple_hash
        return
    }
}