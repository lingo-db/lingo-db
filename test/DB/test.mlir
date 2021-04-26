// RUN: db-run %s | FileCheck %s

//CHECK: int(10)
//CHECK: int(10)
//CHECK: bool(true)
//CHECK: bool(false)
//CHECK: decimal(100.0000000001)
//CHECK: date(2020-06-11)
//CHECK: timestamp(2020-06-11 12:30:00)
//CHECK: interval(90)
//CHECK: interval(12)
//CHECK: float(100)
//CHECK: float(100)
//CHECK: string("hello world!")
//CHECK: int(20)
//CHECK: int(20)
//CHECK: float(200)
//CHECK: float(200)
//CHECK: decimal(200.0000000002)
//CHECK: int(0)
//CHECK: int(0)
//CHECK: float(0)
//CHECK: float(0)
//CHECK: decimal(0.E-10)
//CHECK: int(100)
//CHECK: int(100)
//CHECK: float(10000)
//CHECK: float(10000)
//CHECK: decimal(1000200.01)
//CHECK: int(1)
//CHECK: int(1)
//CHECK: float(10000)
//CHECK: float(10000)
//CHECK: decimal(1.00)
 module {
 	func @main () {
 		 %int32_const = db.constant ( 10 ) : !db.int<32>
 		 %int64_const = db.constant ( 10 ) : !db.int<64>
 		 %bool_true_const = db.constant ( 1 ) : !db.bool
		 %bool_false_const = db.constant ( 0 ) : !db.bool
		 %decimal10_const = db.constant ( "100.0000000001" ):!db.decimal<15,10>
		 %decimal2_const = db.constant ( "100.01" ):!db.decimal<15,2>
		 %date_const = db.constant ( "2020-06-11") : !db.date
		 %timestamp_const = db.constant ( "2020-06-11 12:30:00" ) :!db.timestamp
		 %interval_days_const = db.constant ( 90 ) :!db.interval<days>
		 %interval_months_const = db.constant ( 12 ) :!db.interval<months>
		 %float32_const = db.constant ( 100.0000000001 ) :!db.float<32>
		 %float64_const = db.constant ( 100.0000000001 ) :!db.float<64>
		 %str_const = db.constant ( "hello world!" ) :!db.string

		 %int32_add= db.add %int32_const : !db.int<32>, %int32_const : !db.int<32>
		 %int64_add= db.add %int64_const : !db.int<64>, %int64_const : !db.int<64>
		 %float32_add= db.add %float32_const : !db.float<32>, %float32_const : !db.float<32>
		 %float64_add= db.add %float64_const : !db.float<64>, %float64_const : !db.float<64>
		 %decimal10_add = db.add %decimal10_const : !db.decimal<15,10>, %decimal10_const : !db.decimal<15,10>
		 %int32_sub= db.sub %int32_const : !db.int<32>, %int32_const : !db.int<32>
		 %int64_sub= db.sub %int64_const : !db.int<64>, %int64_const : !db.int<64>
		 %float32_sub= db.sub %float32_const : !db.float<32>, %float32_const : !db.float<32>
		 %float64_sub= db.sub %float64_const : !db.float<64>, %float64_const : !db.float<64>
		 %decimal10_sub = db.sub %decimal10_const : !db.decimal<15,10>, %decimal10_const : !db.decimal<15,10>
		 %int32_mul= db.mul %int32_const : !db.int<32>, %int32_const : !db.int<32>
		 %int64_mul= db.mul %int64_const : !db.int<64>, %int64_const : !db.int<64>
		 %float32_mul= db.mul %float32_const : !db.float<32>, %float32_const : !db.float<32>
		 %float64_mul= db.mul %float64_const : !db.float<64>, %float64_const : !db.float<64>
		 %decimal2_mul = db.mul %decimal2_const : !db.decimal<15,2>, %decimal2_const : !db.decimal<15,2> //mlir bug: when folding constants > int64_t it fails because it tries to create a name c_%val and calls function that assumes the number is < in64_t
		 %int32_div= db.div %int32_const : !db.int<32>, %int32_const : !db.int<32>
		 %int64_div= db.div %int64_const : !db.int<64>, %int64_const : !db.int<64>
		 %float32_div= db.div %float32_const : !db.float<32>, %float32_const : !db.float<32>
		 %float64_div= db.div %float64_const : !db.float<64>, %float64_const : !db.float<64>
		 %decimal2_div = db.div %decimal2_const : !db.decimal<15,2>, %decimal2_const : !db.decimal<15,2>

 		 db.dump %int32_const : !db.int<32>
 		 db.dump %int64_const : !db.int<64>
 		 db.dump %bool_true_const : !db.bool
 		 db.dump %bool_false_const : !db.bool
 		 db.dump %decimal10_const : !db.decimal<15,10>
 		 db.dump %date_const : !db.date
 		 db.dump %timestamp_const : !db.timestamp
 		 db.dump %interval_days_const : !db.interval<days>
 		 db.dump %interval_months_const : !db.interval<months>
 		 db.dump %float32_const : !db.float<32>
 		 db.dump %float64_const : !db.float<64>
 		 db.dump %str_const : !db.string
 		 db.dump %int32_add: !db.int<32>
 		 db.dump %int64_add: !db.int<64>
 		 db.dump %float32_add: !db.float<32>
 		 db.dump %float64_add: !db.float<64>
 		 db.dump %decimal10_add : !db.decimal<15,10>
 		 db.dump %int32_sub: !db.int<32>
 		 db.dump %int64_sub: !db.int<64>
 		 db.dump %float32_sub: !db.float<32>
 		 db.dump %float64_sub: !db.float<64>
 		 db.dump %decimal10_sub : !db.decimal<15,10>
		 db.dump %int32_mul: !db.int<32>
		 db.dump %int64_mul: !db.int<64>
		 db.dump %float32_mul: !db.float<32>
		 db.dump %float64_mul: !db.float<64>
		 db.dump %decimal2_mul : !db.decimal<15,2>
 		 db.dump %int32_div: !db.int<32>
 		 db.dump %int64_div: !db.int<64>
 		 db.dump %float32_div: !db.float<32>
 		 db.dump %float64_div: !db.float<64>
 		 db.dump %decimal2_div : !db.decimal<15,2>

 		return
  }
 }