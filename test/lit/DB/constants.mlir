// RUN: run-mlir %s | FileCheck %s

//CHECK: char<2>("AB")
//CHECK: int(10)
//CHECK: int(10)
//CHECK: bool(true)
//CHECK: bool(false)
//CHECK: decimal(100.0000000001)
//CHECK: date(2020-06-11)
//CHECK: timestamp(2020-06-11 12:30:00)
//CHECK: interval(90 daytime)
//CHECK: interval(12 months)
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
//CHECK: decimal(0E-10)
//CHECK: int(100)
//CHECK: int(100)
//CHECK: float(10000)
//CHECK: float(10000)
//CHECK: decimal(10002.0001)
//CHECK: int(1)
//CHECK: int(1)
//CHECK: float(1)
//CHECK: float(1)
//CHECK: decimal(1.00000000000000000)
 module {
 	func.func @main () {
 	     %char_const = db.constant ( "AB" ) : !db.char<2>
 		 %int32_const = db.constant ( 10 ) : i32
 		 %int64_const = db.constant ( 10 ) : i64
 		 %bool_true_const = db.constant ( 1 ) : i1
		 %bool_false_const = db.constant ( 0 ) : i1
		 %decimal10_const = db.constant ( "100.0000000001" ):!db.decimal<15,10>
		 %decimal2_const = db.constant ( "100.01" ):!db.decimal<15,2>
		 %date_const = db.constant ( "2020-06-11") : !db.date<day>
		 %timestamp_const = db.constant ( "2020-06-11 12:30:00" ) :!db.timestamp<second>
		 %interval_days_const = db.constant ( 90 ) :!db.interval<daytime>
		 %interval_months_const = db.constant ( 12 ) :!db.interval<months>
		 %float32_const = db.constant ( 100.0000000001 ) :f32
		 %float64_const = db.constant ( 100.0000000001 ) :f64
		 %str_const = db.constant ( "hello world!" ) :!db.string

		 %int32_add= db.add %int32_const : i32, %int32_const : i32
		 %int64_add= db.add %int64_const : i64, %int64_const : i64
		 %float32_add= db.add %float32_const : f32, %float32_const : f32
		 %float64_add= db.add %float64_const : f64, %float64_const : f64
		 %decimal10_add = db.add %decimal10_const : !db.decimal<15,10>, %decimal10_const : !db.decimal<15,10>
		 %int32_sub= db.sub %int32_const : i32, %int32_const : i32
		 %int64_sub= db.sub %int64_const : i64, %int64_const : i64
		 %float32_sub= db.sub %float32_const : f32, %float32_const : f32
		 %float64_sub= db.sub %float64_const : f64, %float64_const : f64
		 %decimal10_sub = db.sub %decimal10_const : !db.decimal<15,10>, %decimal10_const : !db.decimal<15,10>
		 %int32_mul= db.mul %int32_const : i32, %int32_const : i32
		 %int64_mul= db.mul %int64_const : i64, %int64_const : i64
		 %float32_mul= db.mul %float32_const : f32, %float32_const : f32
		 %float64_mul= db.mul %float64_const : f64, %float64_const : f64
		 %decimal2_mul = db.mul %decimal2_const : !db.decimal<15,2>, %decimal2_const : !db.decimal<15,2> //mlir bug: when folding constants > int64_t it fails because it tries to create a name c_%val and calls function that assumes the number is < in64_t
		 %int32_div= db.div %int32_const : i32, %int32_const : i32
		 %int64_div= db.div %int64_const : i64, %int64_const : i64
		 %float32_div= db.div %float32_const : f32, %float32_const : f32
		 %float64_div= db.div %float64_const : f64, %float64_const : f64
		 %decimal2_div = db.div %decimal2_const : !db.decimal<15,2>, %decimal2_const : !db.decimal<15,2>
         db.runtime_call "DumpValue" (%char_const) : (!db.char<2>) -> ()
 		 db.runtime_call "DumpValue" (%int32_const) : (i32) -> ()
 		 db.runtime_call "DumpValue" (%int64_const) : (i64) -> ()
 		 db.runtime_call "DumpValue" (%bool_true_const) : (i1) -> ()
 		 db.runtime_call "DumpValue" (%bool_false_const) : (i1) -> ()
 		 db.runtime_call "DumpValue" (%decimal10_const) : (!db.decimal<15,10>) -> ()
 		 db.runtime_call "DumpValue" (%date_const) : (!db.date<day>) -> ()
 		 db.runtime_call "DumpValue" (%timestamp_const) : (!db.timestamp<second>) -> ()
 		 db.runtime_call "DumpValue" (%interval_days_const) : (!db.interval<daytime>) -> ()
 		 db.runtime_call "DumpValue" (%interval_months_const) : (!db.interval<months>) -> ()
 		 db.runtime_call "DumpValue" (%float32_const) : (f32) -> ()
 		 db.runtime_call "DumpValue" (%float64_const) : (f64) -> ()
 		 db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
 		 db.runtime_call "DumpValue" (%int32_add) : (i32) -> ()
 		 db.runtime_call "DumpValue" (%int64_add) : (i64) -> ()
 		 db.runtime_call "DumpValue" (%float32_add) : (f32) -> ()
 		 db.runtime_call "DumpValue" (%float64_add) : (f64) -> ()
 		 db.runtime_call "DumpValue" (%decimal10_add) : (!db.decimal<15,10>) -> ()
 		 db.runtime_call "DumpValue" (%int32_sub) : (i32) -> ()
 		 db.runtime_call "DumpValue" (%int64_sub) : (i64) -> ()
 		 db.runtime_call "DumpValue" (%float32_sub) : (f32) -> ()
 		 db.runtime_call "DumpValue" (%float64_sub) : (f64) -> ()
 		 db.runtime_call "DumpValue" (%decimal10_sub) : (!db.decimal<15,10>) -> ()
		 db.runtime_call "DumpValue" (%int32_mul) : (i32) -> ()
		 db.runtime_call "DumpValue" (%int64_mul) : (i64) -> ()
		 db.runtime_call "DumpValue" (%float32_mul) : (f32) -> ()
		 db.runtime_call "DumpValue" (%float64_mul) : (f64) -> ()
		 db.runtime_call "DumpValue" (%decimal2_mul) : (!db.decimal<30,4>) -> ()
 		 db.runtime_call "DumpValue" (%int32_div) : (i32) -> ()
 		 db.runtime_call "DumpValue" (%int64_div) : (i64) -> ()
 		 db.runtime_call "DumpValue" (%float32_div) : (f32) -> ()
 		 db.runtime_call "DumpValue" (%float64_div) : (f64) -> ()
 		 db.runtime_call "DumpValue" (%decimal2_div) : (!db.decimal<32,17>) -> ()

 		return
  }
 }
