// RUN: env LINGODB_EXECUTION_MODE=DEFAULT run-mlir %s | FileCheck %s
// RUN: if [ "$(uname)" = "Linux" ]; then env LINGODB_EXECUTION_MODE=BASELINE run-mlir %s | FileCheck %s; fi

 module {
	func.func @main () {
 		%date_const = db.constant ( "2020-06-13") : !db.date<day>
 		%interval_1_month_const = db.constant ( 1 ) :!db.interval<months>
 		%interval_1_year_const = db.constant ( 12 ) :!db.interval<months>
  		//CHECK: date(2020-07-13)
   		%1 = db.runtime_call "DateAdd" (%date_const,%interval_1_month_const) : (!db.date<day>,!db.interval<months>) -> !db.date<day>
  		db.runtime_call "DumpValue" (%1) : (!db.date<day>) -> ()
  		//CHECK: date(2021-06-13)
   		%2 = db.runtime_call "DateAdd" (%date_const,%interval_1_year_const) : (!db.date<day>,!db.interval<months>) -> !db.date<day>
   		db.runtime_call "DumpValue" (%2) : (!db.date<day>) -> ()
   		//CHECK: int(13)
   		%day = db.constant ("day") : !db.string
   		%3 = db.runtime_call "ExtractFromDate"( %day, %date_const) : (!db.string,!db.date<day>) -> i64
   		db.runtime_call "DumpValue" (%3) : (i64) -> ()
   		//CHECK: int(6)
   		%month = db.constant ("month") : !db.string
   		%4 = db.runtime_call "ExtractFromDate"( %month, %date_const) : (!db.string,!db.date<day>) -> i64
   		db.runtime_call "DumpValue" (%4) : (i64) -> ()
   		//CHECK: int(2020)
        %year = db.constant ("year") : !db.string
   		%5 = db.runtime_call "ExtractFromDate"( %year, %date_const) : (!db.string,!db.date<day>) -> i64
		db.runtime_call "DumpValue" (%5) : (i64) -> ()
		return
	}
 }
