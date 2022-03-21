// RUN: db-run %s | FileCheck %s

 module {
	func @main () {
 		%date_const = db.constant ( "2020-06-13") : !db.date<day>
 		%interval_1_month_const = db.constant ( 1 ) :!db.interval<months>
 		%interval_1_year_const = db.constant ( 12 ) :!db.interval<months>
 		%interval_90_days_const = db.constant ( 7776000000 ) :!db.interval<daytime>
  		//CHECK: date(2020-07-13)
  		%1 = db.date_add %date_const:!db.date<day>, %interval_1_month_const :!db.interval<months>
  		db.runtime_call "DumpValue" (%1) : (!db.date<day>) -> ()
  		//CHECK: date(2021-06-13)
   		%2 = db.date_add %date_const:!db.date<day>, %interval_1_year_const :!db.interval<months>
   		db.runtime_call "DumpValue" (%2) : (!db.date<day>) -> ()
   		//CHECK: int(13)
   		%3 = db.date_extract day, %date_const : !db.date<day>
   		db.runtime_call "DumpValue" (%3) : (i64) -> ()
   		//CHECK: int(6)
   		%4 = db.date_extract month, %date_const : !db.date<day>
   		db.runtime_call "DumpValue" (%4) : (i64) -> ()
   		//CHECK: int(2020)
		%5 = db.date_extract year, %date_const : !db.date<day>
		db.runtime_call "DumpValue" (%5) : (i64) -> ()
		return
	}
 }