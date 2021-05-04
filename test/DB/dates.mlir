// RUN: db-run %s | FileCheck %s

 module {
	func @main () {
 		%date_const = db.constant ( "2020-06-13") : !db.date<day>
 		%interval_1_month_const = db.constant ( 1 ) :!db.interval<months>
 		%interval_1_year_const = db.constant ( 12 ) :!db.interval<months>
  		//CHECK: date(2020-07-13)
  		%1 = db.date_add %date_const:!db.date<day>, %interval_1_month_const :!db.interval<months>
  		db.dump %1 : !db.date<day>
  		//CHECK: date(2021-06-13)
   		%2 = db.date_add %date_const:!db.date<day>, %interval_1_year_const :!db.interval<months>
   		db.dump %2 : !db.date<day>
   		//CHECK: int(13)
   		%3 = db.date_extract day, %date_const : !db.date<day>
   		db.dump %3 : !db.int<64>
   		//CHECK: int(6)
   		%4 = db.date_extract month, %date_const : !db.date<day>
   		db.dump %4 : !db.int<64>
   		//CHECK: int(2020)
		%5 = db.date_extract year, %date_const : !db.date<day>
		db.dump %5 : !db.int<64>
		//CHECK: date(2020-05-13)
		%7 = db.date_sub %date_const:!db.date<day>, %interval_1_month_const :!db.interval<months>
		db.dump %7 : !db.date<day>
		//CHECK: date(2019-06-13)
		%8 = db.date_sub %date_const:!db.date<day>, %interval_1_year_const :!db.interval<months>
		db.dump %8 : !db.date<day>
		return
	}
 }