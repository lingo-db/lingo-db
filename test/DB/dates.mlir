// RUN: db-run %s | FileCheck %s

 module {
	func @main () {
 		%date_const = db.constant ( "2020-06-13") : !db.date
 		%interval_90_days_const = db.constant ( 90 ) :!db.interval<days>
 		%interval_1_month_const = db.constant ( 1 ) :!db.interval<months>
 		%interval_1_year_const = db.constant ( 1 ) :!db.interval<years>

 		//CHECK: date(2020-09-11)
 		%0 = db.date_add %date_const:!db.date, %interval_90_days_const :!db.interval<days>
 		db.dump %0 : !db.date
 		//CHECK: date(2020-07-13)
  		%1 = db.date_add %date_const:!db.date, %interval_1_month_const :!db.interval<months>
  		db.dump %1 : !db.date
  		//CHECK: date(2021-06-13)
   		%2 = db.date_add %date_const:!db.date, %interval_1_year_const :!db.interval<years>
   		db.dump %2 : !db.date
   		//CHECK: int(13)
   		%3 = db.date_extract "day", %date_const : !db.date
   		db.dump %3 : !db.int<32>
   		//CHECK: int(6)
   		%4 = db.date_extract "month", %date_const : !db.date
   		db.dump %4 : !db.int<32>
   		//CHECK: int(2020)
		%5 = db.date_extract "year", %date_const : !db.date
		db.dump %5 : !db.int<32>
		//CHECK: date(2020-03-15)
		%6 = db.date_sub %date_const:!db.date, %interval_90_days_const :!db.interval<days>
		db.dump %6 : !db.date
		//CHECK: date(2020-05-13)
		%7 = db.date_sub %date_const:!db.date, %interval_1_month_const :!db.interval<months>
		db.dump %7 : !db.date
		//CHECK: date(2019-06-13)
		%8 = db.date_sub %date_const:!db.date, %interval_1_year_const :!db.interval<years>
		db.dump %8 : !db.date
		return
	}
 }