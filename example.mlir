 module {
	func @main () {
		%false = db.constant ( 0 ) : !db.bool
		%true = db.constant ( 1 ) : !db.bool
		%false_nullable= db.cast %false :!db.bool -> !db.bool<nullable>
		%true_nullable= db.cast %true :!db.bool -> !db.bool<nullable>
		%5 = util.combine %true_nullable, %false_nullable : !db.bool<nullable>, !db.bool<nullable> -> tuple<!db.bool<nullable>, !db.bool<nullable>>
		%6,%7 = util.split_tuple %5 : tuple<!db.bool<nullable>, !db.bool<nullable>> -> !db.bool<nullable>, !db.bool<nullable>
		//CHECK: bool(true)
		//CHECK: bool(false)
		db.dump %6 : !db.bool<nullable>
		db.dump %7 : !db.bool<nullable>

		return
	}
 }