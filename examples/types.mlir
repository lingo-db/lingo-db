module @testmodule  {
  func @main() {
    %0 = db.constant( 2 ) : !db.int<32,nullable>
    %1 = db.constant( 2 ) : !db.int<32>
    %2 = db.constant( 2 ) : !db.string<nullable>
    %3 = db.constant( 2 ) : !db.string
    %4 = db.constant( 2 ) : !db.bool
    %5 = db.constant( 2 ) : !db.bool<nullable>
    %6 = db.constant( 2.0 ) : !db.decimal<4,6>
    %7 = db.constant( 2 ) : !db.decimal<4,6,nullable>


    return
  }
}