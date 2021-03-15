module @testmodule  {
  func @main() -> !db.int {
    %0 = db.constant( 2 ) : !db.int
    %1 = relalg.basetable @abctable  {table_identifier = "abc"} columns: {col1 => @col1({name = "abc", type = !db.int}), col2 => @col2({name = "dupp", type = !db.bool})}
    %2 = relalg.selection %1 (%arg0:!relalg.tuple) {
      %4 = relalg.getattr %arg0 @abctable::@col1 : !db.int
      %5 = db.compare eq, %0 : !db.int, %4 : !db.int
      relalg.return %5 : !db.bool
    }
    %3 = relalg.aggregation @agg1 %1 [@abctable::@col1,@abctable::@col2] (%arg0:!relalg.relation) {
      %4 = relalg.aggr.sum @abctable::@col1 %arg0 : !db.int
      relalg.addattr @sum({name="sum", type=!db.int}) %4 :!db.int
      relalg.return
    }
    return %0 : !db.int
  }
}