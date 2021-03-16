module @testmodule  {
  func @main() {
    %0 = db.constant( 2 ) : !db.int<64>
    %1 = relalg.basetable @abctable  {table_identifier = "abc"} columns: {col1 => @col1({name = "abc", type = !db.int<64>}), col2 => @col2({name = "dupp", type = !db.bool})}
    %2 = relalg.selection %1 (%arg0:!relalg.tuple) {
      %4 = relalg.getattr %arg0 @abctable::@col1 : !db.int<64>
      %5 = db.compare eq %0 : !db.int<64>, %4 : !db.int<64> !db.bool
      relalg.return %5 : !db.bool
    }
    %3 = relalg.aggregation @agg1 %1 [@abctable::@col1,@abctable::@col2] (%arg1:!relalg.relation) {
      %4 = relalg.aggr.sum @abctable::@col1 %arg1 : !db.int<64>
      relalg.addattr @sum({name="sum", type=!db.int<64>}) %4
      relalg.return
    }
    relalg.foreach %3 [@abctable::@col1,@abctable::@col2,@agg1::@sum] (%arg2 : !db.int<64>,%arg3 : !db.int<64>, %arg4 :!db.int<64>){
      %5 = db.add %arg2:!db.int<64>, %arg3 :!db.int<64>
      db.print "res: %d\n", %5:!db.int<64>
      relalg.return
    }
    return
  }
}