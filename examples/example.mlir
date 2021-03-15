module @testmodule  {
  func @main() -> !db.int {
    %0 = db.constant( 2 ) : !db.int
    %1 = relalg.basetable @abctable  {table_identifier = "abc"} columns: {col1 => @col1({name = "abc", type = !db.int}), col2 => @col2({name = "dupp", type = !db.bool})}
    %2 = relalg.selection %1 (%arg0: !relalg.tuple){
        %3 = relalg.getattr %arg0, @abctable::@col1 : !db.int
        %4 = db.compare "eq", %0 :!db.int,%3:!db.int
        relalg.return %4 : !db.bool
    }
    return %0 : !db.int
  }
}