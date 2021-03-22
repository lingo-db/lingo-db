module @testmodule  {
  func @main() {
    %1 = relalg.basetable @abctable  {table_identifier = "abc"} columns: {col1 => @col1({name = "abc", type = !db.int<64>}), col2 => @col2({name = "dupp", type = !db.bool})}
    %2 = relalg.sort %1 [(@abctable::@col1,desc),(@abctable::@colw,asc)]
    %3 = relalg.limit 10 %2
    return
  }
}