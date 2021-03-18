module @testmodule  {
  func @main() {
    %1 = relalg.basetable @abctable  {table_identifier = "abc"} columns: {col1 => @col1({name = "abc", type = !db.int<64>}), col2 => @col2({name = "dupp", type = !db.bool})}
    %2 = relalg.join "inner" %1,%1 (%tuple : !relalg.tuple){
        relalg.return
    }
    %3 = relalg.join "semi" %1,%1 (%tuple : !relalg.tuple){
                                         relalg.return
                                     }
    %4 = relalg.join "antisemi" %1,%1 (%tuple : !relalg.tuple){
                                             relalg.return
                                         }
    %5 = relalg.outerjoin "leftouter" %1,%1 (%tuple : !relalg.tuple){
                                                   relalg.return
                                               }
    %6 = relalg.outerjoin "rightouter" %1,%1 (%tuple : !relalg.tuple){
                                                    relalg.return
                                                }
    %7 = relalg.outerjoin "fullouter" %1,%1 (%tuple : !relalg.tuple){
                                                   relalg.return
                                               }
    %8 = relalg.dependentjoin "semi" [@a,@b,@c] %1,%1 (%tuple : !relalg.tuple){
                                                   relalg.return
                                               }




    %9 = relalg.distinct [@a,@b] %8

    %10 = relalg.union @union1 distinct %9,%8 mapping: {
      @a({type=!db.int<64>})=[@a1,@a2],
      @b({type=!db.int<64>})=[@b1,@b2]
    }

    %11 = relalg.except @union1 distinct %9,%8 mapping: {
      @a({type=!db.int<64>})=[@a1,@a2],
      @b({type=!db.int<64>})=[@b1,@b2]
    }

    %12 = relalg.intersect @union1 distinct %9,%8 mapping: {
      @a({type=!db.int<64>})=[@a1,@a2],
      @b({type=!db.int<64>})=[@b1,@b2]
    }
    return
  }

}