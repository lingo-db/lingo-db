module @tpchq4  {
  func @query() {
    %0 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_orderdate => @o_orderdate({type = !db.date<nullable>}), o_orderkey => @o_orderkey({type = !db.int<64,nullable>}), o_orderpriority => @o_orderpriority({type = !db.string<nullable>})}
    %1 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_commitdate => @l_commitdate({type = !db.date<nullable>}), l_orderkey => @l_orderkey({type = !db.int<64>}), l_receiptdate => @l_receiptdate({type = !db.date<nullable>})}
    %2 = relalg.selection %0 (%arg0: !relalg.tuple) {
      %4 = relalg.getattr %arg0 @orders::@o_orderdate : !db.date<nullable>
      %5 = relalg.getattr %arg0 @orders::@o_orderkey : !db.int<64,nullable>
      %6 = db.constant( "1993-07-01" ) : !db.date<nullable>
      %7 = db.constant( "1993-10-01" ) : !db.date<nullable>
      %8 = db.compare sge %4 : !db.date<nullable>, %6 : !db.date<nullable>
      %9 = db.compare slt %4 : !db.date<nullable>, %7 : !db.date<nullable>
      %10 = relalg.selection %1 (%arg1: !relalg.tuple) {
        %13 = relalg.getattr %arg1 @lineitem::@l_orderkey : !db.int<64,nullable>
        %14 = db.compare eq %5 : !db.int<64,nullable>, %13 : !db.int<64,nullable>
        %15 = relalg.getattr %arg1 @orders::@o_orderkey : !db.int<64,nullable>
        %16 = relalg.getattr %arg1 @lineitem::@l_orderkey : !db.int<64,nullable>
        %17 = db.compare slt %15 : !db.int<64,nullable>, %16 : !db.int<64,nullable>
        %18 = db.and %14:!db.bool<nullable>,%17:!db.bool<nullable>
        relalg.return %18 : !db.bool<nullable>
      }
      %11 = relalg.exists %10
      %12 = db.and %8:!db.bool<nullable>,%9:!db.bool<nullable>,%11:!db.bool
      relalg.return %12 : !db.bool<nullable>
    }
    %3 = relalg.aggregation @agg1 %2 [@orders::@o_orderpriority] (%arg0: !relalg.relation) {
      %4 = relalg.count %arg0
      relalg.addattr @order_count({type = !db.int<64>}) %4
      relalg.return
    }
    return
  }
}