module {
  func @main() -> !db.table {
    %0 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %1 = relalg.selection %0 (%arg0: !relalg.tuple){
      %6 = relalg.getcol %arg0 @lineitem::@l_shipdate : !db.date<day>
      %7 = db.constant("1998-12-01") : !db.date<day>
      %8 = db.constant("90days") : !db.interval<daytime>
      %9 = db.runtime_call "DateSubtract"(%7, %8) : (!db.date<day>, !db.interval<daytime>) -> !db.date<day>
      %10 = db.compare lte %6 : !db.date<day>, %9 : !db.date<day>
      relalg.return %10 : i1
    }
    %2 = relalg.map @map0 %1 (%arg0: !relalg.tuple){
      %6 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %7 = db.constant(1 : i32) : !db.decimal<15, 2>
      %8 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %9 = db.sub %7 : !db.decimal<15, 2>, %8 : !db.decimal<15, 2>
      %10 = db.mul %6 : !db.decimal<15, 2>, %9 : !db.decimal<15, 2>
      %11 = db.constant(1 : i32) : !db.decimal<15, 2>
      %12 = relalg.getcol %arg0 @lineitem::@l_tax : !db.decimal<15, 2>
      %13 = db.add %11 : !db.decimal<15, 2>, %12 : !db.decimal<15, 2>
      %14 = db.mul %10 : !db.decimal<15, 2>, %13 : !db.decimal<15, 2>
      %15 = relalg.addcol %arg0, @tmp_attr5({type = !db.decimal<15, 2>}) %14
      %16 = relalg.getcol %15 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %17 = db.constant(1 : i32) : !db.decimal<15, 2>
      %18 = relalg.getcol %15 @lineitem::@l_discount : !db.decimal<15, 2>
      %19 = db.sub %17 : !db.decimal<15, 2>, %18 : !db.decimal<15, 2>
      %20 = db.mul %16 : !db.decimal<15, 2>, %19 : !db.decimal<15, 2>
      %21 = relalg.addcol %15, @tmp_attr3({type = !db.decimal<15, 2>}) %20
      relalg.return %21 : !relalg.tuple
    }
    %3 = relalg.aggregation @aggr0 %2 [@lineitem::@l_returnflag,@lineitem::@l_linestatus] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %6 = relalg.aggrfn avg @lineitem::@l_discount %arg0 : !db.decimal<15, 2>
      %7 = relalg.addcol %arg1, @tmp_attr8({type = !db.decimal<15, 2>}) %6
      %8 = relalg.count %arg0
      %9 = relalg.addcol %7, @tmp_attr9({type = i64}) %8
      %10 = relalg.aggrfn avg @lineitem::@l_extendedprice %arg0 : !db.decimal<15, 2>
      %11 = relalg.addcol %9, @tmp_attr7({type = !db.decimal<15, 2>}) %10
      %12 = relalg.aggrfn sum @map0::@tmp_attr5 %arg0 : !db.decimal<15, 2>
      %13 = relalg.addcol %11, @tmp_attr4({type = !db.decimal<15, 2>}) %12
      %14 = relalg.aggrfn sum @map0::@tmp_attr3 %arg0 : !db.decimal<15, 2>
      %15 = relalg.addcol %13, @tmp_attr2({type = !db.decimal<15, 2>}) %14
      %16 = relalg.aggrfn sum @lineitem::@l_extendedprice %arg0 : !db.decimal<15, 2>
      %17 = relalg.addcol %15, @tmp_attr1({type = !db.decimal<15, 2>}) %16
      %18 = relalg.aggrfn avg @lineitem::@l_quantity %arg0 : !db.decimal<15, 2>
      %19 = relalg.addcol %17, @tmp_attr6({type = !db.decimal<15, 2>}) %18
      %20 = relalg.aggrfn sum @lineitem::@l_quantity %arg0 : !db.decimal<15, 2>
      %21 = relalg.addcol %19, @tmp_attr0({type = !db.decimal<15, 2>}) %20
      relalg.return %21 : !relalg.tuple
    }
    %4 = relalg.sort %3 [(@lineitem::@l_returnflag,asc),(@lineitem::@l_linestatus,asc)]
    %5 = relalg.materialize %4 [@lineitem::@l_returnflag,@lineitem::@l_linestatus,@aggr0::@tmp_attr0,@aggr0::@tmp_attr1,@aggr0::@tmp_attr2,@aggr0::@tmp_attr4,@aggr0::@tmp_attr6,@aggr0::@tmp_attr7,@aggr0::@tmp_attr8,@aggr0::@tmp_attr9] => ["l_returnflag", "l_linestatus", "sum_qty", "sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price", "avg_disc", "count_order"] : !db.table
    return %5 : !db.table
  }
}
