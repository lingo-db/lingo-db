module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %1 = relalg.selection %0 (%arg0: !relalg.tuple){
      %5 = relalg.getcol %arg0 @lineitem::@l_shipdate : !db.date<day>
      %6 = db.constant("1994-01-01") : !db.date<day>
      %7 = db.compare gte %5 : !db.date<day>, %6 : !db.date<day>
      %8 = relalg.getcol %arg0 @lineitem::@l_shipdate : !db.date<day>
      %9 = db.constant("1995-01-01") : !db.date<day>
      %10 = db.compare lt %8 : !db.date<day>, %9 : !db.date<day>
      %11 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %12 = db.constant("0.06") : !db.decimal<3, 2>
      %13 = db.constant("0.01") : !db.decimal<3, 2>
      %14 = db.sub %12 : !db.decimal<3, 2>, %13 : !db.decimal<3, 2>
      %15 = db.constant("0.06") : !db.decimal<3, 2>
      %16 = db.constant("0.01") : !db.decimal<3, 2>
      %17 = db.add %15 : !db.decimal<3, 2>, %16 : !db.decimal<3, 2>
      %18 = db.cast %14 : !db.decimal<3, 2> -> !db.decimal<15, 2>
      %19 = db.cast %17 : !db.decimal<3, 2> -> !db.decimal<15, 2>
      %20 = db.between %11 : !db.decimal<15, 2> between %18 : !db.decimal<15, 2>, %19 : !db.decimal<15, 2>, lowerInclusive : true, upperInclusive : true
      %21 = relalg.getcol %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %22 = db.constant(24 : i32) : !db.decimal<15, 2>
      %23 = db.compare lt %21 : !db.decimal<15, 2>, %22 : !db.decimal<15, 2>
      %24 = db.and %7, %10, %20, %23 : i1, i1, i1, i1
      relalg.return %24 : i1
    }
    %2 = relalg.map @map0 %1 computes : [@tmp_attr1({type = !db.decimal<15, 2>})] (%arg0: !relalg.tuple){
      %5 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %6 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %7 = db.mul %5 : !db.decimal<15, 2>, %6 : !db.decimal<15, 2>
      relalg.return %7 : !db.decimal<15, 2>
    }
    %3 = relalg.aggregation @aggr0 %2 [] computes : [@tmp_attr0({type = !db.nullable<!db.decimal<15, 2>>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %5 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.nullable<!db.decimal<15, 2>>
      relalg.return %5 : !db.nullable<!db.decimal<15, 2>>
    }
    %4 = relalg.materialize %3 [@aggr0::@tmp_attr0] => ["revenue"] : !dsa.table
    return %4 : !dsa.table
  }
}
