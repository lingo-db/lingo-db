//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                    l_shipmode  |               high_line_count  |                low_line_count  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                        "MAIL"  |                           647  |                           945  |
//CHECK: |                        "SHIP"  |                           620  |                           943  |
module {
  func @main() -> !db.table {
    %0 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %1 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.selection %2 (%arg0: !relalg.tuple){
      %8 = relalg.getcol %arg0 @orders::@o_orderkey : i32
      %9 = relalg.getcol %arg0 @lineitem::@l_orderkey : i32
      %10 = db.compare eq %8 : i32, %9 : i32
      %11 = db.constant("MAIL") : !db.string
      %12 = db.constant("SHIP") : !db.string
      %13 = relalg.getcol %arg0 @lineitem::@l_shipmode : !db.string
      %14 = db.oneof %13 : !db.string ? %11, %12 : !db.string, !db.string
      %15 = relalg.getcol %arg0 @lineitem::@l_commitdate : !db.date<day>
      %16 = relalg.getcol %arg0 @lineitem::@l_receiptdate : !db.date<day>
      %17 = db.compare lt %15 : !db.date<day>, %16 : !db.date<day>
      %18 = relalg.getcol %arg0 @lineitem::@l_shipdate : !db.date<day>
      %19 = relalg.getcol %arg0 @lineitem::@l_commitdate : !db.date<day>
      %20 = db.compare lt %18 : !db.date<day>, %19 : !db.date<day>
      %21 = relalg.getcol %arg0 @lineitem::@l_receiptdate : !db.date<day>
      %22 = db.constant("1994-01-01") : !db.date<day>
      %23 = db.compare gte %21 : !db.date<day>, %22 : !db.date<day>
      %24 = relalg.getcol %arg0 @lineitem::@l_receiptdate : !db.date<day>
      %25 = db.constant("1995-01-01") : !db.date<day>
      %26 = db.compare lt %24 : !db.date<day>, %25 : !db.date<day>
      %27 = db.and %10, %14, %17, %20, %23, %26 : i1, i1, i1, i1, i1, i1
      relalg.return %27 : i1
    }
    %4 = relalg.map @map0 %3 (%arg0: !relalg.tuple){
      %8 = relalg.getcol %arg0 @orders::@o_orderpriority : !db.string
      %9 = db.constant("1-URGENT") : !db.string
      %10 = db.compare neq %8 : !db.string, %9 : !db.string
      %11 = relalg.getcol %arg0 @orders::@o_orderpriority : !db.string
      %12 = db.constant("2-HIGH") : !db.string
      %13 = db.compare neq %11 : !db.string, %12 : !db.string
      %14 = db.and %10, %13 : i1, i1
      %15 = scf.if %14 -> (i32) {
        %26 = db.constant(1 : i32) : i32
        scf.yield %26 : i32
      } else {
        %26 = db.constant(0 : i32) : i32
        scf.yield %26 : i32
      }
      %16 = relalg.addcol %arg0, @tmp_attr3({type = i32}) %15
      %17 = relalg.getcol %16 @orders::@o_orderpriority : !db.string
      %18 = db.constant("1-URGENT") : !db.string
      %19 = db.compare eq %17 : !db.string, %18 : !db.string
      %20 = relalg.getcol %16 @orders::@o_orderpriority : !db.string
      %21 = db.constant("2-HIGH") : !db.string
      %22 = db.compare eq %20 : !db.string, %21 : !db.string
      %23 = db.or %19, %22 : i1, i1
      %24 = scf.if %23 -> (i32) {
        %26 = db.constant(1 : i32) : i32
        scf.yield %26 : i32
      } else {
        %26 = db.constant(0 : i32) : i32
        scf.yield %26 : i32
      }
      %25 = relalg.addcol %16, @tmp_attr1({type = i32}) %24
      relalg.return %25 : !relalg.tuple
    }
    %5 = relalg.aggregation @aggr0 %4 [@lineitem::@l_shipmode] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %8 = relalg.aggrfn sum @map0::@tmp_attr3 %arg0 : i32
      %9 = relalg.addcol %arg1, @tmp_attr2({type = i32}) %8
      %10 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : i32
      %11 = relalg.addcol %9, @tmp_attr0({type = i32}) %10
      relalg.return %11 : !relalg.tuple
    }
    %6 = relalg.sort %5 [(@lineitem::@l_shipmode,asc)]
    %7 = relalg.materialize %6 [@lineitem::@l_shipmode,@aggr0::@tmp_attr0,@aggr0::@tmp_attr2] => ["l_shipmode", "high_line_count", "low_line_count"] : !db.table
    return %7 : !db.table
  }
}

