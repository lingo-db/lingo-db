//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |               o_orderpriority  |                   order_count  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                    "1-URGENT"  |                           999  |
//CHECK: |                      "2-HIGH"  |                           997  |
//CHECK: |                    "3-MEDIUM"  |                          1031  |
//CHECK: |             "4-NOT SPECIFIED"  |                           989  |
//CHECK: |                       "5-LOW"  |                          1077  |
module {
  func @main() -> !db.table {
    %0 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %1 = relalg.selection %0 (%arg0: !relalg.tuple){
      %5 = relalg.getattr %arg0 @orders::@o_orderdate : !db.date<day>
      %6 = db.constant("1993-07-01") : !db.date<day>
      %7 = db.compare gte %5 : !db.date<day>, %6 : !db.date<day>
      %8 = relalg.getattr %arg0 @orders::@o_orderdate : !db.date<day>
      %9 = db.constant("1993-10-01") : !db.date<day>
      %10 = db.compare lt %8 : !db.date<day>, %9 : !db.date<day>
      %11 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
      %12 = relalg.selection %11 (%arg1: !relalg.tuple){
        %15 = relalg.getattr %arg1 @lineitem::@l_orderkey : i32
        %16 = relalg.getattr %arg1 @orders::@o_orderkey : i32
        %17 = db.compare eq %15 : i32, %16 : i32
        %18 = relalg.getattr %arg1 @lineitem::@l_commitdate : !db.date<day>
        %19 = relalg.getattr %arg1 @lineitem::@l_receiptdate : !db.date<day>
        %20 = db.compare lt %18 : !db.date<day>, %19 : !db.date<day>
        %21 = db.and %17, %20 : i1, i1
        relalg.return %21 : i1
      }
      %13 = relalg.exists %12
      %14 = db.and %7, %10, %13 : i1, i1, i1
      relalg.return %14 : i1
    }
    %2 = relalg.aggregation @aggr0 %1 [@orders::@o_orderpriority] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %5 = relalg.count %arg0
      %6 = relalg.addattr %arg1, @tmp_attr0({type = i64}) %5
      relalg.return %6 : !relalg.tuple
    }
    %3 = relalg.sort %2 [(@orders::@o_orderpriority,asc)]
    %4 = relalg.materialize %3 [@orders::@o_orderpriority,@aggr0::@tmp_attr0] => ["o_orderpriority", "order_count"] : !db.table
    return %4 : !db.table
  }
}

