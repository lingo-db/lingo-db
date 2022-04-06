//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                        c_name  |                     c_custkey  |                    o_orderkey  |                   o_orderdate  |                  o_totalprice  |                           sum  |
//CHECK: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//CHECK: |          "Customer#000001639"  |                          1639  |                        502886  |                    1994-04-12  |                     456423.88  |                        312.00  |
//CHECK: |          "Customer#000006655"  |                          6655  |                         29158  |                    1995-10-21  |                     452805.02  |                        305.00  |
//CHECK: |          "Customer#000014110"  |                         14110  |                        565574  |                    1995-09-24  |                     425099.85  |                        301.00  |
//CHECK: |          "Customer#000001775"  |                          1775  |                          6882  |                    1997-04-09  |                     408368.10  |                        303.00  |
//CHECK: |          "Customer#000011459"  |                         11459  |                        551136  |                    1993-05-19  |                     386812.74  |                        308.00  |
module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
    %1 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.selection %4 (%arg0: !relalg.tuple){
      %10 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
      %11 = relalg.aggregation @aggr0 %10 [@lineitem::@l_orderkey] computes : [@tmp_attr0({type = !db.decimal<15, 2>})] (%arg1: !relalg.tuplestream,%arg2: !relalg.tuple){
        %23 = relalg.aggrfn sum @lineitem::@l_quantity %arg1 : !db.decimal<15, 2>
        relalg.return %23 : !db.decimal<15, 2>
      }
      %12 = relalg.selection %11 (%arg1: !relalg.tuple){
        %23 = relalg.getcol %arg1 @aggr0::@tmp_attr0 : !db.decimal<15, 2>
        %24 = db.constant(300 : i32) : !db.decimal<15, 2>
        %25 = db.compare gt %23 : !db.decimal<15, 2>, %24 : !db.decimal<15, 2>
        relalg.return %25 : i1
      }
      %13 = relalg.projection all [@lineitem::@l_orderkey] %12
      %14 = relalg.getcol %arg0 @orders::@o_orderkey : i32
      %15 = relalg.in %14 : i32, %13
      %16 = relalg.getcol %arg0 @customer::@c_custkey : i32
      %17 = relalg.getcol %arg0 @orders::@o_custkey : i32
      %18 = db.compare eq %16 : i32, %17 : i32
      %19 = relalg.getcol %arg0 @orders::@o_orderkey : i32
      %20 = relalg.getcol %arg0 @lineitem::@l_orderkey : i32
      %21 = db.compare eq %19 : i32, %20 : i32
      %22 = db.and %15, %18, %21 : i1, i1, i1
      relalg.return %22 : i1
    }
    %6 = relalg.aggregation @aggr1 %5 [@customer::@c_name,@customer::@c_custkey,@orders::@o_orderkey,@orders::@o_orderdate,@orders::@o_totalprice] computes : [@tmp_attr1({type = !db.decimal<15, 2>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %10 = relalg.aggrfn sum @lineitem::@l_quantity %arg0 : !db.decimal<15, 2>
      relalg.return %10 : !db.decimal<15, 2>
    }
    %7 = relalg.sort %6 [(@orders::@o_totalprice,desc),(@orders::@o_orderdate,asc)]
    %8 = relalg.limit 100 %7
    %9 = relalg.materialize %8 [@customer::@c_name,@customer::@c_custkey,@orders::@o_orderkey,@orders::@o_orderdate,@orders::@o_totalprice,@aggr1::@tmp_attr1] => ["c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice", "sum"] : !dsa.table
    return %9 : !dsa.table
  }
}

