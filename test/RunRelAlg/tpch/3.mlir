//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                    l_orderkey  |                       revenue  |                   o_orderdate  |                o_shippriority  |
//CHECK: -------------------------------------------------------------------------------------------------------------------------------------
//CHECK: |                        223140  |                     355369.04  |                    1995-03-14  |                             0  |
//CHECK: |                        584291  |                     354494.71  |                    1995-02-21  |                             0  |
//CHECK: |                        405063  |                     353125.42  |                    1995-03-03  |                             0  |
//CHECK: |                        573861  |                     351238.24  |                    1995-03-09  |                             0  |
//CHECK: |                        554757  |                     349181.72  |                    1995-03-14  |                             0  |
//CHECK: |                        506021  |                     321075.55  |                    1995-03-10  |                             0  |
//CHECK: |                        121604  |                     318576.39  |                    1995-03-07  |                             0  |
//CHECK: |                        108514  |                     314967.05  |                    1995-02-20  |                             0  |
//CHECK: |                        462502  |                     312604.51  |                    1995-03-08  |                             0  |
//CHECK: |                        178727  |                     309728.91  |                    1995-02-25  |                             0  |
module {
  func @main() -> !db.table {
    %0 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
    %1 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.selection %4 (%arg0: !relalg.tuple){
      %11 = relalg.getattr %arg0 @customer::@c_mktsegment : !db.string
      %12 = db.constant("BUILDING") : !db.string
      %13 = db.compare eq %11 : !db.string, %12 : !db.string
      %14 = relalg.getattr %arg0 @customer::@c_custkey : i32
      %15 = relalg.getattr %arg0 @orders::@o_custkey : i32
      %16 = db.compare eq %14 : i32, %15 : i32
      %17 = relalg.getattr %arg0 @lineitem::@l_orderkey : i32
      %18 = relalg.getattr %arg0 @orders::@o_orderkey : i32
      %19 = db.compare eq %17 : i32, %18 : i32
      %20 = relalg.getattr %arg0 @orders::@o_orderdate : !db.date<day>
      %21 = db.constant("1995-03-15") : !db.date<day>
      %22 = db.compare lt %20 : !db.date<day>, %21 : !db.date<day>
      %23 = relalg.getattr %arg0 @lineitem::@l_shipdate : !db.date<day>
      %24 = db.constant("1995-03-15") : !db.date<day>
      %25 = db.compare gt %23 : !db.date<day>, %24 : !db.date<day>
      %26 = db.and %13:i1,%16:i1,%19:i1,%22:i1,%25:i1
      relalg.return %26 : i1
    }
    %6 = relalg.map @map0 %5 (%arg0: !relalg.tuple){
      %11 = relalg.getattr %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %12 = db.constant(1 : i32) : !db.decimal<15, 2>
      %13 = relalg.getattr %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %14 = db.sub %12 : !db.decimal<15, 2>, %13 : !db.decimal<15, 2>
      %15 = db.mul %11 : !db.decimal<15, 2>, %14 : !db.decimal<15, 2>
      %16 = relalg.addattr %arg0, @tmp_attr1({type = !db.decimal<15, 2>}) %15
      relalg.return %16 : !relalg.tuple
    }
    %7 = relalg.aggregation @aggr0 %6 [@lineitem::@l_orderkey,@orders::@o_orderdate,@orders::@o_shippriority] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %11 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.nullable<!db.decimal<15, 2>>
      %12 = relalg.addattr %arg1, @tmp_attr0({type = !db.nullable<!db.decimal<15, 2>>}) %11
      relalg.return %12 : !relalg.tuple
    }
    %8 = relalg.sort %7 [(@aggr0::@tmp_attr0,desc),(@orders::@o_orderdate,asc)]
    %9 = relalg.limit 10 %8
    %10 = relalg.materialize %9 [@lineitem::@l_orderkey,@aggr0::@tmp_attr0,@orders::@o_orderdate,@orders::@o_shippriority] => ["l_orderkey", "revenue", "o_orderdate", "o_shippriority"] : !db.table
    return %10 : !db.table
  }
}

