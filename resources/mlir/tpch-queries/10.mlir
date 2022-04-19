module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
    %1 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %6 = relalg.crossproduct %4, %5
    %7 = relalg.selection %6 (%arg0: !relalg.tuple){
      %13 = relalg.getcol %arg0 @customer::@c_custkey : i32
      %14 = relalg.getcol %arg0 @orders::@o_custkey : i32
      %15 = db.compare eq %13 : i32, %14 : i32
      %16 = relalg.getcol %arg0 @lineitem::@l_orderkey : i32
      %17 = relalg.getcol %arg0 @orders::@o_orderkey : i32
      %18 = db.compare eq %16 : i32, %17 : i32
      %19 = relalg.getcol %arg0 @orders::@o_orderdate : !db.date<day>
      %20 = db.constant("1993-10-01") : !db.date<day>
      %21 = db.compare gte %19 : !db.date<day>, %20 : !db.date<day>
      %22 = relalg.getcol %arg0 @orders::@o_orderdate : !db.date<day>
      %23 = db.constant("1994-01-01") : !db.date<day>
      %24 = db.compare lt %22 : !db.date<day>, %23 : !db.date<day>
      %25 = relalg.getcol %arg0 @lineitem::@l_returnflag : !db.char<1>
      %26 = db.constant("R") : !db.char<1>
      %27 = db.compare eq %25 : !db.char<1>, %26 : !db.char<1>
      %28 = relalg.getcol %arg0 @customer::@c_nationkey : i32
      %29 = relalg.getcol %arg0 @nation::@n_nationkey : i32
      %30 = db.compare eq %28 : i32, %29 : i32
      %31 = db.and %15, %18, %21, %24, %27, %30 : i1, i1, i1, i1, i1, i1
      relalg.return %31 : i1
    }
    %8 = relalg.map @map0 %7 computes : [@tmp_attr1({type = !db.decimal<15, 4>})] (%arg0: !relalg.tuple){
      %13 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %14 = db.constant(1 : i32) : !db.decimal<15, 2>
      %15 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %16 = db.sub %14 : !db.decimal<15, 2>, %15 : !db.decimal<15, 2>
      %17 = db.mul %13 : !db.decimal<15, 2>, %16 : !db.decimal<15, 2>
      relalg.return %17 : !db.decimal<15, 4>
    }
    %9 = relalg.aggregation @aggr0 %8 [@customer::@c_custkey,@customer::@c_name,@customer::@c_acctbal,@customer::@c_phone,@nation::@n_name,@customer::@c_address,@customer::@c_comment] computes : [@tmp_attr0({type = !db.decimal<15, 4>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %13 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.decimal<15, 4>
      relalg.return %13 : !db.decimal<15, 4>
    }
    %10 = relalg.sort %9 [(@aggr0::@tmp_attr0,desc)]
    %11 = relalg.limit 20 %10
    %12 = relalg.materialize %11 [@customer::@c_custkey,@customer::@c_name,@aggr0::@tmp_attr0,@customer::@c_acctbal,@nation::@n_name,@customer::@c_address,@customer::@c_phone,@customer::@c_comment] => ["c_custkey", "c_name", "revenue", "c_acctbal", "n_name", "c_address", "c_phone", "c_comment"] : !dsa.table
    return %12 : !dsa.table
  }
}
