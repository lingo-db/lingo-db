module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
    %1 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
    %6 = relalg.crossproduct %4, %5
    %7 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %8 = relalg.crossproduct %6, %7
    %9 = relalg.basetable @region  {table_identifier = "region"} columns: {r_comment => @r_comment({type = !db.nullable<!db.string>}), r_name => @r_name({type = !db.string}), r_regionkey => @r_regionkey({type = i32})}
    %10 = relalg.crossproduct %8, %9
    %11 = relalg.selection %10 (%arg0: !relalg.tuple){
      %16 = relalg.getcol %arg0 @customer::@c_custkey : i32
      %17 = relalg.getcol %arg0 @orders::@o_custkey : i32
      %18 = db.compare eq %16 : i32, %17 : i32
      %19 = relalg.getcol %arg0 @lineitem::@l_orderkey : i32
      %20 = relalg.getcol %arg0 @orders::@o_orderkey : i32
      %21 = db.compare eq %19 : i32, %20 : i32
      %22 = relalg.getcol %arg0 @lineitem::@l_suppkey : i32
      %23 = relalg.getcol %arg0 @supplier::@s_suppkey : i32
      %24 = db.compare eq %22 : i32, %23 : i32
      %25 = relalg.getcol %arg0 @customer::@c_nationkey : i32
      %26 = relalg.getcol %arg0 @supplier::@s_nationkey : i32
      %27 = db.compare eq %25 : i32, %26 : i32
      %28 = relalg.getcol %arg0 @supplier::@s_nationkey : i32
      %29 = relalg.getcol %arg0 @nation::@n_nationkey : i32
      %30 = db.compare eq %28 : i32, %29 : i32
      %31 = relalg.getcol %arg0 @nation::@n_regionkey : i32
      %32 = relalg.getcol %arg0 @region::@r_regionkey : i32
      %33 = db.compare eq %31 : i32, %32 : i32
      %34 = relalg.getcol %arg0 @region::@r_name : !db.string
      %35 = db.constant("ASIA") : !db.string
      %36 = db.compare eq %34 : !db.string, %35 : !db.string
      %37 = relalg.getcol %arg0 @orders::@o_orderdate : !db.date<day>
      %38 = db.constant("1994-01-01") : !db.date<day>
      %39 = db.compare gte %37 : !db.date<day>, %38 : !db.date<day>
      %40 = relalg.getcol %arg0 @orders::@o_orderdate : !db.date<day>
      %41 = db.constant("1995-01-01") : !db.date<day>
      %42 = db.compare lt %40 : !db.date<day>, %41 : !db.date<day>
      %43 = db.and %18, %21, %24, %27, %30, %33, %36, %39, %42 : i1, i1, i1, i1, i1, i1, i1, i1, i1
      relalg.return %43 : i1
    }
    %12 = relalg.map @map0 %11 (%arg0: !relalg.tuple){
      %16 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %17 = db.constant(1 : i32) : !db.decimal<15, 2>
      %18 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %19 = db.sub %17 : !db.decimal<15, 2>, %18 : !db.decimal<15, 2>
      %20 = db.mul %16 : !db.decimal<15, 2>, %19 : !db.decimal<15, 2>
      %21 = relalg.addcol %arg0, @tmp_attr1({type = !db.decimal<15, 2>}) %20
      relalg.return %21 : !relalg.tuple
    }
    %13 = relalg.aggregation @aggr0 %12 [@nation::@n_name] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %16 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.decimal<15, 2>
      %17 = relalg.addcol %arg1, @tmp_attr0({type = !db.decimal<15, 2>}) %16
      relalg.return %17 : !relalg.tuple
    }
    %14 = relalg.sort %13 [(@aggr0::@tmp_attr0,desc)]
    %15 = relalg.materialize %14 [@nation::@n_name,@aggr0::@tmp_attr0] => ["n_name", "revenue"] : !dsa.table
    return %15 : !dsa.table
  }
}
