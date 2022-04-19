module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
    %1 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
    %6 = relalg.crossproduct %4, %5
    %7 = relalg.basetable @n1  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %8 = relalg.crossproduct %6, %7
    %9 = relalg.basetable @n2  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %10 = relalg.crossproduct %8, %9
    %11 = relalg.selection %10 (%arg0: !relalg.tuple){
      %16 = relalg.getcol %arg0 @supplier::@s_suppkey : i32
      %17 = relalg.getcol %arg0 @lineitem::@l_suppkey : i32
      %18 = db.compare eq %16 : i32, %17 : i32
      %19 = relalg.getcol %arg0 @orders::@o_orderkey : i32
      %20 = relalg.getcol %arg0 @lineitem::@l_orderkey : i32
      %21 = db.compare eq %19 : i32, %20 : i32
      %22 = relalg.getcol %arg0 @customer::@c_custkey : i32
      %23 = relalg.getcol %arg0 @orders::@o_custkey : i32
      %24 = db.compare eq %22 : i32, %23 : i32
      %25 = relalg.getcol %arg0 @supplier::@s_nationkey : i32
      %26 = relalg.getcol %arg0 @n1::@n_nationkey : i32
      %27 = db.compare eq %25 : i32, %26 : i32
      %28 = relalg.getcol %arg0 @customer::@c_nationkey : i32
      %29 = relalg.getcol %arg0 @n2::@n_nationkey : i32
      %30 = db.compare eq %28 : i32, %29 : i32
      %31 = relalg.getcol %arg0 @n1::@n_name : !db.string
      %32 = db.constant("FRANCE") : !db.string
      %33 = db.compare eq %31 : !db.string, %32 : !db.string
      %34 = relalg.getcol %arg0 @n2::@n_name : !db.string
      %35 = db.constant("GERMANY") : !db.string
      %36 = db.compare eq %34 : !db.string, %35 : !db.string
      %37 = db.and %33, %36 : i1, i1
      %38 = relalg.getcol %arg0 @n1::@n_name : !db.string
      %39 = db.constant("GERMANY") : !db.string
      %40 = db.compare eq %38 : !db.string, %39 : !db.string
      %41 = relalg.getcol %arg0 @n2::@n_name : !db.string
      %42 = db.constant("FRANCE") : !db.string
      %43 = db.compare eq %41 : !db.string, %42 : !db.string
      %44 = db.and %40, %43 : i1, i1
      %45 = db.or %37, %44 : i1, i1
      %46 = relalg.getcol %arg0 @lineitem::@l_shipdate : !db.date<day>
      %47 = db.constant("1995-01-01") : !db.date<day>
      %48 = db.constant("1996-12-31") : !db.date<day>
      %49 = db.between %46 : !db.date<day> between %47 : !db.date<day>, %48 : !db.date<day>, lowerInclusive : true, upperInclusive : true
      %50 = db.and %18, %21, %24, %27, %30, %45, %49 : i1, i1, i1, i1, i1, i1, i1
      relalg.return %50 : i1
    }
    %12 = relalg.map @map0 %11 computes : [@tmp_attr1({type = !db.decimal<15, 4>}),@tmp_attr0({type = i64})] (%arg0: !relalg.tuple){
      %16 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %17 = db.constant(1 : i32) : !db.decimal<15, 2>
      %18 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %19 = db.sub %17 : !db.decimal<15, 2>, %18 : !db.decimal<15, 2>
      %20 = db.mul %16 : !db.decimal<15, 2>, %19 : !db.decimal<15, 2>
      %21 = db.constant("year") : !db.char<4>
      %22 = relalg.getcol %arg0 @lineitem::@l_shipdate : !db.date<day>
      %23 = db.runtime_call "ExtractFromDate"(%21, %22) : (!db.char<4>, !db.date<day>) -> i64
      relalg.return %20, %23 : !db.decimal<15, 4>, i64
    }
    %13 = relalg.aggregation @aggr0 %12 [@n1::@n_name,@n2::@n_name,@map0::@tmp_attr0] computes : [@tmp_attr2({type = !db.decimal<15, 4>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %16 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.decimal<15, 4>
      relalg.return %16 : !db.decimal<15, 4>
    }
    %14 = relalg.sort %13 [(@n1::@n_name,asc),(@n2::@n_name,asc),(@map0::@tmp_attr0,asc)]
    %15 = relalg.materialize %14 [@n1::@n_name,@n2::@n_name,@map0::@tmp_attr0,@aggr0::@tmp_attr2] => ["supp_nation", "cust_nation", "l_year", "revenue"] : !dsa.table
    return %15 : !dsa.table
  }
}
