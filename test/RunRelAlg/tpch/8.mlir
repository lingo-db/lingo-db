//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                        o_year  |                     mkt_share  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                          1995  |                          0.02  |
//CHECK: |                          1996  |                          0.01  |
module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @part  {table_identifier = "part"} columns: {p_brand => @p_brand({type = !db.string}), p_comment => @p_comment({type = !db.string}), p_container => @p_container({type = !db.string}), p_mfgr => @p_mfgr({type = !db.string}), p_name => @p_name({type = !db.string}), p_partkey => @p_partkey({type = i32}), p_retailprice => @p_retailprice({type = !db.decimal<15, 2>}), p_size => @p_size({type = i32}), p_type => @p_type({type = !db.string})}
    %1 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %6 = relalg.crossproduct %4, %5
    %7 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
    %8 = relalg.crossproduct %6, %7
    %9 = relalg.basetable @n1  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %10 = relalg.crossproduct %8, %9
    %11 = relalg.basetable @n2  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %12 = relalg.crossproduct %10, %11
    %13 = relalg.basetable @region  {table_identifier = "region"} columns: {r_comment => @r_comment({type = !db.nullable<!db.string>}), r_name => @r_name({type = !db.string}), r_regionkey => @r_regionkey({type = i32})}
    %14 = relalg.crossproduct %12, %13
    %15 = relalg.selection %14 (%arg0: !relalg.tuple){
      %22 = relalg.getcol %arg0 @part::@p_partkey : i32
      %23 = relalg.getcol %arg0 @lineitem::@l_partkey : i32
      %24 = db.compare eq %22 : i32, %23 : i32
      %25 = relalg.getcol %arg0 @supplier::@s_suppkey : i32
      %26 = relalg.getcol %arg0 @lineitem::@l_suppkey : i32
      %27 = db.compare eq %25 : i32, %26 : i32
      %28 = relalg.getcol %arg0 @lineitem::@l_orderkey : i32
      %29 = relalg.getcol %arg0 @orders::@o_orderkey : i32
      %30 = db.compare eq %28 : i32, %29 : i32
      %31 = relalg.getcol %arg0 @orders::@o_custkey : i32
      %32 = relalg.getcol %arg0 @customer::@c_custkey : i32
      %33 = db.compare eq %31 : i32, %32 : i32
      %34 = relalg.getcol %arg0 @customer::@c_nationkey : i32
      %35 = relalg.getcol %arg0 @n1::@n_nationkey : i32
      %36 = db.compare eq %34 : i32, %35 : i32
      %37 = relalg.getcol %arg0 @n1::@n_regionkey : i32
      %38 = relalg.getcol %arg0 @region::@r_regionkey : i32
      %39 = db.compare eq %37 : i32, %38 : i32
      %40 = relalg.getcol %arg0 @region::@r_name : !db.string
      %41 = db.constant("AMERICA") : !db.string
      %42 = db.compare eq %40 : !db.string, %41 : !db.string
      %43 = relalg.getcol %arg0 @supplier::@s_nationkey : i32
      %44 = relalg.getcol %arg0 @n2::@n_nationkey : i32
      %45 = db.compare eq %43 : i32, %44 : i32
      %46 = relalg.getcol %arg0 @orders::@o_orderdate : !db.date<day>
      %47 = db.constant("1995-01-01") : !db.date<day>
      %48 = db.constant("1996-12-31") : !db.date<day>
      %49 = db.between %46 : !db.date<day> between %47 : !db.date<day>, %48 : !db.date<day>, lowerInclusive : true, upperInclusive : true
      %50 = relalg.getcol %arg0 @part::@p_type : !db.string
      %51 = db.constant("ECONOMY ANODIZED STEEL") : !db.string
      %52 = db.compare eq %50 : !db.string, %51 : !db.string
      %53 = db.and %24, %27, %30, %33, %36, %39, %42, %45, %49, %52 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
      relalg.return %53 : i1
    }
    %16 = relalg.map @map0 %15 computes : [@tmp_attr1({type = !db.decimal<15, 2>}),@tmp_attr0({type = i64})] (%arg0: !relalg.tuple){
      %22 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %23 = db.constant(1 : i32) : !db.decimal<15, 2>
      %24 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %25 = db.sub %23 : !db.decimal<15, 2>, %24 : !db.decimal<15, 2>
      %26 = db.mul %22 : !db.decimal<15, 2>, %25 : !db.decimal<15, 2>
      %27 = db.constant("year") : !db.char<4>
      %28 = relalg.getcol %arg0 @orders::@o_orderdate : !db.date<day>
      %29 = db.runtime_call "ExtractFromDate"(%27, %28) : (!db.char<4>, !db.date<day>) -> i64
      relalg.return %26, %29 : !db.decimal<15, 2>, i64
    }
    %17 = relalg.map @map1 %16 computes : [@tmp_attr3({type = !db.decimal<15, 2>})] (%arg0: !relalg.tuple){
      %22 = relalg.getcol %arg0 @n2::@n_name : !db.string
      %23 = db.constant("BRAZIL") : !db.string
      %24 = db.compare eq %22 : !db.string, %23 : !db.string
      %25 = scf.if %24 -> (!db.decimal<15, 2>) {
        %26 = relalg.getcol %arg0 @map0::@tmp_attr1 : !db.decimal<15, 2>
        scf.yield %26 : !db.decimal<15, 2>
      } else {
        %26 = db.constant(0 : i32) : !db.decimal<15, 2>
        scf.yield %26 : !db.decimal<15, 2>
      }
      relalg.return %25 : !db.decimal<15, 2>
    }
    %18 = relalg.aggregation @aggr0 %17 [@map0::@tmp_attr0] computes : [@tmp_attr4({type = !db.decimal<15, 2>}),@tmp_attr2({type = !db.decimal<15, 2>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %22 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.decimal<15, 2>
      %23 = relalg.aggrfn sum @map1::@tmp_attr3 %arg0 : !db.decimal<15, 2>
      relalg.return %22, %23 : !db.decimal<15, 2>, !db.decimal<15, 2>
    }
    %19 = relalg.map @map2 %18 computes : [@tmp_attr5({type = !db.decimal<15, 2>})] (%arg0: !relalg.tuple){
      %22 = relalg.getcol %arg0 @aggr0::@tmp_attr2 : !db.decimal<15, 2>
      %23 = relalg.getcol %arg0 @aggr0::@tmp_attr4 : !db.decimal<15, 2>
      %24 = db.div %22 : !db.decimal<15, 2>, %23 : !db.decimal<15, 2>
      relalg.return %24 : !db.decimal<15, 2>
    }
    %20 = relalg.sort %19 [(@map0::@tmp_attr0,asc)]
    %21 = relalg.materialize %20 [@map0::@tmp_attr0,@map2::@tmp_attr5] => ["o_year", "mkt_share"] : !dsa.table
    return %21 : !dsa.table
  }
}

