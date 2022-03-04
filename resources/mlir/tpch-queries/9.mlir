module {
  func @main() -> !db.table {
    %0 = relalg.basetable @part  {table_identifier = "part"} columns: {p_brand => @p_brand({type = !db.string}), p_comment => @p_comment({type = !db.string}), p_container => @p_container({type = !db.string}), p_mfgr => @p_mfgr({type = !db.string}), p_name => @p_name({type = !db.string}), p_partkey => @p_partkey({type = i32}), p_retailprice => @p_retailprice({type = !db.decimal<15, 2>}), p_size => @p_size({type = i32}), p_type => @p_type({type = !db.string})}
    %1 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = i32}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = i32}), ps_suppkey => @ps_suppkey({type = i32}), ps_supplycost => @ps_supplycost({type = !db.decimal<15, 2>})}
    %6 = relalg.crossproduct %4, %5
    %7 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %8 = relalg.crossproduct %6, %7
    %9 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %10 = relalg.crossproduct %8, %9
    %11 = relalg.selection %10 (%arg0: !relalg.tuple){
      %16 = relalg.getattr %arg0 @supplier::@s_suppkey : i32
      %17 = relalg.getattr %arg0 @lineitem::@l_suppkey : i32
      %18 = db.compare eq %16 : i32, %17 : i32
      %19 = relalg.getattr %arg0 @partsupp::@ps_suppkey : i32
      %20 = relalg.getattr %arg0 @lineitem::@l_suppkey : i32
      %21 = db.compare eq %19 : i32, %20 : i32
      %22 = relalg.getattr %arg0 @partsupp::@ps_partkey : i32
      %23 = relalg.getattr %arg0 @lineitem::@l_partkey : i32
      %24 = db.compare eq %22 : i32, %23 : i32
      %25 = relalg.getattr %arg0 @part::@p_partkey : i32
      %26 = relalg.getattr %arg0 @lineitem::@l_partkey : i32
      %27 = db.compare eq %25 : i32, %26 : i32
      %28 = relalg.getattr %arg0 @orders::@o_orderkey : i32
      %29 = relalg.getattr %arg0 @lineitem::@l_orderkey : i32
      %30 = db.compare eq %28 : i32, %29 : i32
      %31 = relalg.getattr %arg0 @supplier::@s_nationkey : i32
      %32 = relalg.getattr %arg0 @nation::@n_nationkey : i32
      %33 = db.compare eq %31 : i32, %32 : i32
      %34 = relalg.getattr %arg0 @part::@p_name : !db.string
      %35 = db.constant("%green%") : !db.string
      %36 = db.compare like %34 : !db.string, %35 : !db.string
      %37 = db.and %18:i1,%21:i1,%24:i1,%27:i1,%30:i1,%33:i1,%36:i1
      relalg.return %37 : i1
    }
    %12 = relalg.map @map0 %11 (%arg0: !relalg.tuple){
      %16 = relalg.getattr %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %17 = db.constant(1 : i32) : !db.decimal<15, 2>
      %18 = relalg.getattr %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %19 = db.sub %17 : !db.decimal<15, 2>, %18 : !db.decimal<15, 2>
      %20 = db.mul %16 : !db.decimal<15, 2>, %19 : !db.decimal<15, 2>
      %21 = relalg.getattr %arg0 @partsupp::@ps_supplycost : !db.decimal<15, 2>
      %22 = relalg.getattr %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %23 = db.mul %21 : !db.decimal<15, 2>, %22 : !db.decimal<15, 2>
      %24 = db.sub %20 : !db.decimal<15, 2>, %23 : !db.decimal<15, 2>
      %25 = relalg.addattr %arg0, @tmp_attr1({type = !db.decimal<15, 2>}) %24
      %26 = relalg.getattr %25 @orders::@o_orderdate : !db.date<day>
      %27 = db.date_extract year, %26 : <day>
      %28 = relalg.addattr %25, @tmp_attr0({type = i64}) %27
      relalg.return %28 : !relalg.tuple
    }
    %13 = relalg.aggregation @aggr0 %12 [@nation::@n_name,@map0::@tmp_attr0] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %16 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.nullable<!db.decimal<15, 2>>
      %17 = relalg.addattr %arg1, @tmp_attr2({type = !db.nullable<!db.decimal<15, 2>>}) %16
      relalg.return %17 : !relalg.tuple
    }
    %14 = relalg.sort %13 [(@nation::@n_name,asc),(@map0::@tmp_attr0,desc)]
    %15 = relalg.materialize %14 [@nation::@n_name,@map0::@tmp_attr0,@aggr0::@tmp_attr2] => ["nation", "o_year", "sum_profit"] : !db.table
    return %15 : !db.table
  }
}
