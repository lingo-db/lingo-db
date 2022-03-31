module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
    %1 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.selection %2 (%arg0: !relalg.tuple){
      %6 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = i32}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = i32}), ps_suppkey => @ps_suppkey({type = i32}), ps_supplycost => @ps_supplycost({type = !db.decimal<15, 2>})}
      %7 = relalg.selection %6 (%arg1: !relalg.tuple){
        %18 = relalg.basetable @part  {table_identifier = "part"} columns: {p_brand => @p_brand({type = !db.string}), p_comment => @p_comment({type = !db.string}), p_container => @p_container({type = !db.string}), p_mfgr => @p_mfgr({type = !db.string}), p_name => @p_name({type = !db.string}), p_partkey => @p_partkey({type = i32}), p_retailprice => @p_retailprice({type = !db.decimal<15, 2>}), p_size => @p_size({type = i32}), p_type => @p_type({type = !db.string})}
        %19 = relalg.selection %18 (%arg2: !relalg.tuple){
          %32 = relalg.getcol %arg2 @part::@p_name : !db.string
          %33 = db.constant("forest%") : !db.string
          %34 = db.compare like %32 : !db.string, %33 : !db.string
          relalg.return %34 : i1
        }
        %20 = relalg.projection all [@part::@p_partkey] %19
        %21 = relalg.getcol %arg1 @partsupp::@ps_partkey : i32
        %22 = relalg.in %21 : i32, %20
        %23 = relalg.getcol %arg1 @partsupp::@ps_availqty : i32
        %24 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
        %25 = relalg.selection %24 (%arg2: !relalg.tuple){
          %32 = relalg.getcol %arg2 @lineitem::@l_partkey : i32
          %33 = relalg.getcol %arg2 @partsupp::@ps_partkey : i32
          %34 = db.compare eq %32 : i32, %33 : i32
          %35 = relalg.getcol %arg2 @lineitem::@l_suppkey : i32
          %36 = relalg.getcol %arg2 @partsupp::@ps_suppkey : i32
          %37 = db.compare eq %35 : i32, %36 : i32
          %38 = relalg.getcol %arg2 @lineitem::@l_shipdate : !db.date<day>
          %39 = db.constant("1994-01-01") : !db.date<day>
          %40 = db.compare gte %38 : !db.date<day>, %39 : !db.date<day>
          %41 = relalg.getcol %arg2 @lineitem::@l_shipdate : !db.date<day>
          %42 = db.constant("1995-01-01") : !db.date<day>
          %43 = db.compare lt %41 : !db.date<day>, %42 : !db.date<day>
          %44 = db.and %34, %37, %40, %43 : i1, i1, i1, i1
          relalg.return %44 : i1
        }
        %26 = relalg.aggregation @aggr0 %25 [] (%arg2: !relalg.tuplestream,%arg3: !relalg.tuple){
          %32 = relalg.aggrfn sum @lineitem::@l_quantity %arg2 : !db.nullable<!db.decimal<15, 2>>
          %33 = relalg.addcol %arg3, @tmp_attr0({type = !db.nullable<!db.decimal<15, 2>>}) %32
          relalg.return %33 : !relalg.tuple
        }
        %27 = relalg.map @map0 %26 (%arg2: !relalg.tuple){
          %32 = db.constant("0.5") : !db.decimal<15, 2>
          %33 = relalg.getcol %arg2 @aggr0::@tmp_attr0 : !db.nullable<!db.decimal<15, 2>>
          %34 = db.mul %32 : !db.decimal<15, 2>, %33 : !db.nullable<!db.decimal<15, 2>>
          %35 = relalg.addcol %arg2, @tmp_attr1({type = !db.nullable<!db.decimal<15, 2>>}) %34
          relalg.return %35 : !relalg.tuple
        }
        %28 = relalg.getscalar @map0::@tmp_attr1 %27 : !db.nullable<!db.decimal<15, 2>>
        %29 = db.cast %23 : i32 -> !db.decimal<15, 2>
        %30 = db.compare gt %29 : !db.decimal<15, 2>, %28 : !db.nullable<!db.decimal<15, 2>>
        %31 = db.and %22, %30 : i1, !db.nullable<i1>
        relalg.return %31 : !db.nullable<i1>
      }
      %8 = relalg.projection all [@partsupp::@ps_suppkey] %7
      %9 = relalg.getcol %arg0 @supplier::@s_suppkey : i32
      %10 = relalg.in %9 : i32, %8
      %11 = relalg.getcol %arg0 @supplier::@s_nationkey : i32
      %12 = relalg.getcol %arg0 @nation::@n_nationkey : i32
      %13 = db.compare eq %11 : i32, %12 : i32
      %14 = relalg.getcol %arg0 @nation::@n_name : !db.string
      %15 = db.constant("CANADA") : !db.string
      %16 = db.compare eq %14 : !db.string, %15 : !db.string
      %17 = db.and %10, %13, %16 : i1, i1, i1
      relalg.return %17 : i1
    }
    %4 = relalg.sort %3 [(@supplier::@s_name,asc)]
    %5 = relalg.materialize %4 [@supplier::@s_name,@supplier::@s_address] => ["s_name", "s_address"] : !dsa.table
    return %5 : !dsa.table
  }
}
