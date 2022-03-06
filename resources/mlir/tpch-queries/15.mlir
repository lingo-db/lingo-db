module {
  func @main() -> !db.table {
    %0 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
    %1 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %2 = relalg.selection %1 (%arg0: !relalg.tuple){
      %9 = relalg.getattr %arg0 @lineitem::@l_shipdate : !db.date<day>
      %10 = db.constant("1996-01-01") : !db.date<day>
      %11 = db.compare gte %9 : !db.date<day>, %10 : !db.date<day>
      %12 = relalg.getattr %arg0 @lineitem::@l_shipdate : !db.date<day>
      %13 = db.constant("1996-04-01") : !db.date<day>
      %14 = db.compare lt %12 : !db.date<day>, %13 : !db.date<day>
      %15 = db.and %11:i1,%14:i1
      relalg.return %15 : i1
    }
    %3 = relalg.map @map0 %2 (%arg0: !relalg.tuple){
      %9 = relalg.getattr %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %10 = db.constant(1 : i32) : !db.decimal<15, 2>
      %11 = relalg.getattr %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %12 = db.sub %10 : !db.decimal<15, 2>, %11 : !db.decimal<15, 2>
      %13 = db.mul %9 : !db.decimal<15, 2>, %12 : !db.decimal<15, 2>
      %14 = relalg.addattr %arg0, @tmp_attr1({type = !db.decimal<15, 2>}) %13
      relalg.return %14 : !relalg.tuple
    }
    %4 = relalg.aggregation @aggr0 %3 [@lineitem::@l_suppkey] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %9 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.decimal<15, 2>
      %10 = relalg.addattr %arg1, @tmp_attr0({type = !db.decimal<15, 2>}) %9
      relalg.return %10 : !relalg.tuple
    }
    %5 = relalg.crossproduct %0, %4
    %6 = relalg.selection %5 (%arg0: !relalg.tuple){
      %9 = relalg.getattr %arg0 @supplier::@s_suppkey : i32
      %10 = relalg.getattr %arg0 @lineitem::@l_suppkey : i32
      %11 = db.compare eq %9 : i32, %10 : i32
      %12 = relalg.getattr %arg0 @aggr0::@tmp_attr0 : !db.decimal<15, 2>
      %13 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
      %14 = relalg.selection %13 (%arg1: !relalg.tuple){
        %21 = relalg.getattr %arg1 @lineitem::@l_shipdate : !db.date<day>
        %22 = db.constant("1996-01-01") : !db.date<day>
        %23 = db.compare gte %21 : !db.date<day>, %22 : !db.date<day>
        %24 = relalg.getattr %arg1 @lineitem::@l_shipdate : !db.date<day>
        %25 = db.constant("1996-04-01") : !db.date<day>
        %26 = db.compare lt %24 : !db.date<day>, %25 : !db.date<day>
        %27 = db.and %23:i1,%26:i1
        relalg.return %27 : i1
      }
      %15 = relalg.map @map1 %14 (%arg1: !relalg.tuple){
        %21 = relalg.getattr %arg1 @lineitem::@l_extendedprice : !db.decimal<15, 2>
        %22 = db.constant(1 : i32) : !db.decimal<15, 2>
        %23 = relalg.getattr %arg1 @lineitem::@l_discount : !db.decimal<15, 2>
        %24 = db.sub %22 : !db.decimal<15, 2>, %23 : !db.decimal<15, 2>
        %25 = db.mul %21 : !db.decimal<15, 2>, %24 : !db.decimal<15, 2>
        %26 = relalg.addattr %arg1, @tmp_attr3({type = !db.decimal<15, 2>}) %25
        relalg.return %26 : !relalg.tuple
      }
      %16 = relalg.aggregation @aggr1 %15 [@lineitem::@l_suppkey] (%arg1: !relalg.tuplestream,%arg2: !relalg.tuple){
        %21 = relalg.aggrfn sum @map1::@tmp_attr3 %arg1 : !db.decimal<15, 2>
        %22 = relalg.addattr %arg2, @tmp_attr2({type = !db.decimal<15, 2>}) %21
        relalg.return %22 : !relalg.tuple
      }
      %17 = relalg.aggregation @aggr2 %16 [] (%arg1: !relalg.tuplestream,%arg2: !relalg.tuple){
        %21 = relalg.aggrfn max @aggr1::@tmp_attr2 %arg1 : !db.nullable<!db.decimal<15, 2>>
        %22 = relalg.addattr %arg2, @tmp_attr4({type = !db.nullable<!db.decimal<15, 2>>}) %21
        relalg.return %22 : !relalg.tuple
      }
      %18 = relalg.getscalar @aggr2::@tmp_attr4 %17 : !db.nullable<!db.decimal<15, 2>>
      %19 = db.compare eq %12 : !db.decimal<15, 2>, %18 : !db.nullable<!db.decimal<15, 2>>
      %20 = db.and %11:i1,%19:!db.nullable<i1>
      relalg.return %20 : !db.nullable<i1>
    }
    %7 = relalg.sort %6 [(@supplier::@s_suppkey,asc)]
    %8 = relalg.materialize %7 [@supplier::@s_suppkey,@supplier::@s_name,@supplier::@s_address,@supplier::@s_phone,@aggr0::@tmp_attr0] => ["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"] : !db.table
    return %8 : !db.table
  }
}
