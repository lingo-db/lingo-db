module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %1 = relalg.basetable @part  {table_identifier = "part"} columns: {p_brand => @p_brand({type = !db.string}), p_comment => @p_comment({type = !db.string}), p_container => @p_container({type = !db.string}), p_mfgr => @p_mfgr({type = !db.string}), p_name => @p_name({type = !db.string}), p_partkey => @p_partkey({type = i32}), p_retailprice => @p_retailprice({type = !db.decimal<15, 2>}), p_size => @p_size({type = i32}), p_type => @p_type({type = !db.string})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.selection %2 (%arg0: !relalg.tuple){
      %7 = relalg.getcol %arg0 @part::@p_partkey : i32
      %8 = relalg.getcol %arg0 @lineitem::@l_partkey : i32
      %9 = db.compare eq %7 : i32, %8 : i32
      %10 = relalg.getcol %arg0 @part::@p_brand : !db.string
      %11 = db.constant("Brand#23") : !db.string
      %12 = db.compare eq %10 : !db.string, %11 : !db.string
      %13 = relalg.getcol %arg0 @part::@p_container : !db.string
      %14 = db.constant("MED BOX") : !db.string
      %15 = db.compare eq %13 : !db.string, %14 : !db.string
      %16 = relalg.getcol %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %17 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
      %18 = relalg.selection %17 (%arg1: !relalg.tuple){
        %25 = relalg.getcol %arg1 @lineitem::@l_partkey : i32
        %26 = relalg.getcol %arg1 @part::@p_partkey : i32
        %27 = db.compare eq %25 : i32, %26 : i32
        relalg.return %27 : i1
      }
      %19 = relalg.aggregation @aggr0 %18 [] computes : [@tmp_attr0({type = !db.nullable<!db.decimal<15, 2>>})] (%arg1: !relalg.tuplestream,%arg2: !relalg.tuple){
        %25 = relalg.aggrfn avg @lineitem::@l_quantity %arg1 : !db.nullable<!db.decimal<15, 2>>
        relalg.return %25 : !db.nullable<!db.decimal<15, 2>>
      }
      %20 = relalg.map @map0 %19 computes : [@tmp_attr1({type = !db.nullable<!db.decimal<15, 3>>})] (%arg1: !relalg.tuple){
        %25 = db.constant("0.2") : !db.decimal<2, 1>
        %26 = relalg.getcol %arg1 @aggr0::@tmp_attr0 : !db.nullable<!db.decimal<15, 2>>
        %27 = db.mul %25 : !db.decimal<2, 1>, %26 : !db.nullable<!db.decimal<15, 2>>
        relalg.return %27 : !db.nullable<!db.decimal<15, 3>>
      }
      %21 = relalg.getscalar @map0::@tmp_attr1 %20 : !db.nullable<!db.decimal<15, 3>>
      %22 = db.cast %16 : !db.decimal<15, 2> -> !db.decimal<15, 3>
      %23 = db.compare lt %22 : !db.decimal<15, 3>, %21 : !db.nullable<!db.decimal<15, 3>>
      %24 = db.and %9, %12, %15, %23 : i1, i1, i1, !db.nullable<i1>
      relalg.return %24 : !db.nullable<i1>
    }
    %4 = relalg.aggregation @aggr1 %3 [] computes : [@tmp_attr2({type = !db.nullable<!db.decimal<15, 2>>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %7 = relalg.aggrfn sum @lineitem::@l_extendedprice %arg0 : !db.nullable<!db.decimal<15, 2>>
      relalg.return %7 : !db.nullable<!db.decimal<15, 2>>
    }
    %5 = relalg.map @map1 %4 computes : [@tmp_attr3({type = !db.nullable<!db.decimal<15, 2>>})] (%arg0: !relalg.tuple){
      %7 = relalg.getcol %arg0 @aggr1::@tmp_attr2 : !db.nullable<!db.decimal<15, 2>>
      %8 = db.constant("7.0") : !db.decimal<15, 2>
      %9 = db.div %7 : !db.nullable<!db.decimal<15, 2>>, %8 : !db.decimal<15, 2>
      relalg.return %9 : !db.nullable<!db.decimal<15, 2>>
    }
    %6 = relalg.materialize %5 [@map1::@tmp_attr3] => ["avg_yearly"] : !dsa.table
    return %6 : !dsa.table
  }
}
