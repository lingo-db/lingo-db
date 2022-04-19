module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %1 = relalg.basetable @part  {table_identifier = "part"} columns: {p_brand => @p_brand({type = !db.string}), p_comment => @p_comment({type = !db.string}), p_container => @p_container({type = !db.string}), p_mfgr => @p_mfgr({type = !db.string}), p_name => @p_name({type = !db.string}), p_partkey => @p_partkey({type = i32}), p_retailprice => @p_retailprice({type = !db.decimal<15, 2>}), p_size => @p_size({type = i32}), p_type => @p_type({type = !db.string})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.selection %2 (%arg0: !relalg.tuple){
      %8 = relalg.getcol %arg0 @lineitem::@l_partkey : i32
      %9 = relalg.getcol %arg0 @part::@p_partkey : i32
      %10 = db.compare eq %8 : i32, %9 : i32
      %11 = relalg.getcol %arg0 @lineitem::@l_shipdate : !db.date<day>
      %12 = db.constant("1995-09-01") : !db.date<day>
      %13 = db.compare gte %11 : !db.date<day>, %12 : !db.date<day>
      %14 = relalg.getcol %arg0 @lineitem::@l_shipdate : !db.date<day>
      %15 = db.constant("1995-10-01") : !db.date<day>
      %16 = db.compare lt %14 : !db.date<day>, %15 : !db.date<day>
      %17 = db.and %10, %13, %16 : i1, i1, i1
      relalg.return %17 : i1
    }
    %4 = relalg.map @map0 %3 computes : [@tmp_attr3({type = !db.decimal<15, 4>}),@tmp_attr1({type = !db.decimal<15, 4>})] (%arg0: !relalg.tuple){
      %8 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %9 = db.constant(1 : i32) : !db.decimal<15, 2>
      %10 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %11 = db.sub %9 : !db.decimal<15, 2>, %10 : !db.decimal<15, 2>
      %12 = db.mul %8 : !db.decimal<15, 2>, %11 : !db.decimal<15, 2>
      %13 = relalg.getcol %arg0 @part::@p_type : !db.string
      %14 = db.constant("PROMO%") : !db.string
      %15 = db.runtime_call "Like"(%13, %14) : (!db.string, !db.string) -> i1
      %16 = scf.if %15 -> (!db.decimal<15, 4>) {
        %17 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
        %18 = db.constant(1 : i32) : !db.decimal<15, 2>
        %19 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
        %20 = db.sub %18 : !db.decimal<15, 2>, %19 : !db.decimal<15, 2>
        %21 = db.mul %17 : !db.decimal<15, 2>, %20 : !db.decimal<15, 2>
        scf.yield %21 : !db.decimal<15, 4>
      } else {
        %17 = db.constant(0 : i32) : !db.decimal<15, 4>
        scf.yield %17 : !db.decimal<15, 4>
      }
      relalg.return %12, %16 : !db.decimal<15, 4>, !db.decimal<15, 4>
    }
    %5 = relalg.aggregation @aggr0 %4 [] computes : [@tmp_attr2({type = !db.nullable<!db.decimal<15, 4>>}),@tmp_attr0({type = !db.nullable<!db.decimal<15, 4>>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %8 = relalg.aggrfn sum @map0::@tmp_attr3 %arg0 : !db.nullable<!db.decimal<15, 4>>
      %9 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.nullable<!db.decimal<15, 4>>
      relalg.return %8, %9 : !db.nullable<!db.decimal<15, 4>>, !db.nullable<!db.decimal<15, 4>>
    }
    %6 = relalg.map @map1 %5 computes : [@tmp_attr4({type = !db.nullable<!db.decimal<15, 6>>})] (%arg0: !relalg.tuple){
      %8 = db.constant("100.00") : !db.decimal<5, 2>
      %9 = relalg.getcol %arg0 @aggr0::@tmp_attr0 : !db.nullable<!db.decimal<15, 4>>
      %10 = db.mul %8 : !db.decimal<5, 2>, %9 : !db.nullable<!db.decimal<15, 4>>
      %11 = relalg.getcol %arg0 @aggr0::@tmp_attr2 : !db.nullable<!db.decimal<15, 4>>
      %12 = db.cast %11 : !db.nullable<!db.decimal<15, 4>> -> !db.nullable<!db.decimal<15, 6>>
      %13 = db.div %10 : !db.nullable<!db.decimal<15, 6>>, %12 : !db.nullable<!db.decimal<15, 6>>
      relalg.return %13 : !db.nullable<!db.decimal<15, 6>>
    }
    %7 = relalg.materialize %6 [@map1::@tmp_attr4] => ["promo_revenue"] : !dsa.table
    return %7 : !dsa.table
  }
}
