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
      %11 = db.constant("Brand#12") : !db.string
      %12 = db.compare eq %10 : !db.string, %11 : !db.string
      %13 = db.constant("SM CASE") : !db.string
      %14 = db.constant("SM BOX") : !db.string
      %15 = db.constant("SM PACK") : !db.string
      %16 = db.constant("SM PKG") : !db.string
      %17 = relalg.getcol %arg0 @part::@p_container : !db.string
      %18 = db.oneof %17 : !db.string ? %13, %14, %15, %16 : !db.string, !db.string, !db.string, !db.string
      %19 = relalg.getcol %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %20 = db.constant(1 : i32) : !db.decimal<15, 2>
      %21 = db.compare gte %19 : !db.decimal<15, 2>, %20 : !db.decimal<15, 2>
      %22 = relalg.getcol %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %23 = db.constant(1 : i32) : i32
      %24 = db.constant(10 : i32) : i32
      %25 = db.add %23 : i32, %24 : i32
      %26 = db.cast %25 : i32 -> !db.decimal<15, 2>
      %27 = db.compare lte %22 : !db.decimal<15, 2>, %26 : !db.decimal<15, 2>
      %28 = relalg.getcol %arg0 @part::@p_size : i32
      %29 = db.constant(1 : i32) : i32
      %30 = db.constant(5 : i32) : i32
      %31 = db.between %28 : i32 between %29 : i32, %30 : i32, lowerInclusive : true, upperInclusive : true
      %32 = db.constant("AIR") : !db.string
      %33 = db.constant("AIR REG") : !db.string
      %34 = relalg.getcol %arg0 @lineitem::@l_shipmode : !db.string
      %35 = db.oneof %34 : !db.string ? %32, %33 : !db.string, !db.string
      %36 = relalg.getcol %arg0 @lineitem::@l_shipinstruct : !db.string
      %37 = db.constant("DELIVER IN PERSON") : !db.string
      %38 = db.compare eq %36 : !db.string, %37 : !db.string
      %39 = db.and %9, %12, %18, %21, %27, %31, %35, %38 : i1, i1, i1, i1, i1, i1, i1, i1
      %40 = relalg.getcol %arg0 @part::@p_partkey : i32
      %41 = relalg.getcol %arg0 @lineitem::@l_partkey : i32
      %42 = db.compare eq %40 : i32, %41 : i32
      %43 = relalg.getcol %arg0 @part::@p_brand : !db.string
      %44 = db.constant("Brand#23") : !db.string
      %45 = db.compare eq %43 : !db.string, %44 : !db.string
      %46 = db.constant("MED BAG") : !db.string
      %47 = db.constant("MED BOX") : !db.string
      %48 = db.constant("MED PKG") : !db.string
      %49 = db.constant("MED PACK") : !db.string
      %50 = relalg.getcol %arg0 @part::@p_container : !db.string
      %51 = db.oneof %50 : !db.string ? %46, %47, %48, %49 : !db.string, !db.string, !db.string, !db.string
      %52 = relalg.getcol %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %53 = db.constant(10 : i32) : !db.decimal<15, 2>
      %54 = db.compare gte %52 : !db.decimal<15, 2>, %53 : !db.decimal<15, 2>
      %55 = relalg.getcol %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %56 = db.constant(10 : i32) : i32
      %57 = db.constant(10 : i32) : i32
      %58 = db.add %56 : i32, %57 : i32
      %59 = db.cast %58 : i32 -> !db.decimal<15, 2>
      %60 = db.compare lte %55 : !db.decimal<15, 2>, %59 : !db.decimal<15, 2>
      %61 = relalg.getcol %arg0 @part::@p_size : i32
      %62 = db.constant(1 : i32) : i32
      %63 = db.constant(10 : i32) : i32
      %64 = db.between %61 : i32 between %62 : i32, %63 : i32, lowerInclusive : true, upperInclusive : true
      %65 = db.constant("AIR") : !db.string
      %66 = db.constant("AIR REG") : !db.string
      %67 = relalg.getcol %arg0 @lineitem::@l_shipmode : !db.string
      %68 = db.oneof %67 : !db.string ? %65, %66 : !db.string, !db.string
      %69 = relalg.getcol %arg0 @lineitem::@l_shipinstruct : !db.string
      %70 = db.constant("DELIVER IN PERSON") : !db.string
      %71 = db.compare eq %69 : !db.string, %70 : !db.string
      %72 = db.and %42, %45, %51, %54, %60, %64, %68, %71 : i1, i1, i1, i1, i1, i1, i1, i1
      %73 = relalg.getcol %arg0 @part::@p_partkey : i32
      %74 = relalg.getcol %arg0 @lineitem::@l_partkey : i32
      %75 = db.compare eq %73 : i32, %74 : i32
      %76 = relalg.getcol %arg0 @part::@p_brand : !db.string
      %77 = db.constant("Brand#34") : !db.string
      %78 = db.compare eq %76 : !db.string, %77 : !db.string
      %79 = db.constant("LG CASE") : !db.string
      %80 = db.constant("LG BOX") : !db.string
      %81 = db.constant("LG PACK") : !db.string
      %82 = db.constant("LG PKG") : !db.string
      %83 = relalg.getcol %arg0 @part::@p_container : !db.string
      %84 = db.oneof %83 : !db.string ? %79, %80, %81, %82 : !db.string, !db.string, !db.string, !db.string
      %85 = relalg.getcol %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %86 = db.constant(20 : i32) : !db.decimal<15, 2>
      %87 = db.compare gte %85 : !db.decimal<15, 2>, %86 : !db.decimal<15, 2>
      %88 = relalg.getcol %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %89 = db.constant(20 : i32) : i32
      %90 = db.constant(10 : i32) : i32
      %91 = db.add %89 : i32, %90 : i32
      %92 = db.cast %91 : i32 -> !db.decimal<15, 2>
      %93 = db.compare lte %88 : !db.decimal<15, 2>, %92 : !db.decimal<15, 2>
      %94 = relalg.getcol %arg0 @part::@p_size : i32
      %95 = db.constant(1 : i32) : i32
      %96 = db.constant(15 : i32) : i32
      %97 = db.between %94 : i32 between %95 : i32, %96 : i32, lowerInclusive : true, upperInclusive : true
      %98 = db.constant("AIR") : !db.string
      %99 = db.constant("AIR REG") : !db.string
      %100 = relalg.getcol %arg0 @lineitem::@l_shipmode : !db.string
      %101 = db.oneof %100 : !db.string ? %98, %99 : !db.string, !db.string
      %102 = relalg.getcol %arg0 @lineitem::@l_shipinstruct : !db.string
      %103 = db.constant("DELIVER IN PERSON") : !db.string
      %104 = db.compare eq %102 : !db.string, %103 : !db.string
      %105 = db.and %75, %78, %84, %87, %93, %97, %101, %104 : i1, i1, i1, i1, i1, i1, i1, i1
      %106 = db.or %39, %72, %105 : i1, i1, i1
      relalg.return %106 : i1
    }
    %4 = relalg.map @map0 %3 computes : [@tmp_attr1({type = !db.decimal<15, 2>})] (%arg0: !relalg.tuple){
      %7 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %8 = db.constant(1 : i32) : !db.decimal<15, 2>
      %9 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %10 = db.sub %8 : !db.decimal<15, 2>, %9 : !db.decimal<15, 2>
      %11 = db.mul %7 : !db.decimal<15, 2>, %10 : !db.decimal<15, 2>
      relalg.return %11 : !db.decimal<15, 2>
    }
    %5 = relalg.aggregation @aggr0 %4 [] computes : [@tmp_attr0({type = !db.nullable<!db.decimal<15, 2>>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %7 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.nullable<!db.decimal<15, 2>>
      relalg.return %7 : !db.nullable<!db.decimal<15, 2>>
    }
    %6 = relalg.materialize %5 [@aggr0::@tmp_attr0] => ["revenue"] : !dsa.table
    return %6 : !dsa.table
  }
}
