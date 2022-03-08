//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                       revenue  |
//CHECK: ----------------------------------
//CHECK: |                     168597.24  |
module {
  func @main() -> !db.table {
    %0 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %1 = relalg.basetable @part  {table_identifier = "part"} columns: {p_brand => @p_brand({type = !db.string}), p_comment => @p_comment({type = !db.string}), p_container => @p_container({type = !db.string}), p_mfgr => @p_mfgr({type = !db.string}), p_name => @p_name({type = !db.string}), p_partkey => @p_partkey({type = i32}), p_retailprice => @p_retailprice({type = !db.decimal<15, 2>}), p_size => @p_size({type = i32}), p_type => @p_type({type = !db.string})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.selection %2 (%arg0: !relalg.tuple){
      %7 = relalg.getattr %arg0 @part::@p_partkey : i32
      %8 = relalg.getattr %arg0 @lineitem::@l_partkey : i32
      %9 = db.compare eq %7 : i32, %8 : i32
      %10 = relalg.getattr %arg0 @part::@p_brand : !db.string
      %11 = db.constant("Brand#12") : !db.string
      %12 = db.compare eq %10 : !db.string, %11 : !db.string
      %13 = db.constant("SM CASE") : !db.string
      %14 = db.constant("SM BOX") : !db.string
      %15 = db.constant("SM PACK") : !db.string
      %16 = db.constant("SM PKG") : !db.string
      %17 = relalg.getattr %arg0 @part::@p_container : !db.string
      %18 = db.oneof %17 : !db.string ? %13, %14, %15, %16 : !db.string, !db.string, !db.string, !db.string
      %19 = relalg.getattr %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %20 = db.constant(1 : i32) : !db.decimal<15, 2>
      %21 = db.compare gte %19 : !db.decimal<15, 2>, %20 : !db.decimal<15, 2>
      %22 = relalg.getattr %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %23 = db.constant(1 : i32) : i32
      %24 = db.constant(10 : i32) : i32
      %25 = db.add %23 : i32, %24 : i32
      %26 = db.cast %25 : i32 -> !db.decimal<15, 2>
      %27 = db.compare lte %22 : !db.decimal<15, 2>, %26 : !db.decimal<15, 2>
      %28 = relalg.getattr %arg0 @part::@p_size : i32
      %29 = db.constant(1 : i32) : i32
      %30 = db.constant(5 : i32) : i32
      %31 = db.between %28 : i32 between %29 : i32, %30 : i32, lowerInclusive : true, upperInclusive : true
      %32 = db.constant("AIR") : !db.string
      %33 = db.constant("AIR REG") : !db.string
      %34 = relalg.getattr %arg0 @lineitem::@l_shipmode : !db.string
      %35 = db.oneof %34 : !db.string ? %32, %33 : !db.string, !db.string
      %36 = relalg.getattr %arg0 @lineitem::@l_shipinstruct : !db.string
      %37 = db.constant("DELIVER IN PERSON") : !db.string
      %38 = db.compare eq %36 : !db.string, %37 : !db.string
      %39 = db.and %9:i1,%12:i1,%18:i1,%21:i1,%27:i1,%31:i1,%35:i1,%38:i1
      %40 = relalg.getattr %arg0 @part::@p_partkey : i32
      %41 = relalg.getattr %arg0 @lineitem::@l_partkey : i32
      %42 = db.compare eq %40 : i32, %41 : i32
      %43 = relalg.getattr %arg0 @part::@p_brand : !db.string
      %44 = db.constant("Brand#23") : !db.string
      %45 = db.compare eq %43 : !db.string, %44 : !db.string
      %46 = db.constant("MED BAG") : !db.string
      %47 = db.constant("MED BOX") : !db.string
      %48 = db.constant("MED PKG") : !db.string
      %49 = db.constant("MED PACK") : !db.string
      %50 = relalg.getattr %arg0 @part::@p_container : !db.string
      %51 = db.oneof %50 : !db.string ? %46, %47, %48, %49 : !db.string, !db.string, !db.string, !db.string
      %52 = relalg.getattr %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %53 = db.constant(10 : i32) : !db.decimal<15, 2>
      %54 = db.compare gte %52 : !db.decimal<15, 2>, %53 : !db.decimal<15, 2>
      %55 = relalg.getattr %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %56 = db.constant(10 : i32) : i32
      %57 = db.constant(10 : i32) : i32
      %58 = db.add %56 : i32, %57 : i32
      %59 = db.cast %58 : i32 -> !db.decimal<15, 2>
      %60 = db.compare lte %55 : !db.decimal<15, 2>, %59 : !db.decimal<15, 2>
      %61 = relalg.getattr %arg0 @part::@p_size : i32
      %62 = db.constant(1 : i32) : i32
      %63 = db.constant(10 : i32) : i32
      %64 = db.between %61 : i32 between %62 : i32, %63 : i32, lowerInclusive : true, upperInclusive : true
      %65 = db.constant("AIR") : !db.string
      %66 = db.constant("AIR REG") : !db.string
      %67 = relalg.getattr %arg0 @lineitem::@l_shipmode : !db.string
      %68 = db.oneof %67 : !db.string ? %65, %66 : !db.string, !db.string
      %69 = relalg.getattr %arg0 @lineitem::@l_shipinstruct : !db.string
      %70 = db.constant("DELIVER IN PERSON") : !db.string
      %71 = db.compare eq %69 : !db.string, %70 : !db.string
      %72 = db.and %42:i1,%45:i1,%51:i1,%54:i1,%60:i1,%64:i1,%68:i1,%71:i1
      %73 = relalg.getattr %arg0 @part::@p_partkey : i32
      %74 = relalg.getattr %arg0 @lineitem::@l_partkey : i32
      %75 = db.compare eq %73 : i32, %74 : i32
      %76 = relalg.getattr %arg0 @part::@p_brand : !db.string
      %77 = db.constant("Brand#34") : !db.string
      %78 = db.compare eq %76 : !db.string, %77 : !db.string
      %79 = db.constant("LG CASE") : !db.string
      %80 = db.constant("LG BOX") : !db.string
      %81 = db.constant("LG PACK") : !db.string
      %82 = db.constant("LG PKG") : !db.string
      %83 = relalg.getattr %arg0 @part::@p_container : !db.string
      %84 = db.oneof %83 : !db.string ? %79, %80, %81, %82 : !db.string, !db.string, !db.string, !db.string
      %85 = relalg.getattr %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %86 = db.constant(20 : i32) : !db.decimal<15, 2>
      %87 = db.compare gte %85 : !db.decimal<15, 2>, %86 : !db.decimal<15, 2>
      %88 = relalg.getattr %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %89 = db.constant(20 : i32) : i32
      %90 = db.constant(10 : i32) : i32
      %91 = db.add %89 : i32, %90 : i32
      %92 = db.cast %91 : i32 -> !db.decimal<15, 2>
      %93 = db.compare lte %88 : !db.decimal<15, 2>, %92 : !db.decimal<15, 2>
      %94 = relalg.getattr %arg0 @part::@p_size : i32
      %95 = db.constant(1 : i32) : i32
      %96 = db.constant(15 : i32) : i32
      %97 = db.between %94 : i32 between %95 : i32, %96 : i32, lowerInclusive : true, upperInclusive : true
      %98 = db.constant("AIR") : !db.string
      %99 = db.constant("AIR REG") : !db.string
      %100 = relalg.getattr %arg0 @lineitem::@l_shipmode : !db.string
      %101 = db.oneof %100 : !db.string ? %98, %99 : !db.string, !db.string
      %102 = relalg.getattr %arg0 @lineitem::@l_shipinstruct : !db.string
      %103 = db.constant("DELIVER IN PERSON") : !db.string
      %104 = db.compare eq %102 : !db.string, %103 : !db.string
      %105 = db.and %75:i1,%78:i1,%84:i1,%87:i1,%93:i1,%97:i1,%101:i1,%104:i1
      %106 = db.or %39:i1,%72:i1,%105:i1
      relalg.return %106 : i1
    }
    %4 = relalg.map @map0 %3 (%arg0: !relalg.tuple){
      %7 = relalg.getattr %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %8 = db.constant(1 : i32) : !db.decimal<15, 2>
      %9 = relalg.getattr %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %10 = db.sub %8 : !db.decimal<15, 2>, %9 : !db.decimal<15, 2>
      %11 = db.mul %7 : !db.decimal<15, 2>, %10 : !db.decimal<15, 2>
      %12 = relalg.addattr %arg0, @tmp_attr1({type = !db.decimal<15, 2>}) %11
      relalg.return %12 : !relalg.tuple
    }
    %5 = relalg.aggregation @aggr0 %4 [] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %7 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.nullable<!db.decimal<15, 2>>
      %8 = relalg.addattr %arg1, @tmp_attr0({type = !db.nullable<!db.decimal<15, 2>>}) %7
      relalg.return %8 : !relalg.tuple
    }
    %6 = relalg.materialize %5 [@aggr0::@tmp_attr0] => ["revenue"] : !db.table
    return %6 : !db.table
  }
}

