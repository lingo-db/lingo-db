//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                     cntrycode  |                       numcust  |                    totacctbal  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                          "13"  |                            94  |                     714035.05  |
//CHECK: |                          "17"  |                            96  |                     722560.15  |
//CHECK: |                          "18"  |                            99  |                     738012.52  |
//CHECK: |                          "23"  |                            93  |                     708285.25  |
//CHECK: |                          "29"  |                            85  |                     632693.46  |
//CHECK: |                          "30"  |                            87  |                     646748.02  |
//CHECK: |                          "31"  |                            87  |                     647372.50  |
module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
    %1 = relalg.selection %0 (%arg0: !relalg.tuple){
      %6 = db.constant("13") : !db.string
      %7 = db.constant("31") : !db.string
      %8 = db.constant("23") : !db.string
      %9 = db.constant("29") : !db.string
      %10 = db.constant("30") : !db.string
      %11 = db.constant("18") : !db.string
      %12 = db.constant("17") : !db.string
      %13 = relalg.getcol %arg0 @customer::@c_phone : !db.string
      %14 = db.constant(1 : i32) : i32
      %15 = db.constant(2 : i32) : i32
      %16 = db.runtime_call "Substring"(%13, %14, %15) : (!db.string, i32, i32) -> !db.string
      %17 = db.oneof %16 : !db.string ? %6, %7, %8, %9, %10, %11, %12 : !db.string, !db.string, !db.string, !db.string, !db.string, !db.string, !db.string
      %18 = relalg.getcol %arg0 @customer::@c_acctbal : !db.decimal<15, 2>
      %19 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
      %20 = relalg.selection %19 (%arg1: !relalg.tuple){
        %29 = relalg.getcol %arg1 @customer::@c_acctbal : !db.decimal<15, 2>
        %30 = db.constant("0.00") : !db.decimal<15, 2>
        %31 = db.compare gt %29 : !db.decimal<15, 2>, %30 : !db.decimal<15, 2>
        %32 = db.constant("13") : !db.string
        %33 = db.constant("31") : !db.string
        %34 = db.constant("23") : !db.string
        %35 = db.constant("29") : !db.string
        %36 = db.constant("30") : !db.string
        %37 = db.constant("18") : !db.string
        %38 = db.constant("17") : !db.string
        %39 = relalg.getcol %arg1 @customer::@c_phone : !db.string
        %40 = db.constant(1 : i32) : i32
        %41 = db.constant(2 : i32) : i32
        %42 = db.runtime_call "Substring"(%39, %40, %41) : (!db.string, i32, i32) -> !db.string
        %43 = db.oneof %42 : !db.string ? %32, %33, %34, %35, %36, %37, %38 : !db.string, !db.string, !db.string, !db.string, !db.string, !db.string, !db.string
        %44 = db.and %31, %43 : i1, i1
        relalg.return %44 : i1
      }
      %21 = relalg.aggregation @aggr0 %20 [] (%arg1: !relalg.tuplestream,%arg2: !relalg.tuple){
        %29 = relalg.aggrfn avg @customer::@c_acctbal %arg1 : !db.nullable<!db.decimal<15, 2>>
        %30 = relalg.addcol %arg2, @tmp_attr0({type = !db.nullable<!db.decimal<15, 2>>}) %29
        relalg.return %30 : !relalg.tuple
      }
      %22 = relalg.getscalar @aggr0::@tmp_attr0 %21 : !db.nullable<!db.decimal<15, 2>>
      %23 = db.compare gt %18 : !db.decimal<15, 2>, %22 : !db.nullable<!db.decimal<15, 2>>
      %24 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
      %25 = relalg.selection %24 (%arg1: !relalg.tuple){
        %29 = relalg.getcol %arg1 @orders::@o_custkey : i32
        %30 = relalg.getcol %arg1 @customer::@c_custkey : i32
        %31 = db.compare eq %29 : i32, %30 : i32
        relalg.return %31 : i1
      }
      %26 = relalg.exists %25
      %27 = db.not %26 : i1
      %28 = db.and %17, %23, %27 : i1, !db.nullable<i1>, i1
      relalg.return %28 : !db.nullable<i1>
    }
    %2 = relalg.map @map0 %1 (%arg0: !relalg.tuple){
      %6 = relalg.getcol %arg0 @customer::@c_phone : !db.string
      %7 = db.constant(1 : i32) : i32
      %8 = db.constant(2 : i32) : i32
      %9 = db.runtime_call "Substring"(%6, %7, %8) : (!db.string, i32, i32) -> !db.string
      %10 = relalg.addcol %arg0, @tmp_attr1({type = !db.string}) %9
      relalg.return %10 : !relalg.tuple
    }
    %3 = relalg.aggregation @aggr1 %2 [@map0::@tmp_attr1] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %6 = relalg.aggrfn sum @customer::@c_acctbal %arg0 : !db.decimal<15, 2>
      %7 = relalg.addcol %arg1, @tmp_attr3({type = !db.decimal<15, 2>}) %6
      %8 = relalg.count %arg0
      %9 = relalg.addcol %7, @tmp_attr2({type = i64}) %8
      relalg.return %9 : !relalg.tuple
    }
    %4 = relalg.sort %3 [(@map0::@tmp_attr1,asc)]
    %5 = relalg.materialize %4 [@map0::@tmp_attr1,@aggr1::@tmp_attr2,@aggr1::@tmp_attr3] => ["cntrycode", "numcust", "totacctbal"] : !dsa.table
    return %5 : !dsa.table
  }
}

