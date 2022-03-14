module {
  func @main() -> !db.table {
    %0 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
    %1 = relalg.selection %0 (%arg0: !relalg.tuple){
      %6 = db.constant("13") : !db.string
      %7 = db.constant("31") : !db.string
      %8 = db.constant("23") : !db.string
      %9 = db.constant("29") : !db.string
      %10 = db.constant("30") : !db.string
      %11 = db.constant("18") : !db.string
      %12 = db.constant("17") : !db.string
      %13 = relalg.getattr %arg0 @customer::@c_phone : !db.string
      %14 = db.substr %13[1 : 2] : !db.string
      %15 = db.oneof %14 : !db.string ? %6, %7, %8, %9, %10, %11, %12 : !db.string, !db.string, !db.string, !db.string, !db.string, !db.string, !db.string
      %16 = relalg.getattr %arg0 @customer::@c_acctbal : !db.decimal<15, 2>
      %17 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
      %18 = relalg.selection %17 (%arg1: !relalg.tuple){
        %27 = relalg.getattr %arg1 @customer::@c_acctbal : !db.decimal<15, 2>
        %28 = db.constant("0.00") : !db.decimal<15, 2>
        %29 = db.compare gt %27 : !db.decimal<15, 2>, %28 : !db.decimal<15, 2>
        %30 = db.constant("13") : !db.string
        %31 = db.constant("31") : !db.string
        %32 = db.constant("23") : !db.string
        %33 = db.constant("29") : !db.string
        %34 = db.constant("30") : !db.string
        %35 = db.constant("18") : !db.string
        %36 = db.constant("17") : !db.string
        %37 = relalg.getattr %arg1 @customer::@c_phone : !db.string
        %38 = db.substr %37[1 : 2] : !db.string
        %39 = db.oneof %38 : !db.string ? %30, %31, %32, %33, %34, %35, %36 : !db.string, !db.string, !db.string, !db.string, !db.string, !db.string, !db.string
        %40 = db.and %29, %39 : i1, i1
        relalg.return %40 : i1
      }
      %19 = relalg.aggregation @aggr0 %18 [] (%arg1: !relalg.tuplestream,%arg2: !relalg.tuple){
        %27 = relalg.aggrfn avg @customer::@c_acctbal %arg1 : !db.nullable<!db.decimal<15, 2>>
        %28 = relalg.addattr %arg2, @tmp_attr0({type = !db.nullable<!db.decimal<15, 2>>}) %27
        relalg.return %28 : !relalg.tuple
      }
      %20 = relalg.getscalar @aggr0::@tmp_attr0 %19 : !db.nullable<!db.decimal<15, 2>>
      %21 = db.compare gt %16 : !db.decimal<15, 2>, %20 : !db.nullable<!db.decimal<15, 2>>
      %22 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
      %23 = relalg.selection %22 (%arg1: !relalg.tuple){
        %27 = relalg.getattr %arg1 @orders::@o_custkey : i32
        %28 = relalg.getattr %arg1 @customer::@c_custkey : i32
        %29 = db.compare eq %27 : i32, %28 : i32
        relalg.return %29 : i1
      }
      %24 = relalg.exists %23
      %25 = db.not %24 : i1
      %26 = db.and %15, %21, %25 : i1, !db.nullable<i1>, i1
      relalg.return %26 : !db.nullable<i1>
    }
    %2 = relalg.map @map0 %1 (%arg0: !relalg.tuple){
      %6 = relalg.getattr %arg0 @customer::@c_phone : !db.string
      %7 = db.substr %6[1 : 2] : !db.string
      %8 = relalg.addattr %arg0, @tmp_attr1({type = !db.string}) %7
      relalg.return %8 : !relalg.tuple
    }
    %3 = relalg.aggregation @aggr1 %2 [@map0::@tmp_attr1] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %6 = relalg.aggrfn sum @customer::@c_acctbal %arg0 : !db.decimal<15, 2>
      %7 = relalg.addattr %arg1, @tmp_attr3({type = !db.decimal<15, 2>}) %6
      %8 = relalg.count %arg0
      %9 = relalg.addattr %7, @tmp_attr2({type = i64}) %8
      relalg.return %9 : !relalg.tuple
    }
    %4 = relalg.sort %3 [(@map0::@tmp_attr1,asc)]
    %5 = relalg.materialize %4 [@map0::@tmp_attr1,@aggr1::@tmp_attr2,@aggr1::@tmp_attr3] => ["cntrycode", "numcust", "totacctbal"] : !db.table
    return %5 : !db.table
  }
}
