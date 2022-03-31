module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = i32}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = i32}), ps_suppkey => @ps_suppkey({type = i32}), ps_supplycost => @ps_supplycost({type = !db.decimal<15, 2>})}
    %1 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.selection %4 (%arg0: !relalg.tuple){
      %11 = relalg.getcol %arg0 @partsupp::@ps_suppkey : i32
      %12 = relalg.getcol %arg0 @supplier::@s_suppkey : i32
      %13 = db.compare eq %11 : i32, %12 : i32
      %14 = relalg.getcol %arg0 @supplier::@s_nationkey : i32
      %15 = relalg.getcol %arg0 @nation::@n_nationkey : i32
      %16 = db.compare eq %14 : i32, %15 : i32
      %17 = relalg.getcol %arg0 @nation::@n_name : !db.string
      %18 = db.constant("GERMANY") : !db.string
      %19 = db.compare eq %17 : !db.string, %18 : !db.string
      %20 = db.and %13, %16, %19 : i1, i1, i1
      relalg.return %20 : i1
    }
    %6 = relalg.map @map0 %5 (%arg0: !relalg.tuple){
      %11 = relalg.getcol %arg0 @partsupp::@ps_supplycost : !db.decimal<15, 2>
      %12 = relalg.getcol %arg0 @partsupp::@ps_availqty : i32
      %13 = db.cast %12 : i32 -> !db.decimal<15, 2>
      %14 = db.mul %11 : !db.decimal<15, 2>, %13 : !db.decimal<15, 2>
      %15 = relalg.addcol %arg0, @tmp_attr3({type = !db.decimal<15, 2>}) %14
      %16 = relalg.getcol %15 @partsupp::@ps_supplycost : !db.decimal<15, 2>
      %17 = relalg.getcol %15 @partsupp::@ps_availqty : i32
      %18 = db.cast %17 : i32 -> !db.decimal<15, 2>
      %19 = db.mul %16 : !db.decimal<15, 2>, %18 : !db.decimal<15, 2>
      %20 = relalg.addcol %15, @tmp_attr1({type = !db.decimal<15, 2>}) %19
      relalg.return %20 : !relalg.tuple
    }
    %7 = relalg.aggregation @aggr0 %6 [@partsupp::@ps_partkey] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %11 = relalg.aggrfn sum @map0::@tmp_attr3 %arg0 : !db.decimal<15, 2>
      %12 = relalg.addcol %arg1, @tmp_attr2({type = !db.decimal<15, 2>}) %11
      %13 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.decimal<15, 2>
      %14 = relalg.addcol %12, @tmp_attr0({type = !db.decimal<15, 2>}) %13
      relalg.return %14 : !relalg.tuple
    }
    %8 = relalg.selection %7 (%arg0: !relalg.tuple){
      %11 = relalg.getcol %arg0 @aggr0::@tmp_attr2 : !db.decimal<15, 2>
      %12 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = i32}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = i32}), ps_suppkey => @ps_suppkey({type = i32}), ps_supplycost => @ps_supplycost({type = !db.decimal<15, 2>})}
      %13 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
      %14 = relalg.crossproduct %12, %13
      %15 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
      %16 = relalg.crossproduct %14, %15
      %17 = relalg.selection %16 (%arg1: !relalg.tuple){
        %24 = relalg.getcol %arg1 @partsupp::@ps_suppkey : i32
        %25 = relalg.getcol %arg1 @supplier::@s_suppkey : i32
        %26 = db.compare eq %24 : i32, %25 : i32
        %27 = relalg.getcol %arg1 @supplier::@s_nationkey : i32
        %28 = relalg.getcol %arg1 @nation::@n_nationkey : i32
        %29 = db.compare eq %27 : i32, %28 : i32
        %30 = relalg.getcol %arg1 @nation::@n_name : !db.string
        %31 = db.constant("GERMANY") : !db.string
        %32 = db.compare eq %30 : !db.string, %31 : !db.string
        %33 = db.and %26, %29, %32 : i1, i1, i1
        relalg.return %33 : i1
      }
      %18 = relalg.map @map1 %17 (%arg1: !relalg.tuple){
        %24 = relalg.getcol %arg1 @partsupp::@ps_supplycost : !db.decimal<15, 2>
        %25 = relalg.getcol %arg1 @partsupp::@ps_availqty : i32
        %26 = db.cast %25 : i32 -> !db.decimal<15, 2>
        %27 = db.mul %24 : !db.decimal<15, 2>, %26 : !db.decimal<15, 2>
        %28 = relalg.addcol %arg1, @tmp_attr5({type = !db.decimal<15, 2>}) %27
        relalg.return %28 : !relalg.tuple
      }
      %19 = relalg.aggregation @aggr1 %18 [] (%arg1: !relalg.tuplestream,%arg2: !relalg.tuple){
        %24 = relalg.aggrfn sum @map1::@tmp_attr5 %arg1 : !db.nullable<!db.decimal<15, 2>>
        %25 = relalg.addcol %arg2, @tmp_attr4({type = !db.nullable<!db.decimal<15, 2>>}) %24
        relalg.return %25 : !relalg.tuple
      }
      %20 = relalg.map @map2 %19 (%arg1: !relalg.tuple){
        %24 = relalg.getcol %arg1 @aggr1::@tmp_attr4 : !db.nullable<!db.decimal<15, 2>>
        %25 = db.constant("0.0001") : !db.decimal<15, 4>
        %26 = db.cast %24 : !db.nullable<!db.decimal<15, 2>> -> !db.nullable<!db.decimal<15, 4>>
        %27 = db.mul %26 : !db.nullable<!db.decimal<15, 4>>, %25 : !db.decimal<15, 4>
        %28 = relalg.addcol %arg1, @tmp_attr6({type = !db.nullable<!db.decimal<15, 4>>}) %27
        relalg.return %28 : !relalg.tuple
      }
      %21 = relalg.getscalar @map2::@tmp_attr6 %20 : !db.nullable<!db.decimal<15, 4>>
      %22 = db.cast %11 : !db.decimal<15, 2> -> !db.decimal<15, 4>
      %23 = db.compare gt %22 : !db.decimal<15, 4>, %21 : !db.nullable<!db.decimal<15, 4>>
      relalg.return %23 : !db.nullable<i1>
    }
    %9 = relalg.sort %8 [(@aggr0::@tmp_attr0,desc)]
    %10 = relalg.materialize %9 [@partsupp::@ps_partkey,@aggr0::@tmp_attr0] => ["ps_partkey", "value"] : !dsa.table
    return %10 : !dsa.table
  }
}
