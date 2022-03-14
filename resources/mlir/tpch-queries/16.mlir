module {
  func @main() -> !db.table {
    %0 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = i32}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = i32}), ps_suppkey => @ps_suppkey({type = i32}), ps_supplycost => @ps_supplycost({type = !db.decimal<15, 2>})}
    %1 = relalg.basetable @part  {table_identifier = "part"} columns: {p_brand => @p_brand({type = !db.string}), p_comment => @p_comment({type = !db.string}), p_container => @p_container({type = !db.string}), p_mfgr => @p_mfgr({type = !db.string}), p_name => @p_name({type = !db.string}), p_partkey => @p_partkey({type = i32}), p_retailprice => @p_retailprice({type = !db.decimal<15, 2>}), p_size => @p_size({type = i32}), p_type => @p_type({type = !db.string})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.selection %2 (%arg0: !relalg.tuple){
      %7 = relalg.getattr %arg0 @part::@p_partkey : i32
      %8 = relalg.getattr %arg0 @partsupp::@ps_partkey : i32
      %9 = db.compare eq %7 : i32, %8 : i32
      %10 = relalg.getattr %arg0 @part::@p_brand : !db.string
      %11 = db.constant("Brand#45") : !db.string
      %12 = db.compare neq %10 : !db.string, %11 : !db.string
      %13 = relalg.getattr %arg0 @part::@p_type : !db.string
      %14 = db.constant("MEDIUM POLISHED%") : !db.string
      %15 = db.compare like %13 : !db.string, %14 : !db.string
      %16 = db.not %15 : i1
      %17 = db.constant(49 : i32) : i32
      %18 = db.constant(14 : i32) : i32
      %19 = db.constant(23 : i32) : i32
      %20 = db.constant(45 : i32) : i32
      %21 = db.constant(19 : i32) : i32
      %22 = db.constant(3 : i32) : i32
      %23 = db.constant(36 : i32) : i32
      %24 = db.constant(9 : i32) : i32
      %25 = relalg.getattr %arg0 @part::@p_size : i32
      %26 = db.oneof %25 : i32 ? %17, %18, %19, %20, %21, %22, %23, %24 : i32, i32, i32, i32, i32, i32, i32, i32
      %27 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
      %28 = relalg.selection %27 (%arg1: !relalg.tuple){
        %34 = relalg.getattr %arg1 @supplier::@s_comment : !db.string
        %35 = db.constant("%Customer%Complaints%") : !db.string
        %36 = db.compare like %34 : !db.string, %35 : !db.string
        relalg.return %36 : i1
      }
      %29 = relalg.projection all [@supplier::@s_suppkey] %28
      %30 = relalg.getattr %arg0 @partsupp::@ps_suppkey : i32
      %31 = relalg.in %30 : i32, %29
      %32 = db.not %31 : i1
      %33 = db.and %9, %12, %16, %26, %32 : i1, i1, i1, i1, i1
      relalg.return %33 : i1
    }
    %4 = relalg.aggregation @aggr0 %3 [@part::@p_brand,@part::@p_type,@part::@p_size] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %7 = relalg.projection distinct [@partsupp::@ps_suppkey] %arg0
      %8 = relalg.aggrfn count @partsupp::@ps_suppkey %7 : i64
      %9 = relalg.addattr %arg1, @tmp_attr0({type = i64}) %8
      relalg.return %9 : !relalg.tuple
    }
    %5 = relalg.sort %4 [(@aggr0::@tmp_attr0,desc),(@part::@p_brand,asc),(@part::@p_type,asc),(@part::@p_size,asc)]
    %6 = relalg.materialize %5 [@part::@p_brand,@part::@p_type,@part::@p_size,@aggr0::@tmp_attr0] => ["p_brand", "p_type", "p_size", "supplier_cnt"] : !db.table
    return %6 : !db.table
  }
}
