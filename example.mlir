module @querymodule  {
  func @query() -> !db.matcollection<!db.decimal<15,2>,!db.string,!db.string,!db.int<32>,!db.string,!db.string,!db.string,!db.string> {
    %0 = relalg.basetable @part  {table_identifier = "part"} columns: {p_brand => @p_brand({type = !db.string}), p_comment => @p_comment({type = !db.string}), p_container => @p_container({type = !db.string}), p_mfgr => @p_mfgr({type = !db.string}), p_name => @p_name({type = !db.string}), p_partkey => @p_partkey({type = !db.int<32>}), p_retailprice => @p_retailprice({type = !db.decimal<15,2>}), p_size => @p_size({type = !db.int<32>}), p_type => @p_type({type = !db.string})}
    %1 = relalg.selection %0 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @part::@p_type : !db.string
      %38 = db.constant( "%BRASS" ) : !db.string
      %39 = db.compare like %37 : !db.string, %38 : !db.string
      relalg.return %39 : !db.bool
    }
    %2 = relalg.selection %1 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @part::@p_size : !db.int<32>
      %38 = db.constant( "15" ) : !db.int<32>
      %39 = db.compare eq %37 : !db.int<32>, %38 : !db.int<32>
      relalg.return %39 : !db.bool
    }
    %3 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15,2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = !db.int<32>}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = !db.int<32>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = !db.int<32>}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = !db.int<32>}), ps_suppkey => @ps_suppkey({type = !db.int<32>}), ps_supplycost => @ps_supplycost({type = !db.decimal<15,2>})}
    %6 = relalg.crossproduct %4, %5
    %7 = relalg.selection %6 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @supplier::@s_suppkey : !db.int<32>
      %38 = relalg.getattr %arg0 @partsupp::@ps_suppkey : !db.int<32>
      %39 = db.compare eq %37 : !db.int<32>, %38 : !db.int<32>
      relalg.return %39 : !db.bool
    }
    %8 = relalg.selection %7 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @part::@p_partkey : !db.int<32>
      %38 = relalg.getattr %arg0 @partsupp::@ps_partkey : !db.int<32>
      %39 = db.compare eq %37 : !db.int<32>, %38 : !db.int<32>
      relalg.return %39 : !db.bool
    }
    %9 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.string<nullable>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = !db.int<32>}), n_regionkey => @n_regionkey({type = !db.int<32>})}
    %10 = relalg.crossproduct %8, %9
    %11 = relalg.selection %10 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @supplier::@s_nationkey : !db.int<32>
      %38 = relalg.getattr %arg0 @nation::@n_nationkey : !db.int<32>
      %39 = db.compare eq %37 : !db.int<32>, %38 : !db.int<32>
      relalg.return %39 : !db.bool
    }
    %12 = relalg.basetable @region  {table_identifier = "region"} columns: {r_comment => @r_comment({type = !db.string<nullable>}), r_name => @r_name({type = !db.string}), r_regionkey => @r_regionkey({type = !db.int<32>})}
    %13 = relalg.selection %12 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @region::@r_name : !db.string
      %38 = db.constant( "EUROPE" ) : !db.string
      %39 = db.compare eq %37 : !db.string, %38 : !db.string
      relalg.return %39 : !db.bool
    }
    %14 = relalg.crossproduct %11, %13
    %15 = relalg.selection %14 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @nation::@n_regionkey : !db.int<32>
      %38 = relalg.getattr %arg0 @region::@r_regionkey : !db.int<32>
      %39 = db.compare eq %37 : !db.int<32>, %38 : !db.int<32>
      relalg.return %39 : !db.bool
    }
    %16 = relalg.projection distinct [@part::@p_partkey] %15
    %17 = relalg.basetable @partsupp1  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = !db.int<32>}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = !db.int<32>}), ps_suppkey => @ps_suppkey({type = !db.int<32>}), ps_supplycost => @ps_supplycost({type = !db.decimal<15,2>})}
    %18 = relalg.crossproduct %17, %16
    %19 = relalg.basetable @supplier1  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15,2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = !db.int<32>}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = !db.int<32>})}
    %20 = relalg.crossproduct %18, %19
    %21 = relalg.basetable @nation1  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.string<nullable>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = !db.int<32>}), n_regionkey => @n_regionkey({type = !db.int<32>})}
    %22 = relalg.selection %20 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @supplier1::@s_suppkey : !db.int<32>
      %38 = relalg.getattr %arg0 @partsupp1::@ps_suppkey : !db.int<32>
      %39 = db.compare eq %37 : !db.int<32>, %38 : !db.int<32>
      relalg.return %39 : !db.bool
    }
    %23 = relalg.crossproduct %22, %21
    %24 = relalg.basetable @region1  {table_identifier = "region"} columns: {r_comment => @r_comment({type = !db.string<nullable>}), r_name => @r_name({type = !db.string}), r_regionkey => @r_regionkey({type = !db.int<32>})}
    %25 = relalg.selection %23 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @supplier1::@s_nationkey : !db.int<32>
      %38 = relalg.getattr %arg0 @nation1::@n_nationkey : !db.int<32>
      %39 = db.compare eq %37 : !db.int<32>, %38 : !db.int<32>
      relalg.return %39 : !db.bool
    }
    %26 = relalg.selection %24 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @region1::@r_name : !db.string
      %38 = db.constant( "EUROPE" ) : !db.string
      %39 = db.compare eq %37 : !db.string, %38 : !db.string
      relalg.return %39 : !db.bool
    }
    %27 = relalg.crossproduct %25, %26
    %28 = relalg.selection %27 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @nation1::@n_regionkey : !db.int<32>
      %38 = relalg.getattr %arg0 @region1::@r_regionkey : !db.int<32>
      %39 = db.compare eq %37 : !db.int<32>, %38 : !db.int<32>
      relalg.return %39 : !db.bool
    }
    %29 = relalg.selection %28 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @part::@p_partkey : !db.int<32>
      %38 = relalg.getattr %arg0 @partsupp1::@ps_partkey : !db.int<32>
      %39 = db.compare eq %37 : !db.int<32>, %38 : !db.int<32>
      relalg.return %39 : !db.bool
    }
    %30 = relalg.aggregation @aggr1 %29 [@part::@p_partkey] (%arg0: !relalg.relation) {
      %37 = relalg.aggrfn min @partsupp1::@ps_supplycost %arg0 : !db.decimal<15,2,nullable>
      relalg.addattr @aggfmname1({type = !db.decimal<15,2,nullable>}) %37
      relalg.return
    }
    %31 = relalg.renaming @renaming %30  attributes: [@renamed0({type = !db.int<32>})=[@part::@p_partkey]]
    %32 = relalg.singlejoin left %15, %31(%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @part::@p_partkey : !db.int<32>
      %38 = relalg.getattr %arg0 @renaming::@renamed0 : !db.int<32>
      %39 = db.compare eq %37 : !db.int<32>, %38 : !db.int<32>
      relalg.return %39 : !db.bool
    }
    %33 = relalg.selection %32 (%arg0: !relalg.tuple) {
      %37 = relalg.getattr %arg0 @partsupp::@ps_supplycost : !db.decimal<15,2>
      %38 = relalg.getattr %arg0 @aggr1::@aggfmname1 : !db.decimal<15,2,nullable>
      %39 = db.compare eq %37 : !db.decimal<15,2>, %38 : !db.decimal<15,2,nullable>
      relalg.return %39 : !db.bool<nullable>
    }
    %34 = relalg.sort %33 [(@supplier::@s_acctbal,desc),(@nation::@n_name,asc),(@supplier::@s_name,asc),(@part::@p_partkey,asc)]
    %35 = relalg.limit 100 %34
    %36 = relalg.materialize %35 [@supplier::@s_acctbal,@supplier::@s_name,@nation::@n_name,@part::@p_partkey,@part::@p_mfgr,@supplier::@s_address,@supplier::@s_phone,@supplier::@s_comment] : !db.matcollection<!db.decimal<15,2>,!db.string,!db.string,!db.int<32>,!db.string,!db.string,!db.string,!db.string>
    return %36 : !db.matcollection<!db.decimal<15,2>,!db.string,!db.string,!db.int<32>,!db.string,!db.string,!db.string,!db.string>
  }
}

