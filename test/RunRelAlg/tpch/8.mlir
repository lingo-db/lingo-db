//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                        o_year  |                     mkt_share  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                          1995  |                          0.02  |
//CHECK: |                          1996  |                          0.01  |
module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=!db.int<32>}),
            p_name => @p_name({type=!db.string}),
            p_mfgr => @p_mfgr({type=!db.string}),
            p_brand => @p_brand({type=!db.string}),
            p_type => @p_type({type=!db.string}),
            p_size => @p_size({type=!db.int<32>}),
            p_container => @p_container({type=!db.string}),
            p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
            p_comment => @p_comment({type=!db.string})
        }
        %2 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=!db.int<32>}),
            s_name => @s_name({type=!db.string}),
            s_address => @s_address({type=!db.string}),
            s_nationkey => @s_nationkey({type=!db.int<32>}),
            s_phone => @s_phone({type=!db.string}),
            s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
            s_comment => @s_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %4 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<32>}),
            l_partkey => @l_partkey({type=!db.int<32>}),
            l_suppkey => @l_suppkey({type=!db.int<32>}),
            l_linenumber => @l_linenumber({type=!db.int<32>}),
            l_quantity => @l_quantity({type=!db.decimal<15,2>}),
            l_extendedprice => @l_extendedprice({type=!db.decimal<15,2>}),
            l_discount => @l_discount({type=!db.decimal<15,2>}),
            l_tax => @l_tax({type=!db.decimal<15,2>}),
            l_returnflag => @l_returnflag({type=!db.char<1>}),
            l_linestatus => @l_linestatus({type=!db.char<1>}),
            l_shipdate => @l_shipdate({type=!db.date<day>}),
            l_commitdate => @l_commitdate({type=!db.date<day>}),
            l_receiptdate => @l_receiptdate({type=!db.date<day>}),
            l_shipinstruct => @l_shipinstruct({type=!db.string}),
            l_shipmode => @l_shipmode({type=!db.string}),
            l_comment => @l_comment({type=!db.string})
        }
        %5 = relalg.crossproduct %3, %4
        %6 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=!db.int<32>}),
            o_custkey => @o_custkey({type=!db.int<32>}),
            o_orderstatus => @o_orderstatus({type=!db.char<1>}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=!db.int<32>}),
            o_comment => @o_comment({type=!db.string})
        }
        %7 = relalg.crossproduct %5, %6
        %8 = relalg.basetable @customer { table_identifier="customer", rows=15000 , pkey=["c_custkey"]} columns: {c_custkey => @c_custkey({type=!db.int<32>}),
            c_name => @c_name({type=!db.string}),
            c_address => @c_address({type=!db.string}),
            c_nationkey => @c_nationkey({type=!db.int<32>}),
            c_phone => @c_phone({type=!db.string}),
            c_acctbal => @c_acctbal({type=!db.decimal<15,2>}),
            c_mktsegment => @c_mktsegment({type=!db.string}),
            c_comment => @c_comment({type=!db.string})
        }
        %9 = relalg.crossproduct %7, %8
        %10 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<32>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<32>}),
            n_comment => @n_comment({type=!db.nullable<!db.string>})
        }
        %11 = relalg.crossproduct %9, %10
        %12 = relalg.basetable @nation1 { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<32>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<32>}),
            n_comment => @n_comment({type=!db.nullable<!db.string>})
        }
        %13 = relalg.crossproduct %11, %12
        %14 = relalg.basetable @region { table_identifier="region", rows=5 , pkey=["r_regionkey"]} columns: {r_regionkey => @r_regionkey({type=!db.int<32>}),
            r_name => @r_name({type=!db.string}),
            r_comment => @r_comment({type=!db.nullable<!db.string>})
        }
        %15 = relalg.crossproduct %13, %14
        %17 = relalg.selection %15(%16: !relalg.tuple) {
            %18 = relalg.getattr %16 @part::@p_partkey : !db.int<32>
            %19 = relalg.getattr %16 @lineitem::@l_partkey : !db.int<32>
            %20 = db.compare eq %18 : !db.int<32>,%19 : !db.int<32>
            %21 = relalg.getattr %16 @supplier::@s_suppkey : !db.int<32>
            %22 = relalg.getattr %16 @lineitem::@l_suppkey : !db.int<32>
            %23 = db.compare eq %21 : !db.int<32>,%22 : !db.int<32>
            %24 = relalg.getattr %16 @lineitem::@l_orderkey : !db.int<32>
            %25 = relalg.getattr %16 @orders::@o_orderkey : !db.int<32>
            %26 = db.compare eq %24 : !db.int<32>,%25 : !db.int<32>
            %27 = relalg.getattr %16 @orders::@o_custkey : !db.int<32>
            %28 = relalg.getattr %16 @customer::@c_custkey : !db.int<32>
            %29 = db.compare eq %27 : !db.int<32>,%28 : !db.int<32>
            %30 = relalg.getattr %16 @customer::@c_nationkey : !db.int<32>
            %31 = relalg.getattr %16 @nation::@n_nationkey : !db.int<32>
            %32 = db.compare eq %30 : !db.int<32>,%31 : !db.int<32>
            %33 = relalg.getattr %16 @nation::@n_regionkey : !db.int<32>
            %34 = relalg.getattr %16 @region::@r_regionkey : !db.int<32>
            %35 = db.compare eq %33 : !db.int<32>,%34 : !db.int<32>
            %36 = relalg.getattr %16 @region::@r_name : !db.string
            %37 = db.constant ("AMERICA") :!db.string
            %38 = db.compare eq %36 : !db.string,%37 : !db.string
            %39 = relalg.getattr %16 @supplier::@s_nationkey : !db.int<32>
            %40 = relalg.getattr %16 @nation1::@n_nationkey : !db.int<32>
            %41 = db.compare eq %39 : !db.int<32>,%40 : !db.int<32>
            %42 = relalg.getattr %16 @orders::@o_orderdate : !db.date<day>
            %43 = db.constant ("1995-01-01") :!db.date<day>
            %44 = db.constant ("1996-12-31") :!db.date<day>
            %45 = db.compare gte %42 : !db.date<day>,%43 : !db.date<day>
            %46 = db.compare lte %42 : !db.date<day>,%44 : !db.date<day>
            %47 = db.and %45 : i1,%46 : i1
            %48 = relalg.getattr %16 @part::@p_type : !db.string
            %49 = db.constant ("ECONOMY ANODIZED STEEL") :!db.string
            %50 = db.compare eq %48 : !db.string,%49 : !db.string
            %51 = db.and %20 : i1,%23 : i1,%26 : i1,%29 : i1,%32 : i1,%35 : i1,%38 : i1,%41 : i1,%47 : i1,%50 : i1
            relalg.return %51 : i1
        }
        %53 = relalg.map @map1 %17 (%52: !relalg.tuple) {
            %54 = relalg.getattr %52 @orders::@o_orderdate : !db.date<day>
            %55 = db.date_extract year, %54 : !db.date<day>
            %56 = relalg.addattr %52, @aggfmname1({type=!db.int<64>}) %55
            %57 = relalg.getattr %52 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %58 = db.constant (1) :!db.decimal<15,2>
            %59 = relalg.getattr %52 @lineitem::@l_discount : !db.decimal<15,2>
            %60 = db.sub %58 : !db.decimal<15,2>,%59 : !db.decimal<15,2>
            %61 = db.mul %57 : !db.decimal<15,2>,%60 : !db.decimal<15,2>
            %62 = relalg.addattr %56, @aggfmname2({type=!db.decimal<15,2>}) %61
            relalg.return %62 : !relalg.tuple
        }
        %64 = relalg.map @map2 %53 (%63: !relalg.tuple) {
            %65 = relalg.getattr %63 @nation1::@n_name : !db.string
            %66 = db.constant ("BRAZIL") :!db.string
            %67 = db.compare eq %65 : !db.string,%66 : !db.string
            %71 = db.if %67 : i1  -> (!db.decimal<15,2>) {
                %69 = relalg.getattr %63 @map1::@aggfmname2 : !db.decimal<15,2>
                db.yield %69 : !db.decimal<15,2>
            } else {
                %70 = db.constant (0) :!db.decimal<15,2>
                db.yield %70 : !db.decimal<15,2>
            }
            %72 = relalg.addattr %63, @aggfmname1({type=!db.decimal<15,2>}) %71
            relalg.return %72 : !relalg.tuple
        }
        %75 = relalg.aggregation @aggr1 %64 [@map1::@aggfmname1] (%73 : !relalg.tuplestream, %74 : !relalg.tuple) {
            %76 = relalg.aggrfn sum @map2::@aggfmname1 %73 : !db.decimal<15,2>
            %77 = relalg.addattr %74, @aggfmname2({type=!db.decimal<15,2>}) %76
            %78 = relalg.aggrfn sum @map1::@aggfmname2 %73 : !db.decimal<15,2>
            %79 = relalg.addattr %77, @aggfmname3({type=!db.decimal<15,2>}) %78
            relalg.return %79 : !relalg.tuple
        }
        %81 = relalg.map @map3 %75 (%80: !relalg.tuple) {
            %82 = relalg.getattr %80 @aggr1::@aggfmname2 : !db.decimal<15,2>
            %83 = relalg.getattr %80 @aggr1::@aggfmname3 : !db.decimal<15,2>
            %84 = db.div %82 : !db.decimal<15,2>,%83 : !db.decimal<15,2>
            %85 = relalg.addattr %80, @aggfmname4({type=!db.decimal<15,2>}) %84
            relalg.return %85 : !relalg.tuple
        }
        %86 = relalg.sort %81 [(@map1::@aggfmname1,asc)]
        %87 = relalg.materialize %86 [@map1::@aggfmname1,@map3::@aggfmname4] => ["o_year","mkt_share"] : !db.table
        return %87 : !db.table
    }
}


