module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=i32}),
            p_name => @p_name({type=!db.string}),
            p_mfgr => @p_mfgr({type=!db.string}),
            p_brand => @p_brand({type=!db.string}),
            p_type => @p_type({type=!db.string}),
            p_size => @p_size({type=i32}),
            p_container => @p_container({type=!db.string}),
            p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
            p_comment => @p_comment({type=!db.string})
        }
        %2 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=i32}),
            s_name => @s_name({type=!db.string}),
            s_address => @s_address({type=!db.string}),
            s_nationkey => @s_nationkey({type=i32}),
            s_phone => @s_phone({type=!db.string}),
            s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
            s_comment => @s_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %4 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=i32}),
            l_partkey => @l_partkey({type=i32}),
            l_suppkey => @l_suppkey({type=i32}),
            l_linenumber => @l_linenumber({type=i32}),
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
        %6 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=i32}),
            o_custkey => @o_custkey({type=i32}),
            o_orderstatus => @o_orderstatus({type=!db.char<1>}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=i32}),
            o_comment => @o_comment({type=!db.string})
        }
        %7 = relalg.crossproduct %5, %6
        %8 = relalg.basetable @customer { table_identifier="customer", rows=15000 , pkey=["c_custkey"]} columns: {c_custkey => @c_custkey({type=i32}),
            c_name => @c_name({type=!db.string}),
            c_address => @c_address({type=!db.string}),
            c_nationkey => @c_nationkey({type=i32}),
            c_phone => @c_phone({type=!db.string}),
            c_acctbal => @c_acctbal({type=!db.decimal<15,2>}),
            c_mktsegment => @c_mktsegment({type=!db.string}),
            c_comment => @c_comment({type=!db.string})
        }
        %9 = relalg.crossproduct %7, %8
        %10 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=i32}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=i32}),
            n_comment => @n_comment({type=!db.nullable<!db.string>})
        }
        %11 = relalg.crossproduct %9, %10
        %12 = relalg.basetable @nation1 { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=i32}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=i32}),
            n_comment => @n_comment({type=!db.nullable<!db.string>})
        }
        %13 = relalg.crossproduct %11, %12
        %14 = relalg.basetable @region { table_identifier="region", rows=5 , pkey=["r_regionkey"]} columns: {r_regionkey => @r_regionkey({type=i32}),
            r_name => @r_name({type=!db.string}),
            r_comment => @r_comment({type=!db.nullable<!db.string>})
        }
        %15 = relalg.crossproduct %13, %14
        %17 = relalg.selection %15(%16: !relalg.tuple) {
            %18 = relalg.getattr %16 @part::@p_partkey : i32
            %19 = relalg.getattr %16 @lineitem::@l_partkey : i32
            %20 = db.compare eq %18 : i32,%19 : i32
            %21 = relalg.getattr %16 @supplier::@s_suppkey : i32
            %22 = relalg.getattr %16 @lineitem::@l_suppkey : i32
            %23 = db.compare eq %21 : i32,%22 : i32
            %24 = relalg.getattr %16 @lineitem::@l_orderkey : i32
            %25 = relalg.getattr %16 @orders::@o_orderkey : i32
            %26 = db.compare eq %24 : i32,%25 : i32
            %27 = relalg.getattr %16 @orders::@o_custkey : i32
            %28 = relalg.getattr %16 @customer::@c_custkey : i32
            %29 = db.compare eq %27 : i32,%28 : i32
            %30 = relalg.getattr %16 @customer::@c_nationkey : i32
            %31 = relalg.getattr %16 @nation::@n_nationkey : i32
            %32 = db.compare eq %30 : i32,%31 : i32
            %33 = relalg.getattr %16 @nation::@n_regionkey : i32
            %34 = relalg.getattr %16 @region::@r_regionkey : i32
            %35 = db.compare eq %33 : i32,%34 : i32
            %36 = relalg.getattr %16 @region::@r_name : !db.string
            %37 = db.constant ("AMERICA") :!db.string
            %38 = db.compare eq %36 : !db.string,%37 : !db.string
            %39 = relalg.getattr %16 @supplier::@s_nationkey : i32
            %40 = relalg.getattr %16 @nation1::@n_nationkey : i32
            %41 = db.compare eq %39 : i32,%40 : i32
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
            %56 = relalg.addattr %52, @aggfmname1({type=i64}) %55
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
            %68 = db.derive_truth %67 : i1
            %72 = scf.if %68  -> (!db.decimal<15,2>) {
                %70 = relalg.getattr %63 @map1::@aggfmname2 : !db.decimal<15,2>
                scf.yield %70 : !db.decimal<15,2>
            } else {
                %71 = db.constant (0) :!db.decimal<15,2>
                scf.yield %71 : !db.decimal<15,2>
            }
            %73 = relalg.addattr %63, @aggfmname1({type=!db.decimal<15,2>}) %72
            relalg.return %73 : !relalg.tuple
        }
        %76 = relalg.aggregation @aggr1 %64 [@map1::@aggfmname1] (%74 : !relalg.tuplestream, %75 : !relalg.tuple) {
            %77 = relalg.aggrfn sum @map2::@aggfmname1 %74 : !db.decimal<15,2>
            %78 = relalg.addattr %75, @aggfmname2({type=!db.decimal<15,2>}) %77
            %79 = relalg.aggrfn sum @map1::@aggfmname2 %74 : !db.decimal<15,2>
            %80 = relalg.addattr %78, @aggfmname3({type=!db.decimal<15,2>}) %79
            relalg.return %80 : !relalg.tuple
        }
        %82 = relalg.map @map3 %76 (%81: !relalg.tuple) {
            %83 = relalg.getattr %81 @aggr1::@aggfmname2 : !db.decimal<15,2>
            %84 = relalg.getattr %81 @aggr1::@aggfmname3 : !db.decimal<15,2>
            %85 = db.div %83 : !db.decimal<15,2>,%84 : !db.decimal<15,2>
            %86 = relalg.addattr %81, @aggfmname4({type=!db.decimal<15,2>}) %85
            relalg.return %86 : !relalg.tuple
        }
        %87 = relalg.sort %82 [(@map1::@aggfmname1,asc)]
        %88 = relalg.materialize %87 [@map1::@aggfmname1,@map3::@aggfmname4] => ["o_year","mkt_share"] : !db.table
        return %88 : !db.table
    }
}

