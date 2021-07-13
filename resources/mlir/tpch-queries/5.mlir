module @querymodule{
    func @main (%executionContext: !util.generic_memref<i8>)  -> !db.table{
        %1 = relalg.basetable @customer { table_identifier="customer", rows=15000 , pkey=["c_custkey"]} columns: {c_custkey => @c_custkey({type=!db.int<64>}),
            c_name => @c_name({type=!db.string}),
            c_address => @c_address({type=!db.string}),
            c_nationkey => @c_nationkey({type=!db.int<64>}),
            c_phone => @c_phone({type=!db.string}),
            c_acctbal => @c_acctbal({type=!db.decimal<15,2>}),
            c_mktsegment => @c_mktsegment({type=!db.string}),
            c_comment => @c_comment({type=!db.string})
        }
        %2 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=!db.int<64>}),
            o_custkey => @o_custkey({type=!db.int<64>}),
            o_orderstatus => @o_orderstatus({type=!db.string}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=!db.int<32>}),
            o_comment => @o_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %4 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
            l_partkey => @l_partkey({type=!db.int<64>}),
            l_suppkey => @l_suppkey({type=!db.int<64>}),
            l_linenumber => @l_linenumber({type=!db.int<32>}),
            l_quantity => @l_quantity({type=!db.decimal<15,2>}),
            l_extendedprice => @l_extendedprice({type=!db.decimal<15,2>}),
            l_discount => @l_discount({type=!db.decimal<15,2>}),
            l_tax => @l_tax({type=!db.decimal<15,2>}),
            l_returnflag => @l_returnflag({type=!db.string}),
            l_linestatus => @l_linestatus({type=!db.string}),
            l_shipdate => @l_shipdate({type=!db.date<day>}),
            l_commitdate => @l_commitdate({type=!db.date<day>}),
            l_receiptdate => @l_receiptdate({type=!db.date<day>}),
            l_shipinstruct => @l_shipinstruct({type=!db.string}),
            l_shipmode => @l_shipmode({type=!db.string}),
            l_comment => @l_comment({type=!db.string})
        }
        %5 = relalg.crossproduct %3, %4
        %6 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=!db.int<64>}),
            s_name => @s_name({type=!db.string}),
            s_address => @s_address({type=!db.string}),
            s_nationkey => @s_nationkey({type=!db.int<64>}),
            s_phone => @s_phone({type=!db.string}),
            s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
            s_comment => @s_comment({type=!db.string})
        }
        %7 = relalg.crossproduct %5, %6
        %8 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<64>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<64>}),
            n_comment => @n_comment({type=!db.string<nullable>})
        }
        %9 = relalg.crossproduct %7, %8
        %10 = relalg.basetable @region { table_identifier="region", rows=5 , pkey=["r_regionkey"]} columns: {r_regionkey => @r_regionkey({type=!db.int<64>}),
            r_name => @r_name({type=!db.string}),
            r_comment => @r_comment({type=!db.string<nullable>})
        }
        %11 = relalg.crossproduct %9, %10
        %13 = relalg.selection %11(%12: !relalg.tuple) {
            %14 = relalg.getattr %12 @customer::@c_custkey : !db.int<64>
            %15 = relalg.getattr %12 @orders::@o_custkey : !db.int<64>
            %16 = db.compare eq %14 : !db.int<64>,%15 : !db.int<64>
            %17 = relalg.getattr %12 @lineitem::@l_orderkey : !db.int<64>
            %18 = relalg.getattr %12 @orders::@o_orderkey : !db.int<64>
            %19 = db.compare eq %17 : !db.int<64>,%18 : !db.int<64>
            %20 = relalg.getattr %12 @lineitem::@l_suppkey : !db.int<64>
            %21 = relalg.getattr %12 @supplier::@s_suppkey : !db.int<64>
            %22 = db.compare eq %20 : !db.int<64>,%21 : !db.int<64>
            %23 = relalg.getattr %12 @customer::@c_nationkey : !db.int<64>
            %24 = relalg.getattr %12 @supplier::@s_nationkey : !db.int<64>
            %25 = db.compare eq %23 : !db.int<64>,%24 : !db.int<64>
            %26 = relalg.getattr %12 @supplier::@s_nationkey : !db.int<64>
            %27 = relalg.getattr %12 @nation::@n_nationkey : !db.int<64>
            %28 = db.compare eq %26 : !db.int<64>,%27 : !db.int<64>
            %29 = relalg.getattr %12 @nation::@n_regionkey : !db.int<64>
            %30 = relalg.getattr %12 @region::@r_regionkey : !db.int<64>
            %31 = db.compare eq %29 : !db.int<64>,%30 : !db.int<64>
            %32 = relalg.getattr %12 @region::@r_name : !db.string
            %33 = db.constant ("ASIA") :!db.string
            %34 = db.compare eq %32 : !db.string,%33 : !db.string
            %35 = relalg.getattr %12 @orders::@o_orderdate : !db.date<day>
            %36 = db.constant ("1994-01-01") :!db.date<day>
            %37 = db.compare gte %35 : !db.date<day>,%36 : !db.date<day>
            %38 = relalg.getattr %12 @orders::@o_orderdate : !db.date<day>
            %39 = db.constant ("1995-01-01") :!db.date<day>
            %40 = db.compare lt %38 : !db.date<day>,%39 : !db.date<day>
            %41 = db.and %16 : !db.bool,%19 : !db.bool,%22 : !db.bool,%25 : !db.bool,%28 : !db.bool,%31 : !db.bool,%34 : !db.bool,%37 : !db.bool,%40 : !db.bool
            relalg.return %41 : !db.bool
        }
        %43 = relalg.map @map1 %13 (%42: !relalg.tuple) {
            %44 = relalg.getattr %42 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %45 = db.constant (1) :!db.decimal<15,2>
            %46 = relalg.getattr %42 @lineitem::@l_discount : !db.decimal<15,2>
            %47 = db.sub %45 : !db.decimal<15,2>,%46 : !db.decimal<15,2>
            %48 = db.mul %44 : !db.decimal<15,2>,%47 : !db.decimal<15,2>
            relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %48
            relalg.return
        }
        %50 = relalg.aggregation @aggr1 %43 [@nation::@n_name] (%49 : !relalg.relation) {
            %51 = relalg.aggrfn sum @map1::@aggfmname1 %49 : !db.decimal<15,2>
            relalg.addattr @aggfmname2({type=!db.decimal<15,2>}) %51
            relalg.return
        }
        %52 = relalg.sort %50 [(@aggr1::@aggfmname2,desc)]
        %53 = relalg.materialize %52 [@nation::@n_name,@aggr1::@aggfmname2] => ["n_name","revenue"] : !db.table
        return %53 : !db.table
    }
}

