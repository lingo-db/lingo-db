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
        %6 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<64>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<64>}),
            n_comment => @n_comment({type=!db.string<nullable>})
        }
        %7 = relalg.crossproduct %5, %6
        %9 = relalg.selection %7(%8: !relalg.tuple) {
            %10 = relalg.getattr %8 @customer::@c_custkey : !db.int<64>
            %11 = relalg.getattr %8 @orders::@o_custkey : !db.int<64>
            %12 = db.compare eq %10 : !db.int<64>,%11 : !db.int<64>
            %13 = relalg.getattr %8 @lineitem::@l_orderkey : !db.int<64>
            %14 = relalg.getattr %8 @orders::@o_orderkey : !db.int<64>
            %15 = db.compare eq %13 : !db.int<64>,%14 : !db.int<64>
            %16 = relalg.getattr %8 @orders::@o_orderdate : !db.date<day>
            %17 = db.constant ("1993-10-01") :!db.date<day>
            %18 = db.compare gte %16 : !db.date<day>,%17 : !db.date<day>
            %19 = relalg.getattr %8 @orders::@o_orderdate : !db.date<day>
            %20 = db.constant ("1994-01-01") :!db.date<day>
            %21 = db.compare lt %19 : !db.date<day>,%20 : !db.date<day>
            %22 = relalg.getattr %8 @lineitem::@l_returnflag : !db.string
            %23 = db.constant ("R") :!db.string
            %24 = db.compare eq %22 : !db.string,%23 : !db.string
            %25 = relalg.getattr %8 @customer::@c_nationkey : !db.int<64>
            %26 = relalg.getattr %8 @nation::@n_nationkey : !db.int<64>
            %27 = db.compare eq %25 : !db.int<64>,%26 : !db.int<64>
            %28 = db.and %12 : !db.bool,%15 : !db.bool,%18 : !db.bool,%21 : !db.bool,%24 : !db.bool,%27 : !db.bool
            relalg.return %28 : !db.bool
        }
        %30 = relalg.map @map1 %9 (%29: !relalg.tuple) {
            %31 = relalg.getattr %29 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %32 = db.constant (1) :!db.decimal<15,2>
            %33 = relalg.getattr %29 @lineitem::@l_discount : !db.decimal<15,2>
            %34 = db.sub %32 : !db.decimal<15,2>,%33 : !db.decimal<15,2>
            %35 = db.mul %31 : !db.decimal<15,2>,%34 : !db.decimal<15,2>
            relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %35
            relalg.return
        }
        %37 = relalg.aggregation @aggr1 %30 [@customer::@c_custkey,@customer::@c_name,@customer::@c_acctbal,@customer::@c_phone,@nation::@n_name,@customer::@c_address,@customer::@c_comment] (%36 : !relalg.relation) {
            %38 = relalg.aggrfn sum @map1::@aggfmname1 %36 : !db.decimal<15,2>
            relalg.addattr @aggfmname2({type=!db.decimal<15,2>}) %38
            relalg.return
        }
        %39 = relalg.sort %37 [(@aggr1::@aggfmname2,desc)]
        %40 = relalg.limit 20 %39
        %41 = relalg.materialize %40 [@customer::@c_custkey,@customer::@c_name,@aggr1::@aggfmname2,@customer::@c_acctbal,@nation::@n_name,@customer::@c_address,@customer::@c_phone,@customer::@c_comment] => ["c_custkey","c_name","revenue","c_acctbal","n_name","c_address","c_phone","c_comment"] : !db.table
        return %41 : !db.table
    }
}

