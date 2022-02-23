module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @customer { table_identifier="customer", rows=15000 , pkey=["c_custkey"]} columns: {c_custkey => @c_custkey({type=!db.int<32>}),
            c_name => @c_name({type=!db.string}),
            c_address => @c_address({type=!db.string}),
            c_nationkey => @c_nationkey({type=!db.int<32>}),
            c_phone => @c_phone({type=!db.string}),
            c_acctbal => @c_acctbal({type=!db.decimal<15,2>}),
            c_mktsegment => @c_mktsegment({type=!db.string}),
            c_comment => @c_comment({type=!db.string})
        }
        %2 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=!db.int<32>}),
            o_custkey => @o_custkey({type=!db.int<32>}),
            o_orderstatus => @o_orderstatus({type=!db.char<1>}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=!db.int<32>}),
            o_comment => @o_comment({type=!db.string})
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
        %6 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<32>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<32>}),
            n_comment => @n_comment({type=!db.nullable<!db.string>})
        }
        %7 = relalg.crossproduct %5, %6
        %9 = relalg.selection %7(%8: !relalg.tuple) {
            %10 = relalg.getattr %8 @customer::@c_custkey : !db.int<32>
            %11 = relalg.getattr %8 @orders::@o_custkey : !db.int<32>
            %12 = db.compare eq %10 : !db.int<32>,%11 : !db.int<32>
            %13 = relalg.getattr %8 @lineitem::@l_orderkey : !db.int<32>
            %14 = relalg.getattr %8 @orders::@o_orderkey : !db.int<32>
            %15 = db.compare eq %13 : !db.int<32>,%14 : !db.int<32>
            %16 = relalg.getattr %8 @orders::@o_orderdate : !db.date<day>
            %17 = db.constant ("1993-10-01") :!db.date<day>
            %18 = db.compare gte %16 : !db.date<day>,%17 : !db.date<day>
            %19 = relalg.getattr %8 @orders::@o_orderdate : !db.date<day>
            %20 = db.constant ("1994-01-01") :!db.date<day>
            %21 = db.compare lt %19 : !db.date<day>,%20 : !db.date<day>
            %22 = relalg.getattr %8 @lineitem::@l_returnflag : !db.char<1>
            %23 = db.constant ("R") :!db.char<1>
            %24 = db.compare eq %22 : !db.char<1>,%23 : !db.char<1>
            %25 = relalg.getattr %8 @customer::@c_nationkey : !db.int<32>
            %26 = relalg.getattr %8 @nation::@n_nationkey : !db.int<32>
            %27 = db.compare eq %25 : !db.int<32>,%26 : !db.int<32>
            %28 = db.and %12 : !db.bool,%15 : !db.bool,%18 : !db.bool,%21 : !db.bool,%24 : !db.bool,%27 : !db.bool
            relalg.return %28 : !db.bool
        }
        %30 = relalg.map @map %9 (%29: !relalg.tuple) {
            %31 = relalg.getattr %29 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %32 = db.constant (1) :!db.decimal<15,2>
            %33 = relalg.getattr %29 @lineitem::@l_discount : !db.decimal<15,2>
            %34 = db.sub %32 : !db.decimal<15,2>,%33 : !db.decimal<15,2>
            %35 = db.mul %31 : !db.decimal<15,2>,%34 : !db.decimal<15,2>
            %36 = relalg.addattr %29, @aggfmname1({type=!db.decimal<15,2>}) %35
            relalg.return %36 : !relalg.tuple
        }
        %39 = relalg.aggregation @aggr %30 [@customer::@c_custkey,@customer::@c_name,@customer::@c_acctbal,@customer::@c_phone,@nation::@n_name,@customer::@c_address,@customer::@c_comment] (%37 : !relalg.tuplestream, %38 : !relalg.tuple) {
            %40 = relalg.aggrfn sum @map::@aggfmname1 %37 : !db.decimal<15,2>
            %41 = relalg.addattr %38, @aggfmname2({type=!db.decimal<15,2>}) %40
            relalg.return %41 : !relalg.tuple
        }
        %42 = relalg.sort %39 [(@aggr::@aggfmname2,desc)]
        %43 = relalg.limit 20 %42
        %44 = relalg.materialize %43 [@customer::@c_custkey,@customer::@c_name,@aggr::@aggfmname2,@customer::@c_acctbal,@nation::@n_name,@customer::@c_address,@customer::@c_phone,@customer::@c_comment] => ["c_custkey","c_name","revenue","c_acctbal","n_name","c_address","c_phone","c_comment"] : !db.table
        return %44 : !db.table
    }
}

