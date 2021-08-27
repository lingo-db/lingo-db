module @querymodule{
    func @main ()  -> !db.table{
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
        %7 = relalg.selection %5(%6: !relalg.tuple) {
            %8 = relalg.basetable @lineitem1 { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
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
            %11 = relalg.aggregation @aggr1 %8 [@lineitem1::@l_orderkey] (%9 : !relalg.relation, %10 : !relalg.tuple) {
                %12 = relalg.aggrfn sum @lineitem1::@l_quantity %9 : !db.decimal<15,2>
                %13 = relalg.addattr %10, @aggfmname1({type=!db.decimal<15,2>}) %12
                relalg.return %13 : !relalg.tuple
            }
            %15 = relalg.selection %11(%14: !relalg.tuple) {
                %16 = relalg.getattr %14 @aggr1::@aggfmname1 : !db.decimal<15,2>
                %17 = db.constant (300) :!db.decimal<15,2>
                %18 = db.compare gt %16 : !db.decimal<15,2>,%17 : !db.decimal<15,2>
                relalg.return %18 : !db.bool
            }
            %19 = relalg.projection all [@lineitem1::@l_orderkey]%15
            %20 = relalg.getattr %6 @orders::@o_orderkey : !db.int<64>
            %21 = relalg.in %20 : !db.int<64>, %19
            %22 = relalg.getattr %6 @customer::@c_custkey : !db.int<64>
            %23 = relalg.getattr %6 @orders::@o_custkey : !db.int<64>
            %24 = db.compare eq %22 : !db.int<64>,%23 : !db.int<64>
            %25 = relalg.getattr %6 @orders::@o_orderkey : !db.int<64>
            %26 = relalg.getattr %6 @lineitem::@l_orderkey : !db.int<64>
            %27 = db.compare eq %25 : !db.int<64>,%26 : !db.int<64>
            %28 = db.and %21 : !db.bool,%24 : !db.bool,%27 : !db.bool
            relalg.return %28 : !db.bool
        }
        %31 = relalg.aggregation @aggr2 %7 [@customer::@c_name,@customer::@c_custkey,@orders::@o_orderkey,@orders::@o_orderdate,@orders::@o_totalprice] (%29 : !relalg.relation, %30 : !relalg.tuple) {
            %32 = relalg.aggrfn sum @lineitem::@l_quantity %29 : !db.decimal<15,2>
            %33 = relalg.addattr %30, @aggfmname1({type=!db.decimal<15,2>}) %32
            relalg.return %33 : !relalg.tuple
        }
        %34 = relalg.sort %31 [(@orders::@o_totalprice,desc),(@orders::@o_orderdate,asc)]
        %35 = relalg.limit 100 %34
        %36 = relalg.materialize %35 [@customer::@c_name,@customer::@c_custkey,@orders::@o_orderkey,@orders::@o_orderdate,@orders::@o_totalprice,@aggr2::@aggfmname1] => ["c_name","c_custkey","o_orderkey","o_orderdate","o_totalprice","sum"] : !db.table
        return %36 : !db.table
    }
}

