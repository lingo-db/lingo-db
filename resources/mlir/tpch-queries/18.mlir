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
            %10 = relalg.aggregation @aggr1 %8 [@lineitem1::@l_orderkey] (%9 : !relalg.relation) {
                %11 = relalg.aggrfn sum @lineitem1::@l_quantity %9 : !db.decimal<15,2>
                relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %11
                relalg.return
            }
            %13 = relalg.selection %10(%12: !relalg.tuple) {
                %14 = relalg.getattr %12 @aggr1::@aggfmname1 : !db.decimal<15,2>
                %15 = db.constant (300) :!db.decimal<15,2>
                %16 = db.compare gt %14 : !db.decimal<15,2>,%15 : !db.decimal<15,2>
                relalg.return %16 : !db.bool
            }
            %17 = relalg.projection all [@lineitem1::@l_orderkey]%13
            %18 = relalg.getattr %6 @orders::@o_orderkey : !db.int<64>
            %19 = relalg.in %18 : !db.int<64>, %17
            %20 = relalg.getattr %6 @customer::@c_custkey : !db.int<64>
            %21 = relalg.getattr %6 @orders::@o_custkey : !db.int<64>
            %22 = db.compare eq %20 : !db.int<64>,%21 : !db.int<64>
            %23 = relalg.getattr %6 @orders::@o_orderkey : !db.int<64>
            %24 = relalg.getattr %6 @lineitem::@l_orderkey : !db.int<64>
            %25 = db.compare eq %23 : !db.int<64>,%24 : !db.int<64>
            %26 = db.and %19 : !db.bool,%22 : !db.bool,%25 : !db.bool
            relalg.return %26 : !db.bool
        }
        %28 = relalg.aggregation @aggr2 %7 [@customer::@c_name,@customer::@c_custkey,@orders::@o_orderkey,@orders::@o_orderdate,@orders::@o_totalprice] (%27 : !relalg.relation) {
            %29 = relalg.aggrfn sum @lineitem::@l_quantity %27 : !db.decimal<15,2>
            relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %29
            relalg.return
        }
        %30 = relalg.sort %28 [(@orders::@o_totalprice,desc),(@orders::@o_orderdate,asc)]
        %31 = relalg.limit 100 %30
        %32 = relalg.materialize %31 [@customer::@c_name,@customer::@c_custkey,@orders::@o_orderkey,@orders::@o_orderdate,@orders::@o_totalprice,@aggr2::@aggfmname1] => ["c_name","c_custkey","o_orderkey","o_orderdate","o_totalprice","sum"] : !db.table
        return %32 : !db.table
    }
}

