//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                        s_name  |                       numwait  |
//CHECK: -------------------------------------------------------------------
//CHECK: |          "Supplier#000000445"  |                            16  |
//CHECK: |          "Supplier#000000825"  |                            16  |
//CHECK: |          "Supplier#000000709"  |                            15  |
//CHECK: |          "Supplier#000000762"  |                            15  |
//CHECK: |          "Supplier#000000357"  |                            14  |
//CHECK: |          "Supplier#000000399"  |                            14  |
//CHECK: |          "Supplier#000000496"  |                            14  |
//CHECK: |          "Supplier#000000977"  |                            13  |
//CHECK: |          "Supplier#000000144"  |                            12  |
//CHECK: |          "Supplier#000000188"  |                            12  |
//CHECK: |          "Supplier#000000415"  |                            12  |
//CHECK: |          "Supplier#000000472"  |                            12  |
//CHECK: |          "Supplier#000000633"  |                            12  |
//CHECK: |          "Supplier#000000708"  |                            12  |
//CHECK: |          "Supplier#000000889"  |                            12  |
//CHECK: |          "Supplier#000000380"  |                            11  |
//CHECK: |          "Supplier#000000602"  |                            11  |
//CHECK: |          "Supplier#000000659"  |                            11  |
//CHECK: |          "Supplier#000000821"  |                            11  |
//CHECK: |          "Supplier#000000929"  |                            11  |
//CHECK: |          "Supplier#000000262"  |                            10  |
//CHECK: |          "Supplier#000000460"  |                            10  |
//CHECK: |          "Supplier#000000486"  |                            10  |
//CHECK: |          "Supplier#000000669"  |                            10  |
//CHECK: |          "Supplier#000000718"  |                            10  |
//CHECK: |          "Supplier#000000778"  |                            10  |
//CHECK: |          "Supplier#000000167"  |                             9  |
//CHECK: |          "Supplier#000000578"  |                             9  |
//CHECK: |          "Supplier#000000673"  |                             9  |
//CHECK: |          "Supplier#000000687"  |                             9  |
//CHECK: |          "Supplier#000000074"  |                             8  |
//CHECK: |          "Supplier#000000565"  |                             8  |
//CHECK: |          "Supplier#000000648"  |                             8  |
//CHECK: |          "Supplier#000000918"  |                             8  |
//CHECK: |          "Supplier#000000427"  |                             7  |
//CHECK: |          "Supplier#000000503"  |                             7  |
//CHECK: |          "Supplier#000000610"  |                             7  |
//CHECK: |          "Supplier#000000670"  |                             7  |
//CHECK: |          "Supplier#000000811"  |                             7  |
//CHECK: |          "Supplier#000000114"  |                             6  |
//CHECK: |          "Supplier#000000379"  |                             6  |
//CHECK: |          "Supplier#000000436"  |                             6  |
//CHECK: |          "Supplier#000000500"  |                             6  |
//CHECK: |          "Supplier#000000660"  |                             6  |
//CHECK: |          "Supplier#000000788"  |                             6  |
//CHECK: |          "Supplier#000000846"  |                             6  |
//CHECK: |          "Supplier#000000920"  |                             4  |
module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=!db.int<64>}),
            s_name => @s_name({type=!db.string}),
            s_address => @s_address({type=!db.string}),
            s_nationkey => @s_nationkey({type=!db.int<64>}),
            s_phone => @s_phone({type=!db.string}),
            s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
            s_comment => @s_comment({type=!db.string})
        }
        %2 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
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
        %3 = relalg.crossproduct %1, %2
        %4 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=!db.int<64>}),
            o_custkey => @o_custkey({type=!db.int<64>}),
            o_orderstatus => @o_orderstatus({type=!db.string}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=!db.int<32>}),
            o_comment => @o_comment({type=!db.string})
        }
        %5 = relalg.crossproduct %3, %4
        %6 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<64>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<64>}),
            n_comment => @n_comment({type=!db.string<nullable>})
        }
        %7 = relalg.crossproduct %5, %6
        %9 = relalg.selection %7(%8: !relalg.tuple) {
            %10 = relalg.getattr %8 @supplier::@s_suppkey : !db.int<64>
            %11 = relalg.getattr %8 @lineitem::@l_suppkey : !db.int<64>
            %12 = db.compare eq %10 : !db.int<64>,%11 : !db.int<64>
            %13 = relalg.getattr %8 @orders::@o_orderkey : !db.int<64>
            %14 = relalg.getattr %8 @lineitem::@l_orderkey : !db.int<64>
            %15 = db.compare eq %13 : !db.int<64>,%14 : !db.int<64>
            %16 = relalg.getattr %8 @orders::@o_orderstatus : !db.string
            %17 = db.constant ("F") :!db.string
            %18 = db.compare eq %16 : !db.string,%17 : !db.string
            %19 = relalg.getattr %8 @lineitem::@l_receiptdate : !db.date<day>
            %20 = relalg.getattr %8 @lineitem::@l_commitdate : !db.date<day>
            %21 = db.compare gt %19 : !db.date<day>,%20 : !db.date<day>
            %22 = relalg.basetable @lineitem1 { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
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
            %24 = relalg.selection %22(%23: !relalg.tuple) {
                %25 = relalg.getattr %23 @lineitem1::@l_orderkey : !db.int<64>
                %26 = relalg.getattr %8 @lineitem::@l_orderkey : !db.int<64>
                %27 = db.compare eq %25 : !db.int<64>,%26 : !db.int<64>
                %28 = relalg.getattr %23 @lineitem1::@l_suppkey : !db.int<64>
                %29 = relalg.getattr %8 @lineitem::@l_suppkey : !db.int<64>
                %30 = db.compare neq %28 : !db.int<64>,%29 : !db.int<64>
                %31 = db.and %27 : !db.bool,%30 : !db.bool
                relalg.return %31 : !db.bool
            }
            %32 = relalg.exists%24
            %33 = relalg.basetable @lineitem2 { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
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
            %35 = relalg.selection %33(%34: !relalg.tuple) {
                %36 = relalg.getattr %34 @lineitem2::@l_orderkey : !db.int<64>
                %37 = relalg.getattr %8 @lineitem::@l_orderkey : !db.int<64>
                %38 = db.compare eq %36 : !db.int<64>,%37 : !db.int<64>
                %39 = relalg.getattr %34 @lineitem2::@l_suppkey : !db.int<64>
                %40 = relalg.getattr %8 @lineitem::@l_suppkey : !db.int<64>
                %41 = db.compare neq %39 : !db.int<64>,%40 : !db.int<64>
                %42 = relalg.getattr %34 @lineitem2::@l_receiptdate : !db.date<day>
                %43 = relalg.getattr %34 @lineitem2::@l_commitdate : !db.date<day>
                %44 = db.compare gt %42 : !db.date<day>,%43 : !db.date<day>
                %45 = db.and %38 : !db.bool,%41 : !db.bool,%44 : !db.bool
                relalg.return %45 : !db.bool
            }
            %46 = relalg.exists%35
            %47 = db.not %46 : !db.bool
            %48 = relalg.getattr %8 @supplier::@s_nationkey : !db.int<64>
            %49 = relalg.getattr %8 @nation::@n_nationkey : !db.int<64>
            %50 = db.compare eq %48 : !db.int<64>,%49 : !db.int<64>
            %51 = relalg.getattr %8 @nation::@n_name : !db.string
            %52 = db.constant ("SAUDI ARABIA") :!db.string
            %53 = db.compare eq %51 : !db.string,%52 : !db.string
            %54 = db.and %12 : !db.bool,%15 : !db.bool,%18 : !db.bool,%21 : !db.bool,%32 : !db.bool,%47 : !db.bool,%50 : !db.bool,%53 : !db.bool
            relalg.return %54 : !db.bool
        }
        %57 = relalg.aggregation @aggr2 %9 [@supplier::@s_name] (%55 : !relalg.tuplestream, %56 : !relalg.tuple) {
            %58 = relalg.count %55
            %59 = relalg.addattr %56, @aggfmname1({type=!db.int<64>}) %58
            relalg.return %59 : !relalg.tuple
        }
        %60 = relalg.sort %57 [(@aggr2::@aggfmname1,desc),(@supplier::@s_name,asc)]
        %61 = relalg.limit 100 %60
        %62 = relalg.materialize %61 [@supplier::@s_name,@aggr2::@aggfmname1] => ["s_name","numwait"] : !db.table
        return %62 : !db.table
    }
}


