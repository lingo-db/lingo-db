//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |               o_orderpriority  |                   order_count  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                    "1-URGENT"  |                           999  |
//CHECK: |                      "2-HIGH"  |                           997  |
//CHECK: |                    "3-MEDIUM"  |                          1031  |
//CHECK: |             "4-NOT SPECIFIED"  |                           989  |
//CHECK: |                       "5-LOW"  |                          1077  |
module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=!db.int<32>}),
            o_custkey => @o_custkey({type=!db.int<32>}),
            o_orderstatus => @o_orderstatus({type=!db.char<1>}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=!db.int<32>}),
            o_comment => @o_comment({type=!db.string})
        }
        %3 = relalg.selection %1(%2: !relalg.tuple) {
            %4 = relalg.getattr %2 @orders::@o_orderdate : !db.date<day>
            %5 = db.constant ("1993-07-01") :!db.date<day>
            %6 = db.compare gte %4 : !db.date<day>,%5 : !db.date<day>
            %7 = relalg.getattr %2 @orders::@o_orderdate : !db.date<day>
            %8 = db.constant ("1993-10-01") :!db.date<day>
            %9 = db.compare lt %7 : !db.date<day>,%8 : !db.date<day>
            %10 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<32>}),
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
            %12 = relalg.selection %10(%11: !relalg.tuple) {
                %13 = relalg.getattr %11 @lineitem::@l_orderkey : !db.int<32>
                %14 = relalg.getattr %2 @orders::@o_orderkey : !db.int<32>
                %15 = db.compare eq %13 : !db.int<32>,%14 : !db.int<32>
                %16 = relalg.getattr %11 @lineitem::@l_commitdate : !db.date<day>
                %17 = relalg.getattr %11 @lineitem::@l_receiptdate : !db.date<day>
                %18 = db.compare lt %16 : !db.date<day>,%17 : !db.date<day>
                %19 = db.and %15 : i1,%18 : i1
                relalg.return %19 : i1
            }
            %20 = relalg.exists%12
            %21 = db.and %6 : i1,%9 : i1,%20 : i1
            relalg.return %21 : i1
        }
        %24 = relalg.aggregation @aggr1 %3 [@orders::@o_orderpriority] (%22 : !relalg.tuplestream, %23 : !relalg.tuple) {
            %25 = relalg.count %22
            %26 = relalg.addattr %23, @aggfmname1({type=!db.int<64>}) %25
            relalg.return %26 : !relalg.tuple
        }
        %27 = relalg.sort %24 [(@orders::@o_orderpriority,asc)]
        %28 = relalg.materialize %27 [@orders::@o_orderpriority,@aggr1::@aggfmname1] => ["o_orderpriority","order_count"] : !db.table
        return %28 : !db.table
    }
}


