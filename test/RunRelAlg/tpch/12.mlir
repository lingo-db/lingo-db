//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                    l_shipmode  |               high_line_count  |                low_line_count  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                        "MAIL"  |                           647  |                           945  |
//CHECK: |                        "SHIP"  |                           620  |                           943  |
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
        %2 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<32>}),
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
        %3 = relalg.crossproduct %1, %2
        %5 = relalg.selection %3(%4: !relalg.tuple) {
            %6 = relalg.getattr %4 @orders::@o_orderkey : !db.int<32>
            %7 = relalg.getattr %4 @lineitem::@l_orderkey : !db.int<32>
            %8 = db.compare eq %6 : !db.int<32>,%7 : !db.int<32>
            %9 = relalg.getattr %4 @lineitem::@l_shipmode : !db.string
            %10 = db.constant ("MAIL") :!db.string
            %11 = db.compare eq %9 : !db.string,%10 : !db.string
            %12 = db.constant ("SHIP") :!db.string
            %13 = db.compare eq %9 : !db.string,%12 : !db.string
            %14 = db.or %11 : i1,%13 : i1
            %15 = relalg.getattr %4 @lineitem::@l_commitdate : !db.date<day>
            %16 = relalg.getattr %4 @lineitem::@l_receiptdate : !db.date<day>
            %17 = db.compare lt %15 : !db.date<day>,%16 : !db.date<day>
            %18 = relalg.getattr %4 @lineitem::@l_shipdate : !db.date<day>
            %19 = relalg.getattr %4 @lineitem::@l_commitdate : !db.date<day>
            %20 = db.compare lt %18 : !db.date<day>,%19 : !db.date<day>
            %21 = relalg.getattr %4 @lineitem::@l_receiptdate : !db.date<day>
            %22 = db.constant ("1994-01-01") :!db.date<day>
            %23 = db.compare gte %21 : !db.date<day>,%22 : !db.date<day>
            %24 = relalg.getattr %4 @lineitem::@l_receiptdate : !db.date<day>
            %25 = db.constant ("1995-01-01") :!db.date<day>
            %26 = db.compare lt %24 : !db.date<day>,%25 : !db.date<day>
            %27 = db.and %8 : i1,%14 : i1,%17 : i1,%20 : i1,%23 : i1,%26 : i1
            relalg.return %27 : i1
        }
        %29 = relalg.map @map %5 (%28: !relalg.tuple) {
            %30 = relalg.getattr %28 @orders::@o_orderpriority : !db.string
            %31 = db.constant ("1-URGENT") :!db.string
            %32 = db.compare eq %30 : !db.string,%31 : !db.string
            %33 = relalg.getattr %28 @orders::@o_orderpriority : !db.string
            %34 = db.constant ("2-HIGH") :!db.string
            %35 = db.compare eq %33 : !db.string,%34 : !db.string
            %36 = db.or %32 : i1,%35 : i1
            %40 = db.if %36 : i1  -> (!db.int<64>) {
                %38 = db.constant (1) :!db.int<64>
                db.yield %38 : !db.int<64>
            } else {
                %39 = db.constant (0) :!db.int<64>
                db.yield %39 : !db.int<64>
            }
            %41 = relalg.addattr %28, @aggfmname1({type=!db.int<64>}) %40
            %42 = relalg.getattr %28 @orders::@o_orderpriority : !db.string
            %43 = db.constant ("1-URGENT") :!db.string
            %44 = db.compare neq %42 : !db.string,%43 : !db.string
            %45 = relalg.getattr %28 @orders::@o_orderpriority : !db.string
            %46 = db.constant ("2-HIGH") :!db.string
            %47 = db.compare neq %45 : !db.string,%46 : !db.string
            %48 = db.and %44 : i1,%47 : i1
            %52 = db.if %48 : i1  -> (!db.int<64>) {
                %50 = db.constant (1) :!db.int<64>
                db.yield %50 : !db.int<64>
            } else {
                %51 = db.constant (0) :!db.int<64>
                db.yield %51 : !db.int<64>
            }
            %53 = relalg.addattr %41, @aggfmname3({type=!db.int<64>}) %52
            relalg.return %53 : !relalg.tuple
        }
        %56 = relalg.aggregation @aggr %29 [@lineitem::@l_shipmode] (%54 : !relalg.tuplestream, %55 : !relalg.tuple) {
            %57 = relalg.aggrfn sum @map::@aggfmname1 %54 : !db.int<64>
            %58 = relalg.addattr %55, @aggfmname2({type=!db.int<64>}) %57
            %59 = relalg.aggrfn sum @map::@aggfmname3 %54 : !db.int<64>
            %60 = relalg.addattr %58, @aggfmname4({type=!db.int<64>}) %59
            relalg.return %60 : !relalg.tuple
        }
        %61 = relalg.sort %56 [(@lineitem::@l_shipmode,asc)]
        %62 = relalg.materialize %61 [@lineitem::@l_shipmode,@aggr::@aggfmname2,@aggr::@aggfmname4] => ["l_shipmode","high_line_count","low_line_count"] : !db.table
        return %62 : !db.table
    }
}


