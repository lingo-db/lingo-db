module @querymodule{
    func @main (%executionContext: !util.generic_memref<i8>)  -> !db.table{
        %1 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=!db.int<64>}),
            o_custkey => @o_custkey({type=!db.int<64>}),
            o_orderstatus => @o_orderstatus({type=!db.string}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=!db.int<32>}),
            o_comment => @o_comment({type=!db.string})
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
        %5 = relalg.selection %3(%4: !relalg.tuple) {
            %6 = relalg.getattr %4 @orders::@o_orderkey : !db.int<64>
            %7 = relalg.getattr %4 @lineitem::@l_orderkey : !db.int<64>
            %8 = db.compare eq %6 : !db.int<64>,%7 : !db.int<64>
            %9 = relalg.getattr %4 @lineitem::@l_shipmode : !db.string
            %10 = db.constant ("MAIL") :!db.string
            %11 = db.compare eq %9 : !db.string,%10 : !db.string
            %12 = db.constant ("SHIP") :!db.string
            %13 = db.compare eq %9 : !db.string,%12 : !db.string
            %14 = db.or %11 : !db.bool,%13 : !db.bool
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
            %27 = db.and %8 : !db.bool,%14 : !db.bool,%17 : !db.bool,%20 : !db.bool,%23 : !db.bool,%26 : !db.bool
            relalg.return %27 : !db.bool
        }
        %29 = relalg.map @map1 %5 (%28: !relalg.tuple) {
            %30 = relalg.getattr %28 @orders::@o_orderpriority : !db.string
            %31 = db.constant ("1-URGENT") :!db.string
            %32 = db.compare eq %30 : !db.string,%31 : !db.string
            %33 = relalg.getattr %28 @orders::@o_orderpriority : !db.string
            %34 = db.constant ("2-HIGH") :!db.string
            %35 = db.compare eq %33 : !db.string,%34 : !db.string
            %36 = db.or %32 : !db.bool,%35 : !db.bool
            %37 = db.if %36 : !db.bool  -> !db.int<64> {
                %38 = db.constant (1) :!db.int<64>
                db.yield %38 : !db.int<64>
            } else {
                %39 = db.constant (0) :!db.int<64>
                db.yield %39 : !db.int<64>
            }
            %40 = relalg.addattr %28, @aggfmname1({type=!db.int<64>}) %37
            %41 = relalg.getattr %28 @orders::@o_orderpriority : !db.string
            %42 = db.constant ("1-URGENT") :!db.string
            %43 = db.compare neq %41 : !db.string,%42 : !db.string
            %44 = relalg.getattr %28 @orders::@o_orderpriority : !db.string
            %45 = db.constant ("2-HIGH") :!db.string
            %46 = db.compare neq %44 : !db.string,%45 : !db.string
            %47 = db.and %43 : !db.bool,%46 : !db.bool
            %48 = db.if %47 : !db.bool  -> !db.int<64> {
                %49 = db.constant (1) :!db.int<64>
                db.yield %49 : !db.int<64>
            } else {
                %50 = db.constant (0) :!db.int<64>
                db.yield %50 : !db.int<64>
            }
            %51 = relalg.addattr %40, @aggfmname3({type=!db.int<64>}) %48
            relalg.return %51 : !relalg.tuple
        }
        %54 = relalg.aggregation @aggr1 %29 [@lineitem::@l_shipmode] (%52 : !relalg.relation, %53 : !relalg.tuple) {
            %55 = relalg.aggrfn sum @map1::@aggfmname1 %52 : !db.int<64>
            %56 = relalg.addattr %53, @aggfmname2({type=!db.int<64>}) %55
            %57 = relalg.aggrfn sum @map1::@aggfmname3 %52 : !db.int<64>
            %58 = relalg.addattr %56, @aggfmname4({type=!db.int<64>}) %57
            relalg.return
        }
        %59 = relalg.sort %54 [(@lineitem::@l_shipmode,asc)]
        %60 = relalg.materialize %59 [@lineitem::@l_shipmode,@aggr1::@aggfmname2,@aggr1::@aggfmname4] => ["l_shipmode","high_line_count","low_line_count"] : !db.table
        return %60 : !db.table
    }
}

