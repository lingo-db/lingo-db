module @querymodule{
    func @main (%executionContext: !util.generic_memref<i8>)  -> !db.table{
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
        %4 = relalg.selection %2(%3: !relalg.tuple) {
            %5 = relalg.getattr %3 @lineitem::@l_shipdate : !db.date<day>
            %6 = db.constant ("1996-01-01") :!db.date<day>
            %7 = db.compare gte %5 : !db.date<day>,%6 : !db.date<day>
            %8 = relalg.getattr %3 @lineitem::@l_shipdate : !db.date<day>
            %9 = db.constant ("1996-04-01") :!db.date<day>
            %10 = db.compare lt %8 : !db.date<day>,%9 : !db.date<day>
            %11 = db.and %7 : !db.bool,%10 : !db.bool
            relalg.return %11 : !db.bool
        }
        %13 = relalg.map @map1 %4 (%12: !relalg.tuple) {
            %14 = relalg.getattr %12 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %15 = db.constant (1) :!db.decimal<15,2>
            %16 = relalg.getattr %12 @lineitem::@l_discount : !db.decimal<15,2>
            %17 = db.sub %15 : !db.decimal<15,2>,%16 : !db.decimal<15,2>
            %18 = db.mul %14 : !db.decimal<15,2>,%17 : !db.decimal<15,2>
            relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %18
            relalg.return
        }
        %20 = relalg.aggregation @aggr1 %13 [@lineitem::@l_suppkey] (%19 : !relalg.relation) {
            %21 = relalg.aggrfn sum @map1::@aggfmname1 %19 : !db.decimal<15,2>
            relalg.addattr @aggfmname2({type=!db.decimal<15,2>}) %21
            relalg.return
        }
        %22 = relalg.crossproduct %1, %20
        %24 = relalg.selection %22(%23: !relalg.tuple) {
            %25 = relalg.getattr %23 @supplier::@s_suppkey : !db.int<64>
            %26 = relalg.getattr %23 @lineitem::@l_suppkey : !db.int<64>
            %27 = db.compare eq %25 : !db.int<64>,%26 : !db.int<64>
            %28 = relalg.getattr %23 @aggr1::@aggfmname2 : !db.decimal<15,2>
            %29 = relalg.basetable @lineitem1 { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
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
            %31 = relalg.selection %29(%30: !relalg.tuple) {
                %32 = relalg.getattr %30 @lineitem1::@l_shipdate : !db.date<day>
                %33 = db.constant ("1996-01-01") :!db.date<day>
                %34 = db.compare gte %32 : !db.date<day>,%33 : !db.date<day>
                %35 = relalg.getattr %30 @lineitem1::@l_shipdate : !db.date<day>
                %36 = db.constant ("1996-04-01") :!db.date<day>
                %37 = db.compare lt %35 : !db.date<day>,%36 : !db.date<day>
                %38 = db.and %34 : !db.bool,%37 : !db.bool
                relalg.return %38 : !db.bool
            }
            %40 = relalg.map @map3 %31 (%39: !relalg.tuple) {
                %41 = relalg.getattr %39 @lineitem1::@l_extendedprice : !db.decimal<15,2>
                %42 = db.constant (1) :!db.decimal<15,2>
                %43 = relalg.getattr %39 @lineitem1::@l_discount : !db.decimal<15,2>
                %44 = db.sub %42 : !db.decimal<15,2>,%43 : !db.decimal<15,2>
                %45 = db.mul %41 : !db.decimal<15,2>,%44 : !db.decimal<15,2>
                relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %45
                relalg.return
            }
            %47 = relalg.aggregation @aggr2 %40 [@lineitem1::@l_suppkey] (%46 : !relalg.relation) {
                %48 = relalg.aggrfn sum @map3::@aggfmname1 %46 : !db.decimal<15,2>
                relalg.addattr @aggfmname2({type=!db.decimal<15,2>}) %48
                relalg.return
            }
            %50 = relalg.aggregation @aggr3 %47 [] (%49 : !relalg.relation) {
                %51 = relalg.aggrfn max @aggr2::@aggfmname2 %49 : !db.decimal<15,2,nullable>
                relalg.addattr @aggfmname1({type=!db.decimal<15,2,nullable>}) %51
                relalg.return
            }
            %52 = relalg.getscalar @aggr3::@aggfmname1 %50 : !db.decimal<15,2,nullable>
            %53 = db.compare eq %28 : !db.decimal<15,2>,%52 : !db.decimal<15,2,nullable>
            %54 = db.and %27 : !db.bool,%53 : !db.bool<nullable>
            relalg.return %54 : !db.bool<nullable>
        }
        %55 = relalg.sort %24 [(@supplier::@s_suppkey,asc)]
        %56 = relalg.materialize %55 [@supplier::@s_suppkey,@supplier::@s_name,@supplier::@s_address,@supplier::@s_phone,@aggr1::@aggfmname2] => ["s_suppkey","s_name","s_address","s_phone","total_revenue"] : !db.table
        return %56 : !db.table
    }
}

