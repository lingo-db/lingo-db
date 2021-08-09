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
            %19 = relalg.addattr %12, @aggfmname1({type=!db.decimal<15,2>}) %18
            relalg.return %19 : !relalg.tuple
        }
        %22 = relalg.aggregation @aggr1 %13 [@lineitem::@l_suppkey] (%20 : !relalg.relation, %21 : !relalg.tuple) {
            %23 = relalg.aggrfn sum @map1::@aggfmname1 %20 : !db.decimal<15,2>
            %24 = relalg.addattr %21, @aggfmname2({type=!db.decimal<15,2>}) %23
            relalg.return
        }
        %25 = relalg.crossproduct %1, %22
        %27 = relalg.selection %25(%26: !relalg.tuple) {
            %28 = relalg.getattr %26 @supplier::@s_suppkey : !db.int<64>
            %29 = relalg.getattr %26 @lineitem::@l_suppkey : !db.int<64>
            %30 = db.compare eq %28 : !db.int<64>,%29 : !db.int<64>
            %31 = relalg.getattr %26 @aggr1::@aggfmname2 : !db.decimal<15,2>
            %32 = relalg.basetable @lineitem1 { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
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
            %34 = relalg.selection %32(%33: !relalg.tuple) {
                %35 = relalg.getattr %33 @lineitem1::@l_shipdate : !db.date<day>
                %36 = db.constant ("1996-01-01") :!db.date<day>
                %37 = db.compare gte %35 : !db.date<day>,%36 : !db.date<day>
                %38 = relalg.getattr %33 @lineitem1::@l_shipdate : !db.date<day>
                %39 = db.constant ("1996-04-01") :!db.date<day>
                %40 = db.compare lt %38 : !db.date<day>,%39 : !db.date<day>
                %41 = db.and %37 : !db.bool,%40 : !db.bool
                relalg.return %41 : !db.bool
            }
            %43 = relalg.map @map3 %34 (%42: !relalg.tuple) {
                %44 = relalg.getattr %42 @lineitem1::@l_extendedprice : !db.decimal<15,2>
                %45 = db.constant (1) :!db.decimal<15,2>
                %46 = relalg.getattr %42 @lineitem1::@l_discount : !db.decimal<15,2>
                %47 = db.sub %45 : !db.decimal<15,2>,%46 : !db.decimal<15,2>
                %48 = db.mul %44 : !db.decimal<15,2>,%47 : !db.decimal<15,2>
                %49 = relalg.addattr %42, @aggfmname1({type=!db.decimal<15,2>}) %48
                relalg.return %49 : !relalg.tuple
            }
            %52 = relalg.aggregation @aggr2 %43 [@lineitem1::@l_suppkey] (%50 : !relalg.relation, %51 : !relalg.tuple) {
                %53 = relalg.aggrfn sum @map3::@aggfmname1 %50 : !db.decimal<15,2>
                %54 = relalg.addattr %51, @aggfmname2({type=!db.decimal<15,2>}) %53
                relalg.return
            }
            %57 = relalg.aggregation @aggr3 %52 [] (%55 : !relalg.relation, %56 : !relalg.tuple) {
                %58 = relalg.aggrfn max @aggr2::@aggfmname2 %55 : !db.decimal<15,2,nullable>
                %59 = relalg.addattr %56, @aggfmname1({type=!db.decimal<15,2,nullable>}) %58
                relalg.return
            }
            %60 = relalg.getscalar @aggr3::@aggfmname1 %57 : !db.decimal<15,2,nullable>
            %61 = db.compare eq %31 : !db.decimal<15,2>,%60 : !db.decimal<15,2,nullable>
            %62 = db.and %30 : !db.bool,%61 : !db.bool<nullable>
            relalg.return %62 : !db.bool<nullable>
        }
        %63 = relalg.sort %27 [(@supplier::@s_suppkey,asc)]
        %64 = relalg.materialize %63 [@supplier::@s_suppkey,@supplier::@s_name,@supplier::@s_address,@supplier::@s_phone,@aggr1::@aggfmname2] => ["s_suppkey","s_name","s_address","s_phone","total_revenue"] : !db.table
        return %64 : !db.table
    }
}

