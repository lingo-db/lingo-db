module @querymodule{
    func @main (%executionContext: !util.generic_memref<i8>)  -> !db.table{
        %1 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
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
        %2 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=!db.int<64>}),
            p_name => @p_name({type=!db.string}),
            p_mfgr => @p_mfgr({type=!db.string}),
            p_brand => @p_brand({type=!db.string}),
            p_type => @p_type({type=!db.string}),
            p_size => @p_size({type=!db.int<32>}),
            p_container => @p_container({type=!db.string}),
            p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
            p_comment => @p_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %5 = relalg.selection %3(%4: !relalg.tuple) {
            %6 = relalg.getattr %4 @lineitem::@l_partkey : !db.int<64>
            %7 = relalg.getattr %4 @part::@p_partkey : !db.int<64>
            %8 = db.compare eq %6 : !db.int<64>,%7 : !db.int<64>
            %9 = relalg.getattr %4 @lineitem::@l_shipdate : !db.date<day>
            %10 = db.constant ("1995-09-01") :!db.date<day>
            %11 = db.compare gte %9 : !db.date<day>,%10 : !db.date<day>
            %12 = relalg.getattr %4 @lineitem::@l_shipdate : !db.date<day>
            %13 = db.constant ("1995-10-01") :!db.date<day>
            %14 = db.compare lt %12 : !db.date<day>,%13 : !db.date<day>
            %15 = db.and %8 : !db.bool,%11 : !db.bool,%14 : !db.bool
            relalg.return %15 : !db.bool
        }
        %17 = relalg.map @map1 %5 (%16: !relalg.tuple) {
            %18 = relalg.getattr %16 @part::@p_type : !db.string
            %19 = db.constant ("PROMO%") :!db.string
            %20 = db.compare like %18 : !db.string,%19 : !db.string
            %21 = db.if %20 : !db.bool  -> !db.decimal<15,2> {
                %22 = relalg.getattr %16 @lineitem::@l_extendedprice : !db.decimal<15,2>
                %23 = db.constant (1) :!db.decimal<15,2>
                %24 = relalg.getattr %16 @lineitem::@l_discount : !db.decimal<15,2>
                %25 = db.sub %23 : !db.decimal<15,2>,%24 : !db.decimal<15,2>
                %26 = db.mul %22 : !db.decimal<15,2>,%25 : !db.decimal<15,2>
                db.yield %26 : !db.decimal<15,2>
            } else {
                %27 = db.constant (0) :!db.decimal<15,2>
                db.yield %27 : !db.decimal<15,2>
            }
            relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %21
            %28 = relalg.getattr %16 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %29 = db.constant (1) :!db.decimal<15,2>
            %30 = relalg.getattr %16 @lineitem::@l_discount : !db.decimal<15,2>
            %31 = db.sub %29 : !db.decimal<15,2>,%30 : !db.decimal<15,2>
            %32 = db.mul %28 : !db.decimal<15,2>,%31 : !db.decimal<15,2>
            relalg.addattr @aggfmname3({type=!db.decimal<15,2>}) %32
            relalg.return
        }
        %34 = relalg.aggregation @aggr1 %17 [] (%33 : !relalg.relation) {
            %35 = relalg.aggrfn sum @map1::@aggfmname1 %33 : !db.decimal<15,2,nullable>
            relalg.addattr @aggfmname2({type=!db.decimal<15,2,nullable>}) %35
            %36 = relalg.aggrfn sum @map1::@aggfmname3 %33 : !db.decimal<15,2,nullable>
            relalg.addattr @aggfmname4({type=!db.decimal<15,2,nullable>}) %36
            relalg.return
        }
        %38 = relalg.map @map2 %34 (%37: !relalg.tuple) {
            %39 = db.constant ("100.0") :!db.decimal<15,2>
            %40 = relalg.getattr %37 @aggr1::@aggfmname2 : !db.decimal<15,2,nullable>
            %41 = db.mul %39 : !db.decimal<15,2>,%40 : !db.decimal<15,2,nullable>
            %42 = relalg.getattr %37 @aggr1::@aggfmname4 : !db.decimal<15,2,nullable>
            %43 = db.div %41 : !db.decimal<15,2,nullable>,%42 : !db.decimal<15,2,nullable>
            relalg.addattr @aggfmname5({type=!db.decimal<15,2,nullable>}) %43
            relalg.return
        }
        %44 = relalg.materialize %38 [@map2::@aggfmname5] => ["promo_revenue"] : !db.table
        return %44 : !db.table
    }
}

