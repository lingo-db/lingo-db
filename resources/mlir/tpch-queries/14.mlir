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
            %28 = relalg.addattr %16, @aggfmname1({type=!db.decimal<15,2>}) %21
            %29 = relalg.getattr %16 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %30 = db.constant (1) :!db.decimal<15,2>
            %31 = relalg.getattr %16 @lineitem::@l_discount : !db.decimal<15,2>
            %32 = db.sub %30 : !db.decimal<15,2>,%31 : !db.decimal<15,2>
            %33 = db.mul %29 : !db.decimal<15,2>,%32 : !db.decimal<15,2>
            %34 = relalg.addattr %28, @aggfmname3({type=!db.decimal<15,2>}) %33
            relalg.return %34 : !relalg.tuple
        }
        %37 = relalg.aggregation @aggr1 %17 [] (%35 : !relalg.relation, %36 : !relalg.tuple) {
            %38 = relalg.aggrfn sum @map1::@aggfmname1 %35 : !db.decimal<15,2,nullable>
            %39 = relalg.addattr %36, @aggfmname2({type=!db.decimal<15,2,nullable>}) %38
            %40 = relalg.aggrfn sum @map1::@aggfmname3 %35 : !db.decimal<15,2,nullable>
            %41 = relalg.addattr %39, @aggfmname4({type=!db.decimal<15,2,nullable>}) %40
            relalg.return %41 : !relalg.tuple
        }
        %43 = relalg.map @map2 %37 (%42: !relalg.tuple) {
            %44 = db.constant ("100.0") :!db.decimal<15,2>
            %45 = relalg.getattr %42 @aggr1::@aggfmname2 : !db.decimal<15,2,nullable>
            %46 = db.mul %44 : !db.decimal<15,2>,%45 : !db.decimal<15,2,nullable>
            %47 = relalg.getattr %42 @aggr1::@aggfmname4 : !db.decimal<15,2,nullable>
            %48 = db.div %46 : !db.decimal<15,2,nullable>,%47 : !db.decimal<15,2,nullable>
            %49 = relalg.addattr %42, @aggfmname5({type=!db.decimal<15,2,nullable>}) %48
            relalg.return %49 : !relalg.tuple
        }
        %50 = relalg.materialize %43 [@map2::@aggfmname5] => ["promo_revenue"] : !db.table
        return %50 : !db.table
    }
}

