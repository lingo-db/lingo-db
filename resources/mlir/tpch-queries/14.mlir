module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=i32}),
            l_partkey => @l_partkey({type=i32}),
            l_suppkey => @l_suppkey({type=i32}),
            l_linenumber => @l_linenumber({type=i32}),
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
        %2 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=i32}),
            p_name => @p_name({type=!db.string}),
            p_mfgr => @p_mfgr({type=!db.string}),
            p_brand => @p_brand({type=!db.string}),
            p_type => @p_type({type=!db.string}),
            p_size => @p_size({type=i32}),
            p_container => @p_container({type=!db.string}),
            p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
            p_comment => @p_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %5 = relalg.selection %3(%4: !relalg.tuple) {
            %6 = relalg.getattr %4 @lineitem::@l_partkey : i32
            %7 = relalg.getattr %4 @part::@p_partkey : i32
            %8 = db.compare eq %6 : i32,%7 : i32
            %9 = relalg.getattr %4 @lineitem::@l_shipdate : !db.date<day>
            %10 = db.constant ("1995-09-01") :!db.date<day>
            %11 = db.compare gte %9 : !db.date<day>,%10 : !db.date<day>
            %12 = relalg.getattr %4 @lineitem::@l_shipdate : !db.date<day>
            %13 = db.constant ("1995-10-01") :!db.date<day>
            %14 = db.compare lt %12 : !db.date<day>,%13 : !db.date<day>
            %15 = db.and %8 : i1,%11 : i1,%14 : i1
            relalg.return %15 : i1
        }
        %17 = relalg.map @map %5 (%16: !relalg.tuple) {
            %18 = relalg.getattr %16 @part::@p_type : !db.string
            %19 = db.constant ("PROMO%") :!db.string
            %20 = db.compare like %18 : !db.string,%19 : !db.string
            %21 = db.derive_truth %20 : i1
            %29 = scf.if %21  -> (!db.decimal<15,2>) {
                %23 = relalg.getattr %16 @lineitem::@l_extendedprice : !db.decimal<15,2>
                %24 = db.constant (1) :!db.decimal<15,2>
                %25 = relalg.getattr %16 @lineitem::@l_discount : !db.decimal<15,2>
                %26 = db.sub %24 : !db.decimal<15,2>,%25 : !db.decimal<15,2>
                %27 = db.mul %23 : !db.decimal<15,2>,%26 : !db.decimal<15,2>
                scf.yield %27 : !db.decimal<15,2>
            } else {
                %28 = db.constant (0) :!db.decimal<15,2>
                scf.yield %28 : !db.decimal<15,2>
            }
            %30 = relalg.addattr %16, @aggfmname1({type=!db.decimal<15,2>}) %29
            %31 = relalg.getattr %16 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %32 = db.constant (1) :!db.decimal<15,2>
            %33 = relalg.getattr %16 @lineitem::@l_discount : !db.decimal<15,2>
            %34 = db.sub %32 : !db.decimal<15,2>,%33 : !db.decimal<15,2>
            %35 = db.mul %31 : !db.decimal<15,2>,%34 : !db.decimal<15,2>
            %36 = relalg.addattr %30, @aggfmname3({type=!db.decimal<15,2>}) %35
            relalg.return %36 : !relalg.tuple
        }
        %39 = relalg.aggregation @aggr %17 [] (%37 : !relalg.tuplestream, %38 : !relalg.tuple) {
            %40 = relalg.aggrfn sum @map::@aggfmname1 %37 : !db.nullable<!db.decimal<15,2>>
            %41 = relalg.addattr %38, @aggfmname2({type=!db.nullable<!db.decimal<15,2>>}) %40
            %42 = relalg.aggrfn sum @map::@aggfmname3 %37 : !db.nullable<!db.decimal<15,2>>
            %43 = relalg.addattr %41, @aggfmname4({type=!db.nullable<!db.decimal<15,2>>}) %42
            relalg.return %43 : !relalg.tuple
        }
        %45 = relalg.map @map1 %39 (%44: !relalg.tuple) {
            %46 = db.constant ("100.0") :!db.decimal<15,2>
            %47 = relalg.getattr %44 @aggr::@aggfmname2 : !db.nullable<!db.decimal<15,2>>
            %48 = db.mul %46 : !db.decimal<15,2>,%47 : !db.nullable<!db.decimal<15,2>>
            %49 = relalg.getattr %44 @aggr::@aggfmname4 : !db.nullable<!db.decimal<15,2>>
            %50 = db.div %48 : !db.nullable<!db.decimal<15,2>>,%49 : !db.nullable<!db.decimal<15,2>>
            %51 = relalg.addattr %44, @aggfmname5({type=!db.nullable<!db.decimal<15,2>>}) %50
            relalg.return %51 : !relalg.tuple
        }
        %52 = relalg.materialize %45 [@map1::@aggfmname5] => ["promo_revenue"] : !db.table
        return %52 : !db.table
    }
}

