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
        %3 = relalg.selection %1(%2: !relalg.tuple) {
            %4 = relalg.getattr %2 @lineitem::@l_shipdate : !db.date<day>
            %5 = db.constant ("1998-12-01") :!db.date<day>
            %6 = db.constant ("7776000000") :!db.interval<daytime>
            %7 = db.date_sub %5 : !db.date<day>,%6 : !db.interval<daytime>
            %8 = db.compare lte %4 : !db.date<day>,%7 : !db.date<day>
            relalg.return %8 : !db.bool
        }
        %10 = relalg.map @map1 %3 (%9: !relalg.tuple) {
            %11 = relalg.getattr %9 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %12 = db.constant (1) :!db.decimal<15,2>
            %13 = relalg.getattr %9 @lineitem::@l_discount : !db.decimal<15,2>
            %14 = db.sub %12 : !db.decimal<15,2>,%13 : !db.decimal<15,2>
            %15 = db.mul %11 : !db.decimal<15,2>,%14 : !db.decimal<15,2>
            %16 = relalg.addattr %9, @aggfmname3({type=!db.decimal<15,2>}) %15
            %17 = relalg.getattr %9 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %18 = db.constant (1) :!db.decimal<15,2>
            %19 = relalg.getattr %9 @lineitem::@l_discount : !db.decimal<15,2>
            %20 = db.sub %18 : !db.decimal<15,2>,%19 : !db.decimal<15,2>
            %21 = db.constant (1) :!db.decimal<15,2>
            %22 = relalg.getattr %9 @lineitem::@l_tax : !db.decimal<15,2>
            %23 = db.add %21 : !db.decimal<15,2>,%22 : !db.decimal<15,2>
            %24 = db.mul %17 : !db.decimal<15,2>,%20 : !db.decimal<15,2>
            %25 = db.mul %23 : !db.decimal<15,2>,%24 : !db.decimal<15,2>
            %26 = relalg.addattr %16, @aggfmname5({type=!db.decimal<15,2>}) %25
            relalg.return %26 : !relalg.tuple
        }
        %29 = relalg.aggregation @aggr1 %10 [@lineitem::@l_returnflag,@lineitem::@l_linestatus] (%27 : !relalg.relation, %28 : !relalg.tuple) {
            %30 = relalg.aggrfn sum @lineitem::@l_quantity %27 : !db.decimal<15,2>
            %31 = relalg.addattr %28, @aggfmname1({type=!db.decimal<15,2>}) %30
            %32 = relalg.aggrfn sum @lineitem::@l_extendedprice %27 : !db.decimal<15,2>
            %33 = relalg.addattr %31, @aggfmname2({type=!db.decimal<15,2>}) %32
            %34 = relalg.aggrfn sum @map1::@aggfmname3 %27 : !db.decimal<15,2>
            %35 = relalg.addattr %33, @aggfmname4({type=!db.decimal<15,2>}) %34
            %36 = relalg.aggrfn sum @map1::@aggfmname5 %27 : !db.decimal<15,2>
            %37 = relalg.addattr %35, @aggfmname6({type=!db.decimal<15,2>}) %36
            %38 = relalg.aggrfn avg @lineitem::@l_quantity %27 : !db.decimal<15,2>
            %39 = relalg.addattr %37, @aggfmname7({type=!db.decimal<15,2>}) %38
            %40 = relalg.aggrfn avg @lineitem::@l_extendedprice %27 : !db.decimal<15,2>
            %41 = relalg.addattr %39, @aggfmname8({type=!db.decimal<15,2>}) %40
            %42 = relalg.aggrfn avg @lineitem::@l_discount %27 : !db.decimal<15,2>
            %43 = relalg.addattr %41, @aggfmname9({type=!db.decimal<15,2>}) %42
            %44 = relalg.count %27
            %45 = relalg.addattr %43, @aggfmname10({type=!db.int<64>}) %44
            relalg.return
        }
        %46 = relalg.sort %29 [(@lineitem::@l_returnflag,asc),(@lineitem::@l_linestatus,asc)]
        %47 = relalg.materialize %46 [@lineitem::@l_returnflag,@lineitem::@l_linestatus,@aggr1::@aggfmname1,@aggr1::@aggfmname2,@aggr1::@aggfmname4,@aggr1::@aggfmname6,@aggr1::@aggfmname7,@aggr1::@aggfmname8,@aggr1::@aggfmname9,@aggr1::@aggfmname10] => ["l_returnflag","l_linestatus","sum_qty","sum_base_price","sum_disc_price","sum_charge","avg_qty","avg_price","avg_disc","count_order"] : !db.table
        return %47 : !db.table
    }
}

