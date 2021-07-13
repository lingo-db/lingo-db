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
            relalg.addattr @aggfmname3({type=!db.decimal<15,2>}) %15
            %16 = relalg.getattr %9 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %17 = db.constant (1) :!db.decimal<15,2>
            %18 = relalg.getattr %9 @lineitem::@l_discount : !db.decimal<15,2>
            %19 = db.sub %17 : !db.decimal<15,2>,%18 : !db.decimal<15,2>
            %20 = db.constant (1) :!db.decimal<15,2>
            %21 = relalg.getattr %9 @lineitem::@l_tax : !db.decimal<15,2>
            %22 = db.add %20 : !db.decimal<15,2>,%21 : !db.decimal<15,2>
            %23 = db.mul %16 : !db.decimal<15,2>,%19 : !db.decimal<15,2>
            %24 = db.mul %22 : !db.decimal<15,2>,%23 : !db.decimal<15,2>
            relalg.addattr @aggfmname5({type=!db.decimal<15,2>}) %24
            relalg.return
        }
        %26 = relalg.aggregation @aggr1 %10 [@lineitem::@l_returnflag,@lineitem::@l_linestatus] (%25 : !relalg.relation) {
            %27 = relalg.aggrfn sum @lineitem::@l_quantity %25 : !db.decimal<15,2>
            relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %27
            %28 = relalg.aggrfn sum @lineitem::@l_extendedprice %25 : !db.decimal<15,2>
            relalg.addattr @aggfmname2({type=!db.decimal<15,2>}) %28
            %29 = relalg.aggrfn sum @map1::@aggfmname3 %25 : !db.decimal<15,2>
            relalg.addattr @aggfmname4({type=!db.decimal<15,2>}) %29
            %30 = relalg.aggrfn sum @map1::@aggfmname5 %25 : !db.decimal<15,2>
            relalg.addattr @aggfmname6({type=!db.decimal<15,2>}) %30
            %31 = relalg.aggrfn avg @lineitem::@l_quantity %25 : !db.decimal<15,2>
            relalg.addattr @aggfmname7({type=!db.decimal<15,2>}) %31
            %32 = relalg.aggrfn avg @lineitem::@l_extendedprice %25 : !db.decimal<15,2>
            relalg.addattr @aggfmname8({type=!db.decimal<15,2>}) %32
            %33 = relalg.aggrfn avg @lineitem::@l_discount %25 : !db.decimal<15,2>
            relalg.addattr @aggfmname9({type=!db.decimal<15,2>}) %33
            %34 = relalg.count %25
            relalg.addattr @aggfmname10({type=!db.int<64>}) %34
            relalg.return
        }
        %35 = relalg.sort %26 [(@lineitem::@l_returnflag,asc),(@lineitem::@l_linestatus,asc)]
        %36 = relalg.materialize %35 [@lineitem::@l_returnflag,@lineitem::@l_linestatus,@aggr1::@aggfmname1,@aggr1::@aggfmname2,@aggr1::@aggfmname4,@aggr1::@aggfmname6,@aggr1::@aggfmname7,@aggr1::@aggfmname8,@aggr1::@aggfmname9,@aggr1::@aggfmname10] => ["l_returnflag","l_linestatus","sum_qty","sum_base_price","sum_disc_price","sum_charge","avg_qty","avg_price","avg_disc","count_order"] : !db.table
        return %36 : !db.table
    }
}

