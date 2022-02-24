//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                       revenue  |
//CHECK: ----------------------------------
//CHECK: |                   11803366.91  |
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
        %3 = relalg.selection %1(%2: !relalg.tuple) {
            %4 = relalg.getattr %2 @lineitem::@l_shipdate : !db.date<day>
            %5 = db.constant ("1994-01-01") :!db.date<day>
            %6 = db.compare gte %4 : !db.date<day>,%5 : !db.date<day>
            %7 = relalg.getattr %2 @lineitem::@l_shipdate : !db.date<day>
            %8 = db.constant ("1995-01-01") :!db.date<day>
            %9 = db.compare lt %7 : !db.date<day>,%8 : !db.date<day>
            %10 = relalg.getattr %2 @lineitem::@l_discount : !db.decimal<15,2>
            %11 = db.constant ("0.06") :!db.decimal<15,2>
            %12 = db.constant ("0.01") :!db.decimal<15,2>
            %13 = db.sub %11 : !db.decimal<15,2>,%12 : !db.decimal<15,2>
            %14 = db.constant ("0.06") :!db.decimal<15,2>
            %15 = db.constant ("0.01") :!db.decimal<15,2>
            %16 = db.add %14 : !db.decimal<15,2>,%15 : !db.decimal<15,2>
            %17 = db.compare gte %10 : !db.decimal<15,2>,%13 : !db.decimal<15,2>
            %18 = db.compare lte %10 : !db.decimal<15,2>,%16 : !db.decimal<15,2>
            %19 = db.and %17 : i1,%18 : i1
            %20 = relalg.getattr %2 @lineitem::@l_quantity : !db.decimal<15,2>
            %21 = db.constant (24) :!db.decimal<15,2>
            %22 = db.compare lt %20 : !db.decimal<15,2>,%21 : !db.decimal<15,2>
            %23 = db.and %6 : i1,%9 : i1,%19 : i1,%22 : i1
            relalg.return %23 : i1
        }
        %25 = relalg.map @map %3 (%24: !relalg.tuple) {
            %26 = relalg.getattr %24 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %27 = relalg.getattr %24 @lineitem::@l_discount : !db.decimal<15,2>
            %28 = db.mul %26 : !db.decimal<15,2>,%27 : !db.decimal<15,2>
            %29 = relalg.addattr %24, @aggfmname1({type=!db.decimal<15,2>}) %28
            relalg.return %29 : !relalg.tuple
        }
        %32 = relalg.aggregation @aggr %25 [] (%30 : !relalg.tuplestream, %31 : !relalg.tuple) {
            %33 = relalg.aggrfn sum @map::@aggfmname1 %30 : !db.nullable<!db.decimal<15,2>>
            %34 = relalg.addattr %31, @aggfmname2({type=!db.nullable<!db.decimal<15,2>>}) %33
            relalg.return %34 : !relalg.tuple
        }
        %35 = relalg.materialize %32 [@aggr::@aggfmname2] => ["revenue"] : !db.table
        return %35 : !db.table
    }
}


