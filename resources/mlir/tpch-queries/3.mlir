module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @customer { table_identifier="customer", rows=15000 , pkey=["c_custkey"]} columns: {c_custkey => @c_custkey({type=i32}),
            c_name => @c_name({type=!db.string}),
            c_address => @c_address({type=!db.string}),
            c_nationkey => @c_nationkey({type=i32}),
            c_phone => @c_phone({type=!db.string}),
            c_acctbal => @c_acctbal({type=!db.decimal<15,2>}),
            c_mktsegment => @c_mktsegment({type=!db.string}),
            c_comment => @c_comment({type=!db.string})
        }
        %2 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=i32}),
            o_custkey => @o_custkey({type=i32}),
            o_orderstatus => @o_orderstatus({type=!db.char<1>}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=i32}),
            o_comment => @o_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %4 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=i32}),
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
        %5 = relalg.crossproduct %3, %4
        %7 = relalg.selection %5(%6: !relalg.tuple) {
            %8 = relalg.getattr %6 @customer::@c_mktsegment : !db.string
            %9 = db.constant ("BUILDING") :!db.string
            %10 = db.compare eq %8 : !db.string,%9 : !db.string
            %11 = relalg.getattr %6 @customer::@c_custkey : i32
            %12 = relalg.getattr %6 @orders::@o_custkey : i32
            %13 = db.compare eq %11 : i32,%12 : i32
            %14 = relalg.getattr %6 @lineitem::@l_orderkey : i32
            %15 = relalg.getattr %6 @orders::@o_orderkey : i32
            %16 = db.compare eq %14 : i32,%15 : i32
            %17 = relalg.getattr %6 @orders::@o_orderdate : !db.date<day>
            %18 = db.constant ("1995-03-15") :!db.date<day>
            %19 = db.compare lt %17 : !db.date<day>,%18 : !db.date<day>
            %20 = relalg.getattr %6 @lineitem::@l_shipdate : !db.date<day>
            %21 = db.constant ("1995-03-15") :!db.date<day>
            %22 = db.compare gt %20 : !db.date<day>,%21 : !db.date<day>
            %23 = db.and %10 : i1,%13 : i1,%16 : i1,%19 : i1,%22 : i1
            relalg.return %23 : i1
        }
        %25 = relalg.map @map %7 (%24: !relalg.tuple) {
            %26 = relalg.getattr %24 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %27 = db.constant (1) :!db.decimal<15,2>
            %28 = relalg.getattr %24 @lineitem::@l_discount : !db.decimal<15,2>
            %29 = db.sub %27 : !db.decimal<15,2>,%28 : !db.decimal<15,2>
            %30 = db.mul %26 : !db.decimal<15,2>,%29 : !db.decimal<15,2>
            %31 = relalg.addattr %24, @aggfmname1({type=!db.decimal<15,2>}) %30
            relalg.return %31 : !relalg.tuple
        }
        %34 = relalg.aggregation @aggr %25 [@lineitem::@l_orderkey,@orders::@o_orderdate,@orders::@o_shippriority] (%32 : !relalg.tuplestream, %33 : !relalg.tuple) {
            %35 = relalg.aggrfn sum @map::@aggfmname1 %32 : !db.decimal<15,2>
            %36 = relalg.addattr %33, @aggfmname2({type=!db.decimal<15,2>}) %35
            relalg.return %36 : !relalg.tuple
        }
        %37 = relalg.sort %34 [(@aggr::@aggfmname2,desc),(@orders::@o_orderdate,asc)]
        %38 = relalg.limit 10 %37
        %39 = relalg.materialize %38 [@lineitem::@l_orderkey,@aggr::@aggfmname2,@orders::@o_orderdate,@orders::@o_shippriority] => ["l_orderkey","revenue","o_orderdate","o_shippriority"] : !db.table
        return %39 : !db.table
    }
}

