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
        %4 = relalg.outerjoin @outerjoin %1, %2(%3: !relalg.tuple) {
            %5 = relalg.getattr %3 @customer::@c_custkey : i32
            %6 = relalg.getattr %3 @orders::@o_custkey : i32
            %7 = db.compare eq %5 : i32,%6 : i32
            %8 = relalg.getattr %3 @orders::@o_comment : !db.string
            %9 = db.constant ("%special%requests%") :!db.string
            %10 = db.compare like %8 : !db.string,%9 : !db.string
            %11 = db.not %10 : i1
            %12 = db.and %7 : i1,%11 : i1
            relalg.return %12 : i1
        } mapping: {@o_orderkey({type=!db.nullable<i32>})=[@orders::@o_orderkey],@o_custkey({type=!db.nullable<i32>})=[@orders::@o_custkey],@o_orderstatus({type=!db.nullable<!db.char<1>>})=[@orders::@o_orderstatus],@o_totalprice({type=!db.nullable<!db.decimal<15,2>>})=[@orders::@o_totalprice],@o_orderdate({type=!db.nullable<!db.date<day>>})=[@orders::@o_orderdate],@o_orderpriority({type=!db.nullable<!db.string>})=[@orders::@o_orderpriority],@o_clerk({type=!db.nullable<!db.string>})=[@orders::@o_clerk],@o_shippriority({type=!db.nullable<i32>})=[@orders::@o_shippriority],@o_comment({type=!db.nullable<!db.string>})=[@orders::@o_comment]}
        %15 = relalg.aggregation @aggr %4 [@customer::@c_custkey] (%13 : !relalg.tuplestream, %14 : !relalg.tuple) {
            %16 = relalg.aggrfn count @outerjoin::@o_orderkey %13 : i64
            %17 = relalg.addattr %14, @aggfmname1({type=i64}) %16
            relalg.return %17 : !relalg.tuple
        }
        %20 = relalg.aggregation @aggr1 %15 [@aggr::@aggfmname1] (%18 : !relalg.tuplestream, %19 : !relalg.tuple) {
            %21 = relalg.count %18
            %22 = relalg.addattr %19, @aggfmname1({type=i64}) %21
            relalg.return %22 : !relalg.tuple
        }
        %23 = relalg.sort %20 [(@aggr1::@aggfmname1,desc),(@aggr::@aggfmname1,desc)]
        %24 = relalg.materialize %23 [@aggr::@aggfmname1,@aggr1::@aggfmname1] => ["c_count","custdist"] : !db.table
        return %24 : !db.table
    }
}

