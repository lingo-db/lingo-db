module @querymodule{
    func @main (%executionContext: !util.generic_memref<i8>)  -> !db.table{
        %1 = relalg.basetable @customer { table_identifier="customer", rows=15000 , pkey=["c_custkey"]} columns: {c_custkey => @c_custkey({type=!db.int<64>}),
            c_name => @c_name({type=!db.string}),
            c_address => @c_address({type=!db.string}),
            c_nationkey => @c_nationkey({type=!db.int<64>}),
            c_phone => @c_phone({type=!db.string}),
            c_acctbal => @c_acctbal({type=!db.decimal<15,2>}),
            c_mktsegment => @c_mktsegment({type=!db.string}),
            c_comment => @c_comment({type=!db.string})
        }
        %2 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=!db.int<64>}),
            o_custkey => @o_custkey({type=!db.int<64>}),
            o_orderstatus => @o_orderstatus({type=!db.string}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=!db.int<32>}),
            o_comment => @o_comment({type=!db.string})
        }
        %4 = relalg.outerjoin @outerjoin1 %1, %2(%3: !relalg.tuple) {
            %5 = relalg.getattr %3 @customer::@c_custkey : !db.int<64>
            %6 = relalg.getattr %3 @orders::@o_custkey : !db.int<64>
            %7 = db.compare eq %5 : !db.int<64>,%6 : !db.int<64>
            %8 = relalg.getattr %3 @orders::@o_comment : !db.string
            %9 = db.constant ("%special%requests%") :!db.string
            %10 = db.compare like %8 : !db.string,%9 : !db.string
            %11 = db.not %10 : !db.bool
            %12 = db.and %7 : !db.bool,%11 : !db.bool
            relalg.return %12 : !db.bool
        } mapping: {@o_orderkey({type=!db.int<64,nullable>})=[@orders::@o_orderkey],@o_custkey({type=!db.int<64,nullable>})=[@orders::@o_custkey],@o_orderstatus({type=!db.string<nullable>})=[@orders::@o_orderstatus],@o_totalprice({type=!db.decimal<15,2,nullable>})=[@orders::@o_totalprice],@o_orderdate({type=!db.date<day,nullable>})=[@orders::@o_orderdate],@o_orderpriority({type=!db.string<nullable>})=[@orders::@o_orderpriority],@o_clerk({type=!db.string<nullable>})=[@orders::@o_clerk],@o_shippriority({type=!db.int<32,nullable>})=[@orders::@o_shippriority],@o_comment({type=!db.string<nullable>})=[@orders::@o_comment]}
        %15 = relalg.aggregation @aggr1 %4 [@customer::@c_custkey] (%13 : !relalg.relation, %14 : !relalg.tuple) {
            %16 = relalg.aggrfn count @outerjoin1::@o_orderkey %13 : !db.int<64>
            %17 = relalg.addattr %14, @aggfmname1({type=!db.int<64>}) %16
            relalg.return %17 : !relalg.tuple
        }
        %20 = relalg.aggregation @aggr2 %15 [@aggr1::@aggfmname1] (%18 : !relalg.relation, %19 : !relalg.tuple) {
            %21 = relalg.count %18
            %22 = relalg.addattr %19, @aggfmname1({type=!db.int<64>}) %21
            relalg.return %22 : !relalg.tuple
        }
        %23 = relalg.sort %20 [(@aggr2::@aggfmname1,desc),(@aggr1::@aggfmname1,desc)]
        %24 = relalg.materialize %23 [@aggr1::@aggfmname1,@aggr2::@aggfmname1] => ["c_count","custdist"] : !db.table
        return %24 : !db.table
    }
}

