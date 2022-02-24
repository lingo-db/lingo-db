//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                       c_count  |                      custdist  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                             0  |                          5000  |
//CHECK: |                            10  |                           665  |
//CHECK: |                             9  |                           657  |
//CHECK: |                            11  |                           621  |
//CHECK: |                            12  |                           567  |
//CHECK: |                             8  |                           564  |
//CHECK: |                            13  |                           492  |
//CHECK: |                            18  |                           482  |
//CHECK: |                             7  |                           480  |
//CHECK: |                            20  |                           456  |
//CHECK: |                            14  |                           456  |
//CHECK: |                            16  |                           449  |
//CHECK: |                            19  |                           447  |
//CHECK: |                            15  |                           432  |
//CHECK: |                            17  |                           423  |
//CHECK: |                            21  |                           412  |
//CHECK: |                            22  |                           371  |
//CHECK: |                             6  |                           337  |
//CHECK: |                            23  |                           323  |
//CHECK: |                            24  |                           256  |
//CHECK: |                            25  |                           204  |
//CHECK: |                             5  |                           204  |
//CHECK: |                            26  |                           155  |
//CHECK: |                            27  |                           141  |
//CHECK: |                            28  |                            97  |
//CHECK: |                             4  |                            94  |
//CHECK: |                            29  |                            64  |
//CHECK: |                             3  |                            48  |
//CHECK: |                            30  |                            27  |
//CHECK: |                            31  |                            26  |
//CHECK: |                            32  |                            14  |
//CHECK: |                            33  |                            11  |
//CHECK: |                             2  |                            11  |
//CHECK: |                            34  |                             6  |
//CHECK: |                            35  |                             5  |
//CHECK: |                             1  |                             2  |
//CHECK: |                            36  |                             1  |
module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @customer { table_identifier="customer", rows=15000 , pkey=["c_custkey"]} columns: {c_custkey => @c_custkey({type=!db.int<32>}),
            c_name => @c_name({type=!db.string}),
            c_address => @c_address({type=!db.string}),
            c_nationkey => @c_nationkey({type=!db.int<32>}),
            c_phone => @c_phone({type=!db.string}),
            c_acctbal => @c_acctbal({type=!db.decimal<15,2>}),
            c_mktsegment => @c_mktsegment({type=!db.string}),
            c_comment => @c_comment({type=!db.string})
        }
        %2 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=!db.int<32>}),
            o_custkey => @o_custkey({type=!db.int<32>}),
            o_orderstatus => @o_orderstatus({type=!db.char<1>}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=!db.int<32>}),
            o_comment => @o_comment({type=!db.string})
        }
        %4 = relalg.outerjoin @outerjoin %1, %2(%3: !relalg.tuple) {
            %5 = relalg.getattr %3 @customer::@c_custkey : !db.int<32>
            %6 = relalg.getattr %3 @orders::@o_custkey : !db.int<32>
            %7 = db.compare eq %5 : !db.int<32>,%6 : !db.int<32>
            %8 = relalg.getattr %3 @orders::@o_comment : !db.string
            %9 = db.constant ("%special%requests%") :!db.string
            %10 = db.compare like %8 : !db.string,%9 : !db.string
            %11 = db.not %10 : i1
            %12 = db.and %7 : i1,%11 : i1
            relalg.return %12 : i1
        } mapping: {@o_orderkey({type=!db.nullable<!db.int<32>>})=[@orders::@o_orderkey],@o_custkey({type=!db.nullable<!db.int<32>>})=[@orders::@o_custkey],@o_orderstatus({type=!db.nullable<!db.char<1>>})=[@orders::@o_orderstatus],@o_totalprice({type=!db.nullable<!db.decimal<15,2>>})=[@orders::@o_totalprice],@o_orderdate({type=!db.nullable<!db.date<day>>})=[@orders::@o_orderdate],@o_orderpriority({type=!db.nullable<!db.string>})=[@orders::@o_orderpriority],@o_clerk({type=!db.nullable<!db.string>})=[@orders::@o_clerk],@o_shippriority({type=!db.nullable<!db.int<32>>})=[@orders::@o_shippriority],@o_comment({type=!db.nullable<!db.string>})=[@orders::@o_comment]}
        %15 = relalg.aggregation @aggr %4 [@customer::@c_custkey] (%13 : !relalg.tuplestream, %14 : !relalg.tuple) {
            %16 = relalg.aggrfn count @outerjoin::@o_orderkey %13 : !db.int<64>
            %17 = relalg.addattr %14, @aggfmname1({type=!db.int<64>}) %16
            relalg.return %17 : !relalg.tuple
        }
        %20 = relalg.aggregation @aggr1 %15 [@aggr::@aggfmname1] (%18 : !relalg.tuplestream, %19 : !relalg.tuple) {
            %21 = relalg.count %18
            %22 = relalg.addattr %19, @aggfmname1({type=!db.int<64>}) %21
            relalg.return %22 : !relalg.tuple
        }
        %23 = relalg.sort %20 [(@aggr1::@aggfmname1,desc),(@aggr::@aggfmname1,desc)]
        %24 = relalg.materialize %23 [@aggr::@aggfmname1,@aggr1::@aggfmname1] => ["c_count","custdist"] : !db.table
        return %24 : !db.table
    }
}


