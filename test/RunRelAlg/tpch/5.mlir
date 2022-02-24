//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                        n_name  |                       revenue  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                       "CHINA"  |                    7822102.06  |
//CHECK: |                       "INDIA"  |                    6376120.75  |
//CHECK: |                       "JAPAN"  |                    6000076.69  |
//CHECK: |                   "INDONESIA"  |                    5580474.62  |
//CHECK: |                     "VIETNAM"  |                    4497839.90  |
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
        %6 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=i32}),
            s_name => @s_name({type=!db.string}),
            s_address => @s_address({type=!db.string}),
            s_nationkey => @s_nationkey({type=i32}),
            s_phone => @s_phone({type=!db.string}),
            s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
            s_comment => @s_comment({type=!db.string})
        }
        %7 = relalg.crossproduct %5, %6
        %8 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=i32}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=i32}),
            n_comment => @n_comment({type=!db.nullable<!db.string>})
        }
        %9 = relalg.crossproduct %7, %8
        %10 = relalg.basetable @region { table_identifier="region", rows=5 , pkey=["r_regionkey"]} columns: {r_regionkey => @r_regionkey({type=i32}),
            r_name => @r_name({type=!db.string}),
            r_comment => @r_comment({type=!db.nullable<!db.string>})
        }
        %11 = relalg.crossproduct %9, %10
        %13 = relalg.selection %11(%12: !relalg.tuple) {
            %14 = relalg.getattr %12 @customer::@c_custkey : i32
            %15 = relalg.getattr %12 @orders::@o_custkey : i32
            %16 = db.compare eq %14 : i32,%15 : i32
            %17 = relalg.getattr %12 @lineitem::@l_orderkey : i32
            %18 = relalg.getattr %12 @orders::@o_orderkey : i32
            %19 = db.compare eq %17 : i32,%18 : i32
            %20 = relalg.getattr %12 @lineitem::@l_suppkey : i32
            %21 = relalg.getattr %12 @supplier::@s_suppkey : i32
            %22 = db.compare eq %20 : i32,%21 : i32
            %23 = relalg.getattr %12 @customer::@c_nationkey : i32
            %24 = relalg.getattr %12 @supplier::@s_nationkey : i32
            %25 = db.compare eq %23 : i32,%24 : i32
            %26 = relalg.getattr %12 @supplier::@s_nationkey : i32
            %27 = relalg.getattr %12 @nation::@n_nationkey : i32
            %28 = db.compare eq %26 : i32,%27 : i32
            %29 = relalg.getattr %12 @nation::@n_regionkey : i32
            %30 = relalg.getattr %12 @region::@r_regionkey : i32
            %31 = db.compare eq %29 : i32,%30 : i32
            %32 = relalg.getattr %12 @region::@r_name : !db.string
            %33 = db.constant ("ASIA") :!db.string
            %34 = db.compare eq %32 : !db.string,%33 : !db.string
            %35 = relalg.getattr %12 @orders::@o_orderdate : !db.date<day>
            %36 = db.constant ("1994-01-01") :!db.date<day>
            %37 = db.compare gte %35 : !db.date<day>,%36 : !db.date<day>
            %38 = relalg.getattr %12 @orders::@o_orderdate : !db.date<day>
            %39 = db.constant ("1995-01-01") :!db.date<day>
            %40 = db.compare lt %38 : !db.date<day>,%39 : !db.date<day>
            %41 = db.and %16 : i1,%19 : i1,%22 : i1,%25 : i1,%28 : i1,%31 : i1,%34 : i1,%37 : i1,%40 : i1
            relalg.return %41 : i1
        }
        %43 = relalg.map @map %13 (%42: !relalg.tuple) {
            %44 = relalg.getattr %42 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %45 = db.constant (1) :!db.decimal<15,2>
            %46 = relalg.getattr %42 @lineitem::@l_discount : !db.decimal<15,2>
            %47 = db.sub %45 : !db.decimal<15,2>,%46 : !db.decimal<15,2>
            %48 = db.mul %44 : !db.decimal<15,2>,%47 : !db.decimal<15,2>
            %49 = relalg.addattr %42, @aggfmname1({type=!db.decimal<15,2>}) %48
            relalg.return %49 : !relalg.tuple
        }
        %52 = relalg.aggregation @aggr %43 [@nation::@n_name] (%50 : !relalg.tuplestream, %51 : !relalg.tuple) {
            %53 = relalg.aggrfn sum @map::@aggfmname1 %50 : !db.decimal<15,2>
            %54 = relalg.addattr %51, @aggfmname2({type=!db.decimal<15,2>}) %53
            relalg.return %54 : !relalg.tuple
        }
        %55 = relalg.sort %52 [(@aggr::@aggfmname2,desc)]
        %56 = relalg.materialize %55 [@nation::@n_name,@aggr::@aggfmname2] => ["n_name","revenue"] : !db.table
        return %56 : !db.table
    }
}


