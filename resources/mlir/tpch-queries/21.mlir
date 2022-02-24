module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=i32}),
            s_name => @s_name({type=!db.string}),
            s_address => @s_address({type=!db.string}),
            s_nationkey => @s_nationkey({type=i32}),
            s_phone => @s_phone({type=!db.string}),
            s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
            s_comment => @s_comment({type=!db.string})
        }
        %2 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=i32}),
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
        %3 = relalg.crossproduct %1, %2
        %4 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=i32}),
            o_custkey => @o_custkey({type=i32}),
            o_orderstatus => @o_orderstatus({type=!db.char<1>}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=i32}),
            o_comment => @o_comment({type=!db.string})
        }
        %5 = relalg.crossproduct %3, %4
        %6 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=i32}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=i32}),
            n_comment => @n_comment({type=!db.nullable<!db.string>})
        }
        %7 = relalg.crossproduct %5, %6
        %9 = relalg.selection %7(%8: !relalg.tuple) {
            %10 = relalg.getattr %8 @supplier::@s_suppkey : i32
            %11 = relalg.getattr %8 @lineitem::@l_suppkey : i32
            %12 = db.compare eq %10 : i32,%11 : i32
            %13 = relalg.getattr %8 @orders::@o_orderkey : i32
            %14 = relalg.getattr %8 @lineitem::@l_orderkey : i32
            %15 = db.compare eq %13 : i32,%14 : i32
            %16 = relalg.getattr %8 @orders::@o_orderstatus : !db.char<1>
            %17 = db.constant ("F") :!db.char<1>
            %18 = db.compare eq %16 : !db.char<1>,%17 : !db.char<1>
            %19 = relalg.getattr %8 @lineitem::@l_receiptdate : !db.date<day>
            %20 = relalg.getattr %8 @lineitem::@l_commitdate : !db.date<day>
            %21 = db.compare gt %19 : !db.date<day>,%20 : !db.date<day>
            %22 = relalg.basetable @lineitem1 { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=i32}),
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
            %24 = relalg.selection %22(%23: !relalg.tuple) {
                %25 = relalg.getattr %23 @lineitem1::@l_orderkey : i32
                %26 = relalg.getattr %8 @lineitem::@l_orderkey : i32
                %27 = db.compare eq %25 : i32,%26 : i32
                %28 = relalg.getattr %23 @lineitem1::@l_suppkey : i32
                %29 = relalg.getattr %8 @lineitem::@l_suppkey : i32
                %30 = db.compare neq %28 : i32,%29 : i32
                %31 = db.and %27 : i1,%30 : i1
                relalg.return %31 : i1
            }
            %32 = relalg.exists%24
            %33 = relalg.basetable @lineitem2 { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=i32}),
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
            %35 = relalg.selection %33(%34: !relalg.tuple) {
                %36 = relalg.getattr %34 @lineitem2::@l_orderkey : i32
                %37 = relalg.getattr %8 @lineitem::@l_orderkey : i32
                %38 = db.compare eq %36 : i32,%37 : i32
                %39 = relalg.getattr %34 @lineitem2::@l_suppkey : i32
                %40 = relalg.getattr %8 @lineitem::@l_suppkey : i32
                %41 = db.compare neq %39 : i32,%40 : i32
                %42 = relalg.getattr %34 @lineitem2::@l_receiptdate : !db.date<day>
                %43 = relalg.getattr %34 @lineitem2::@l_commitdate : !db.date<day>
                %44 = db.compare gt %42 : !db.date<day>,%43 : !db.date<day>
                %45 = db.and %38 : i1,%41 : i1,%44 : i1
                relalg.return %45 : i1
            }
            %46 = relalg.exists%35
            %47 = db.not %46 : i1
            %48 = relalg.getattr %8 @supplier::@s_nationkey : i32
            %49 = relalg.getattr %8 @nation::@n_nationkey : i32
            %50 = db.compare eq %48 : i32,%49 : i32
            %51 = relalg.getattr %8 @nation::@n_name : !db.string
            %52 = db.constant ("SAUDI ARABIA") :!db.string
            %53 = db.compare eq %51 : !db.string,%52 : !db.string
            %54 = db.and %12 : i1,%15 : i1,%18 : i1,%21 : i1,%32 : i1,%47 : i1,%50 : i1,%53 : i1
            relalg.return %54 : i1
        }
        %57 = relalg.aggregation @aggr2 %9 [@supplier::@s_name] (%55 : !relalg.tuplestream, %56 : !relalg.tuple) {
            %58 = relalg.count %55
            %59 = relalg.addattr %56, @aggfmname1({type=i64}) %58
            relalg.return %59 : !relalg.tuple
        }
        %60 = relalg.sort %57 [(@aggr2::@aggfmname1,desc),(@supplier::@s_name,asc)]
        %61 = relalg.limit 100 %60
        %62 = relalg.materialize %61 [@supplier::@s_name,@aggr2::@aggfmname1] => ["s_name","numwait"] : !db.table
        return %62 : !db.table
    }
}

