module @querymodule{
    func @main (%executionContext: !util.generic_memref<i8>)  -> !db.table{
        %1 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=!db.int<64>}),
            s_name => @s_name({type=!db.string}),
            s_address => @s_address({type=!db.string}),
            s_nationkey => @s_nationkey({type=!db.int<64>}),
            s_phone => @s_phone({type=!db.string}),
            s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
            s_comment => @s_comment({type=!db.string})
        }
        %2 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<64>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<64>}),
            n_comment => @n_comment({type=!db.string<nullable>})
        }
        %3 = relalg.crossproduct %1, %2
        %5 = relalg.selection %3(%4: !relalg.tuple) {
            %6 = relalg.basetable @partsupp { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=!db.int<64>}),
                ps_suppkey => @ps_suppkey({type=!db.int<64>}),
                ps_availqty => @ps_availqty({type=!db.int<32>}),
                ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
                ps_comment => @ps_comment({type=!db.string})
            }
            %8 = relalg.selection %6(%7: !relalg.tuple) {
                %9 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=!db.int<64>}),
                    p_name => @p_name({type=!db.string}),
                    p_mfgr => @p_mfgr({type=!db.string}),
                    p_brand => @p_brand({type=!db.string}),
                    p_type => @p_type({type=!db.string}),
                    p_size => @p_size({type=!db.int<32>}),
                    p_container => @p_container({type=!db.string}),
                    p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
                    p_comment => @p_comment({type=!db.string})
                }
                %11 = relalg.selection %9(%10: !relalg.tuple) {
                    %12 = relalg.getattr %10 @part::@p_name : !db.string
                    %13 = db.constant ("forest%") :!db.string
                    %14 = db.compare like %12 : !db.string,%13 : !db.string
                    relalg.return %14 : !db.bool
                }
                %15 = relalg.projection all [@part::@p_partkey]%11
                %16 = relalg.getattr %7 @partsupp::@ps_partkey : !db.int<64>
                %17 = relalg.in %16 : !db.int<64>, %15
                %18 = relalg.getattr %7 @partsupp::@ps_availqty : !db.int<32>
                %19 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
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
                %21 = relalg.selection %19(%20: !relalg.tuple) {
                    %22 = relalg.getattr %20 @lineitem::@l_partkey : !db.int<64>
                    %23 = relalg.getattr %7 @partsupp::@ps_partkey : !db.int<64>
                    %24 = db.compare eq %22 : !db.int<64>,%23 : !db.int<64>
                    %25 = relalg.getattr %20 @lineitem::@l_suppkey : !db.int<64>
                    %26 = relalg.getattr %7 @partsupp::@ps_suppkey : !db.int<64>
                    %27 = db.compare eq %25 : !db.int<64>,%26 : !db.int<64>
                    %28 = relalg.getattr %20 @lineitem::@l_shipdate : !db.date<day>
                    %29 = db.constant ("1994-01-01") :!db.date<day>
                    %30 = db.compare gte %28 : !db.date<day>,%29 : !db.date<day>
                    %31 = relalg.getattr %20 @lineitem::@l_shipdate : !db.date<day>
                    %32 = db.constant ("1995-01-01") :!db.date<day>
                    %33 = db.compare lt %31 : !db.date<day>,%32 : !db.date<day>
                    %34 = db.and %24 : !db.bool,%27 : !db.bool,%30 : !db.bool,%33 : !db.bool
                    relalg.return %34 : !db.bool
                }
                %36 = relalg.aggregation @aggr2 %21 [] (%35 : !relalg.relation) {
                    %37 = relalg.aggrfn sum @lineitem::@l_quantity %35 : !db.decimal<15,2,nullable>
                    relalg.addattr @aggfmname1({type=!db.decimal<15,2,nullable>}) %37
                    relalg.return
                }
                %39 = relalg.map @map4 %36 (%38: !relalg.tuple) {
                    %40 = db.constant ("0.5") :!db.decimal<15,2>
                    %41 = relalg.getattr %38 @aggr2::@aggfmname1 : !db.decimal<15,2,nullable>
                    %42 = db.mul %40 : !db.decimal<15,2>,%41 : !db.decimal<15,2,nullable>
                    relalg.addattr @aggfmname2({type=!db.decimal<15,2,nullable>}) %42
                    relalg.return
                }
                %43 = relalg.getscalar @map4::@aggfmname2 %39 : !db.decimal<15,2,nullable>
                %44 = db.cast %18 : !db.int<32> -> !db.decimal<15,2,nullable>
                %45 = db.compare gt %44 : !db.decimal<15,2,nullable>,%43 : !db.decimal<15,2,nullable>
                %46 = db.and %17 : !db.bool,%45 : !db.bool<nullable>
                relalg.return %46 : !db.bool<nullable>
            }
            %47 = relalg.projection all [@partsupp::@ps_suppkey]%8
            %48 = relalg.getattr %4 @supplier::@s_suppkey : !db.int<64>
            %49 = relalg.in %48 : !db.int<64>, %47
            %50 = relalg.getattr %4 @supplier::@s_nationkey : !db.int<64>
            %51 = relalg.getattr %4 @nation::@n_nationkey : !db.int<64>
            %52 = db.compare eq %50 : !db.int<64>,%51 : !db.int<64>
            %53 = relalg.getattr %4 @nation::@n_name : !db.string
            %54 = db.constant ("CANADA") :!db.string
            %55 = db.compare eq %53 : !db.string,%54 : !db.string
            %56 = db.and %49 : !db.bool,%52 : !db.bool,%55 : !db.bool
            relalg.return %56 : !db.bool
        }
        %57 = relalg.sort %5 [(@supplier::@s_name,asc)]
        %58 = relalg.materialize %57 [@supplier::@s_name,@supplier::@s_address] => ["s_name","s_address"] : !db.table
        return %58 : !db.table
    }
}

