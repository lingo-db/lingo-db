module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=!db.int<32>}),
            p_name => @p_name({type=!db.string}),
            p_mfgr => @p_mfgr({type=!db.string}),
            p_brand => @p_brand({type=!db.string}),
            p_type => @p_type({type=!db.string}),
            p_size => @p_size({type=!db.int<32>}),
            p_container => @p_container({type=!db.string}),
            p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
            p_comment => @p_comment({type=!db.string})
        }
        %2 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=!db.int<32>}),
            s_name => @s_name({type=!db.string}),
            s_address => @s_address({type=!db.string}),
            s_nationkey => @s_nationkey({type=!db.int<32>}),
            s_phone => @s_phone({type=!db.string}),
            s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
            s_comment => @s_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %4 = relalg.basetable @partsupp { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=!db.int<32>}),
            ps_suppkey => @ps_suppkey({type=!db.int<32>}),
            ps_availqty => @ps_availqty({type=!db.int<32>}),
            ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
            ps_comment => @ps_comment({type=!db.string})
        }
        %5 = relalg.crossproduct %3, %4
        %6 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<32>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<32>}),
            n_comment => @n_comment({type=!db.nullable<!db.string>})
        }
        %7 = relalg.crossproduct %5, %6
        %8 = relalg.basetable @region { table_identifier="region", rows=5 , pkey=["r_regionkey"]} columns: {r_regionkey => @r_regionkey({type=!db.int<32>}),
            r_name => @r_name({type=!db.string}),
            r_comment => @r_comment({type=!db.nullable<!db.string>})
        }
        %9 = relalg.crossproduct %7, %8
        %11 = relalg.selection %9(%10: !relalg.tuple) {
            %12 = relalg.getattr %10 @part::@p_partkey : !db.int<32>
            %13 = relalg.getattr %10 @partsupp::@ps_partkey : !db.int<32>
            %14 = db.compare eq %12 : !db.int<32>,%13 : !db.int<32>
            %15 = relalg.getattr %10 @supplier::@s_suppkey : !db.int<32>
            %16 = relalg.getattr %10 @partsupp::@ps_suppkey : !db.int<32>
            %17 = db.compare eq %15 : !db.int<32>,%16 : !db.int<32>
            %18 = relalg.getattr %10 @part::@p_size : !db.int<32>
            %19 = db.constant (15) :!db.int<64>
            %20 = db.cast %18 : !db.int<32> -> !db.int<64>
            %21 = db.compare eq %20 : !db.int<64>,%19 : !db.int<64>
            %22 = relalg.getattr %10 @part::@p_type : !db.string
            %23 = db.constant ("%BRASS") :!db.string
            %24 = db.compare like %22 : !db.string,%23 : !db.string
            %25 = relalg.getattr %10 @supplier::@s_nationkey : !db.int<32>
            %26 = relalg.getattr %10 @nation::@n_nationkey : !db.int<32>
            %27 = db.compare eq %25 : !db.int<32>,%26 : !db.int<32>
            %28 = relalg.getattr %10 @nation::@n_regionkey : !db.int<32>
            %29 = relalg.getattr %10 @region::@r_regionkey : !db.int<32>
            %30 = db.compare eq %28 : !db.int<32>,%29 : !db.int<32>
            %31 = relalg.getattr %10 @region::@r_name : !db.string
            %32 = db.constant ("EUROPE") :!db.string
            %33 = db.compare eq %31 : !db.string,%32 : !db.string
            %34 = relalg.getattr %10 @partsupp::@ps_supplycost : !db.decimal<15,2>
            %35 = relalg.basetable @partsupp1 { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=!db.int<32>}),
                ps_suppkey => @ps_suppkey({type=!db.int<32>}),
                ps_availqty => @ps_availqty({type=!db.int<32>}),
                ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
                ps_comment => @ps_comment({type=!db.string})
            }
            %36 = relalg.basetable @supplier1 { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=!db.int<32>}),
                s_name => @s_name({type=!db.string}),
                s_address => @s_address({type=!db.string}),
                s_nationkey => @s_nationkey({type=!db.int<32>}),
                s_phone => @s_phone({type=!db.string}),
                s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
                s_comment => @s_comment({type=!db.string})
            }
            %37 = relalg.crossproduct %35, %36
            %38 = relalg.basetable @nation1 { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<32>}),
                n_name => @n_name({type=!db.string}),
                n_regionkey => @n_regionkey({type=!db.int<32>}),
                n_comment => @n_comment({type=!db.nullable<!db.string>})
            }
            %39 = relalg.crossproduct %37, %38
            %40 = relalg.basetable @region1 { table_identifier="region", rows=5 , pkey=["r_regionkey"]} columns: {r_regionkey => @r_regionkey({type=!db.int<32>}),
                r_name => @r_name({type=!db.string}),
                r_comment => @r_comment({type=!db.nullable<!db.string>})
            }
            %41 = relalg.crossproduct %39, %40
            %43 = relalg.selection %41(%42: !relalg.tuple) {
                %44 = relalg.getattr %10 @part::@p_partkey : !db.int<32>
                %45 = relalg.getattr %42 @partsupp1::@ps_partkey : !db.int<32>
                %46 = db.compare eq %44 : !db.int<32>,%45 : !db.int<32>
                %47 = relalg.getattr %42 @supplier1::@s_suppkey : !db.int<32>
                %48 = relalg.getattr %42 @partsupp1::@ps_suppkey : !db.int<32>
                %49 = db.compare eq %47 : !db.int<32>,%48 : !db.int<32>
                %50 = relalg.getattr %42 @supplier1::@s_nationkey : !db.int<32>
                %51 = relalg.getattr %42 @nation1::@n_nationkey : !db.int<32>
                %52 = db.compare eq %50 : !db.int<32>,%51 : !db.int<32>
                %53 = relalg.getattr %42 @nation1::@n_regionkey : !db.int<32>
                %54 = relalg.getattr %42 @region1::@r_regionkey : !db.int<32>
                %55 = db.compare eq %53 : !db.int<32>,%54 : !db.int<32>
                %56 = relalg.getattr %42 @region1::@r_name : !db.string
                %57 = db.constant ("EUROPE") :!db.string
                %58 = db.compare eq %56 : !db.string,%57 : !db.string
                %59 = db.and %46 : !db.bool,%49 : !db.bool,%52 : !db.bool,%55 : !db.bool,%58 : !db.bool
                relalg.return %59 : !db.bool
            }
            %62 = relalg.aggregation @aggr %43 [] (%60 : !relalg.tuplestream, %61 : !relalg.tuple) {
                %63 = relalg.aggrfn min @partsupp1::@ps_supplycost %60 : !db.nullable<!db.decimal<15,2>>
                %64 = relalg.addattr %61, @aggfmname1({type=!db.nullable<!db.decimal<15,2>>}) %63
                relalg.return %64 : !relalg.tuple
            }
            %65 = relalg.getscalar @aggr::@aggfmname1 %62 : !db.nullable<!db.decimal<15,2>>
            %66 = db.compare eq %34 : !db.decimal<15,2>,%65 : !db.nullable<!db.decimal<15,2>>
            %67 = db.and %14 : !db.bool,%17 : !db.bool,%21 : !db.bool,%24 : !db.bool,%27 : !db.bool,%30 : !db.bool,%33 : !db.bool,%66 : !db.nullable<!db.bool>
            relalg.return %67 : !db.nullable<!db.bool>
        }
        %68 = relalg.sort %11 [(@supplier::@s_acctbal,desc),(@nation::@n_name,asc),(@supplier::@s_name,asc),(@part::@p_partkey,asc)]
        %69 = relalg.limit 100 %68
        %70 = relalg.materialize %69 [@supplier::@s_acctbal,@supplier::@s_name,@nation::@n_name,@part::@p_partkey,@part::@p_mfgr,@supplier::@s_address,@supplier::@s_phone,@supplier::@s_comment] => ["s_acctbal","s_name","n_name","p_partkey","p_mfgr","s_address","s_phone","s_comment"] : !db.table
        return %70 : !db.table
    }
}

