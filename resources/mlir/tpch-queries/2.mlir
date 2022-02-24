module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=i32}),
            p_name => @p_name({type=!db.string}),
            p_mfgr => @p_mfgr({type=!db.string}),
            p_brand => @p_brand({type=!db.string}),
            p_type => @p_type({type=!db.string}),
            p_size => @p_size({type=i32}),
            p_container => @p_container({type=!db.string}),
            p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
            p_comment => @p_comment({type=!db.string})
        }
        %2 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=i32}),
            s_name => @s_name({type=!db.string}),
            s_address => @s_address({type=!db.string}),
            s_nationkey => @s_nationkey({type=i32}),
            s_phone => @s_phone({type=!db.string}),
            s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
            s_comment => @s_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %4 = relalg.basetable @partsupp { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=i32}),
            ps_suppkey => @ps_suppkey({type=i32}),
            ps_availqty => @ps_availqty({type=i32}),
            ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
            ps_comment => @ps_comment({type=!db.string})
        }
        %5 = relalg.crossproduct %3, %4
        %6 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=i32}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=i32}),
            n_comment => @n_comment({type=!db.nullable<!db.string>})
        }
        %7 = relalg.crossproduct %5, %6
        %8 = relalg.basetable @region { table_identifier="region", rows=5 , pkey=["r_regionkey"]} columns: {r_regionkey => @r_regionkey({type=i32}),
            r_name => @r_name({type=!db.string}),
            r_comment => @r_comment({type=!db.nullable<!db.string>})
        }
        %9 = relalg.crossproduct %7, %8
        %11 = relalg.selection %9(%10: !relalg.tuple) {
            %12 = relalg.getattr %10 @part::@p_partkey : i32
            %13 = relalg.getattr %10 @partsupp::@ps_partkey : i32
            %14 = db.compare eq %12 : i32,%13 : i32
            %15 = relalg.getattr %10 @supplier::@s_suppkey : i32
            %16 = relalg.getattr %10 @partsupp::@ps_suppkey : i32
            %17 = db.compare eq %15 : i32,%16 : i32
            %18 = relalg.getattr %10 @part::@p_size : i32
            %19 = db.constant (15) :i64
            %20 = db.cast %18 : i32 -> i64
            %21 = db.compare eq %20 : i64,%19 : i64
            %22 = relalg.getattr %10 @part::@p_type : !db.string
            %23 = db.constant ("%BRASS") :!db.string
            %24 = db.compare like %22 : !db.string,%23 : !db.string
            %25 = relalg.getattr %10 @supplier::@s_nationkey : i32
            %26 = relalg.getattr %10 @nation::@n_nationkey : i32
            %27 = db.compare eq %25 : i32,%26 : i32
            %28 = relalg.getattr %10 @nation::@n_regionkey : i32
            %29 = relalg.getattr %10 @region::@r_regionkey : i32
            %30 = db.compare eq %28 : i32,%29 : i32
            %31 = relalg.getattr %10 @region::@r_name : !db.string
            %32 = db.constant ("EUROPE") :!db.string
            %33 = db.compare eq %31 : !db.string,%32 : !db.string
            %34 = relalg.getattr %10 @partsupp::@ps_supplycost : !db.decimal<15,2>
            %35 = relalg.basetable @partsupp1 { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=i32}),
                ps_suppkey => @ps_suppkey({type=i32}),
                ps_availqty => @ps_availqty({type=i32}),
                ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
                ps_comment => @ps_comment({type=!db.string})
            }
            %36 = relalg.basetable @supplier1 { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=i32}),
                s_name => @s_name({type=!db.string}),
                s_address => @s_address({type=!db.string}),
                s_nationkey => @s_nationkey({type=i32}),
                s_phone => @s_phone({type=!db.string}),
                s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
                s_comment => @s_comment({type=!db.string})
            }
            %37 = relalg.crossproduct %35, %36
            %38 = relalg.basetable @nation1 { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=i32}),
                n_name => @n_name({type=!db.string}),
                n_regionkey => @n_regionkey({type=i32}),
                n_comment => @n_comment({type=!db.nullable<!db.string>})
            }
            %39 = relalg.crossproduct %37, %38
            %40 = relalg.basetable @region1 { table_identifier="region", rows=5 , pkey=["r_regionkey"]} columns: {r_regionkey => @r_regionkey({type=i32}),
                r_name => @r_name({type=!db.string}),
                r_comment => @r_comment({type=!db.nullable<!db.string>})
            }
            %41 = relalg.crossproduct %39, %40
            %43 = relalg.selection %41(%42: !relalg.tuple) {
                %44 = relalg.getattr %10 @part::@p_partkey : i32
                %45 = relalg.getattr %42 @partsupp1::@ps_partkey : i32
                %46 = db.compare eq %44 : i32,%45 : i32
                %47 = relalg.getattr %42 @supplier1::@s_suppkey : i32
                %48 = relalg.getattr %42 @partsupp1::@ps_suppkey : i32
                %49 = db.compare eq %47 : i32,%48 : i32
                %50 = relalg.getattr %42 @supplier1::@s_nationkey : i32
                %51 = relalg.getattr %42 @nation1::@n_nationkey : i32
                %52 = db.compare eq %50 : i32,%51 : i32
                %53 = relalg.getattr %42 @nation1::@n_regionkey : i32
                %54 = relalg.getattr %42 @region1::@r_regionkey : i32
                %55 = db.compare eq %53 : i32,%54 : i32
                %56 = relalg.getattr %42 @region1::@r_name : !db.string
                %57 = db.constant ("EUROPE") :!db.string
                %58 = db.compare eq %56 : !db.string,%57 : !db.string
                %59 = db.and %46 : i1,%49 : i1,%52 : i1,%55 : i1,%58 : i1
                relalg.return %59 : i1
            }
            %62 = relalg.aggregation @aggr %43 [] (%60 : !relalg.tuplestream, %61 : !relalg.tuple) {
                %63 = relalg.aggrfn min @partsupp1::@ps_supplycost %60 : !db.nullable<!db.decimal<15,2>>
                %64 = relalg.addattr %61, @aggfmname1({type=!db.nullable<!db.decimal<15,2>>}) %63
                relalg.return %64 : !relalg.tuple
            }
            %65 = relalg.getscalar @aggr::@aggfmname1 %62 : !db.nullable<!db.decimal<15,2>>
            %66 = db.compare eq %34 : !db.decimal<15,2>,%65 : !db.nullable<!db.decimal<15,2>>
            %67 = db.and %14 : i1,%17 : i1,%21 : i1,%24 : i1,%27 : i1,%30 : i1,%33 : i1,%66 : !db.nullable<i1>
            relalg.return %67 : !db.nullable<i1>
        }
        %68 = relalg.sort %11 [(@supplier::@s_acctbal,desc),(@nation::@n_name,asc),(@supplier::@s_name,asc),(@part::@p_partkey,asc)]
        %69 = relalg.limit 100 %68
        %70 = relalg.materialize %69 [@supplier::@s_acctbal,@supplier::@s_name,@nation::@n_name,@part::@p_partkey,@part::@p_mfgr,@supplier::@s_address,@supplier::@s_phone,@supplier::@s_comment] => ["s_acctbal","s_name","n_name","p_partkey","p_mfgr","s_address","s_phone","s_comment"] : !db.table
        return %70 : !db.table
    }
}

