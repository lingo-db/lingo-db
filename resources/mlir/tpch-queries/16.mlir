module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @partsupp { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=i32}),
            ps_suppkey => @ps_suppkey({type=i32}),
            ps_availqty => @ps_availqty({type=i32}),
            ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
            ps_comment => @ps_comment({type=!db.string})
        }
        %2 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=i32}),
            p_name => @p_name({type=!db.string}),
            p_mfgr => @p_mfgr({type=!db.string}),
            p_brand => @p_brand({type=!db.string}),
            p_type => @p_type({type=!db.string}),
            p_size => @p_size({type=i32}),
            p_container => @p_container({type=!db.string}),
            p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
            p_comment => @p_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %5 = relalg.selection %3(%4: !relalg.tuple) {
            %6 = relalg.getattr %4 @part::@p_partkey : i32
            %7 = relalg.getattr %4 @partsupp::@ps_partkey : i32
            %8 = db.compare eq %6 : i32,%7 : i32
            %9 = relalg.getattr %4 @part::@p_brand : !db.string
            %10 = db.constant ("Brand#45") :!db.string
            %11 = db.compare neq %9 : !db.string,%10 : !db.string
            %12 = relalg.getattr %4 @part::@p_type : !db.string
            %13 = db.constant ("MEDIUM POLISHED%") :!db.string
            %14 = db.compare like %12 : !db.string,%13 : !db.string
            %15 = db.not %14 : i1
            %16 = relalg.getattr %4 @part::@p_size : i32
            %17 = db.constant (49) :i32
            %18 = db.compare eq %16 : i32,%17 : i32
            %19 = db.constant (14) :i32
            %20 = db.compare eq %16 : i32,%19 : i32
            %21 = db.constant (23) :i32
            %22 = db.compare eq %16 : i32,%21 : i32
            %23 = db.constant (45) :i32
            %24 = db.compare eq %16 : i32,%23 : i32
            %25 = db.constant (19) :i32
            %26 = db.compare eq %16 : i32,%25 : i32
            %27 = db.constant (3) :i32
            %28 = db.compare eq %16 : i32,%27 : i32
            %29 = db.constant (36) :i32
            %30 = db.compare eq %16 : i32,%29 : i32
            %31 = db.constant (9) :i32
            %32 = db.compare eq %16 : i32,%31 : i32
            %33 = db.or %18 : i1,%20 : i1,%22 : i1,%24 : i1,%26 : i1,%28 : i1,%30 : i1,%32 : i1
            %34 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=i32}),
                s_name => @s_name({type=!db.string}),
                s_address => @s_address({type=!db.string}),
                s_nationkey => @s_nationkey({type=i32}),
                s_phone => @s_phone({type=!db.string}),
                s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
                s_comment => @s_comment({type=!db.string})
            }
            %36 = relalg.selection %34(%35: !relalg.tuple) {
                %37 = relalg.getattr %35 @supplier::@s_comment : !db.string
                %38 = db.constant ("%Customer%Complaints%") :!db.string
                %39 = db.compare like %37 : !db.string,%38 : !db.string
                relalg.return %39 : i1
            }
            %40 = relalg.projection all [@supplier::@s_suppkey]%36
            %41 = relalg.getattr %4 @partsupp::@ps_suppkey : i32
            %42 = relalg.in %41 : i32, %40
            %43 = db.not %42 : i1
            %44 = db.and %8 : i1,%11 : i1,%15 : i1,%33 : i1,%43 : i1
            relalg.return %44 : i1
        }
        %47 = relalg.aggregation @aggr1 %5 [@part::@p_brand,@part::@p_type,@part::@p_size] (%45 : !relalg.tuplestream, %46 : !relalg.tuple) {
            %48 = relalg.projection distinct [@partsupp::@ps_suppkey]%45
            %49 = relalg.aggrfn count @partsupp::@ps_suppkey %48 : i64
            %50 = relalg.addattr %46, @aggfmname1({type=i64}) %49
            relalg.return %50 : !relalg.tuple
        }
        %51 = relalg.sort %47 [(@aggr1::@aggfmname1,desc),(@part::@p_brand,asc),(@part::@p_type,asc),(@part::@p_size,asc)]
        %52 = relalg.materialize %51 [@part::@p_brand,@part::@p_type,@part::@p_size,@aggr1::@aggfmname1] => ["p_brand","p_type","p_size","supplier_cnt"] : !db.table
        return %52 : !db.table
    }
}

