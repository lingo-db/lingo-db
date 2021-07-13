module @querymodule{
    func @main (%executionContext: !util.generic_memref<i8>)  -> !db.table{
        %1 = relalg.basetable @partsupp { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=!db.int<64>}),
            ps_suppkey => @ps_suppkey({type=!db.int<64>}),
            ps_availqty => @ps_availqty({type=!db.int<32>}),
            ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
            ps_comment => @ps_comment({type=!db.string})
        }
        %2 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=!db.int<64>}),
            p_name => @p_name({type=!db.string}),
            p_mfgr => @p_mfgr({type=!db.string}),
            p_brand => @p_brand({type=!db.string}),
            p_type => @p_type({type=!db.string}),
            p_size => @p_size({type=!db.int<32>}),
            p_container => @p_container({type=!db.string}),
            p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
            p_comment => @p_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %5 = relalg.selection %3(%4: !relalg.tuple) {
            %6 = relalg.getattr %4 @part::@p_partkey : !db.int<64>
            %7 = relalg.getattr %4 @partsupp::@ps_partkey : !db.int<64>
            %8 = db.compare eq %6 : !db.int<64>,%7 : !db.int<64>
            %9 = relalg.getattr %4 @part::@p_brand : !db.string
            %10 = db.constant ("Brand#45") :!db.string
            %11 = db.compare neq %9 : !db.string,%10 : !db.string
            %12 = relalg.getattr %4 @part::@p_type : !db.string
            %13 = db.constant ("MEDIUM POLISHED%") :!db.string
            %14 = db.compare like %12 : !db.string,%13 : !db.string
            %15 = db.not %14 : !db.bool
            %16 = relalg.getattr %4 @part::@p_size : !db.int<32>
            %17 = db.constant (49) :!db.int<32>
            %18 = db.compare eq %16 : !db.int<32>,%17 : !db.int<32>
            %19 = db.constant (14) :!db.int<32>
            %20 = db.compare eq %16 : !db.int<32>,%19 : !db.int<32>
            %21 = db.constant (23) :!db.int<32>
            %22 = db.compare eq %16 : !db.int<32>,%21 : !db.int<32>
            %23 = db.constant (45) :!db.int<32>
            %24 = db.compare eq %16 : !db.int<32>,%23 : !db.int<32>
            %25 = db.constant (19) :!db.int<32>
            %26 = db.compare eq %16 : !db.int<32>,%25 : !db.int<32>
            %27 = db.constant (3) :!db.int<32>
            %28 = db.compare eq %16 : !db.int<32>,%27 : !db.int<32>
            %29 = db.constant (36) :!db.int<32>
            %30 = db.compare eq %16 : !db.int<32>,%29 : !db.int<32>
            %31 = db.constant (9) :!db.int<32>
            %32 = db.compare eq %16 : !db.int<32>,%31 : !db.int<32>
            %33 = db.or %18 : !db.bool,%20 : !db.bool,%22 : !db.bool,%24 : !db.bool,%26 : !db.bool,%28 : !db.bool,%30 : !db.bool,%32 : !db.bool
            %34 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=!db.int<64>}),
                s_name => @s_name({type=!db.string}),
                s_address => @s_address({type=!db.string}),
                s_nationkey => @s_nationkey({type=!db.int<64>}),
                s_phone => @s_phone({type=!db.string}),
                s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
                s_comment => @s_comment({type=!db.string})
            }
            %36 = relalg.selection %34(%35: !relalg.tuple) {
                %37 = relalg.getattr %35 @supplier::@s_comment : !db.string
                %38 = db.constant ("%Customer%Complaints%") :!db.string
                %39 = db.compare like %37 : !db.string,%38 : !db.string
                relalg.return %39 : !db.bool
            }
            %40 = relalg.projection all [@supplier::@s_suppkey]%36
            %41 = relalg.getattr %4 @partsupp::@ps_suppkey : !db.int<64>
            %42 = relalg.in %41 : !db.int<64>, %40
            %43 = db.not %42 : !db.bool
            %44 = db.and %8 : !db.bool,%11 : !db.bool,%15 : !db.bool,%33 : !db.bool,%43 : !db.bool
            relalg.return %44 : !db.bool
        }
        %46 = relalg.aggregation @aggr2 %5 [@part::@p_brand,@part::@p_type,@part::@p_size] (%45 : !relalg.relation) {
            %47 = relalg.projection distinct [@partsupp::@ps_suppkey]%45
            %48 = relalg.aggrfn count @partsupp::@ps_suppkey %47 : !db.int<64>
            relalg.addattr @aggfmname1({type=!db.int<64>}) %48
            relalg.return
        }
        %49 = relalg.sort %46 [(@aggr2::@aggfmname1,desc),(@part::@p_brand,asc),(@part::@p_type,asc),(@part::@p_size,asc)]
        %50 = relalg.materialize %49 [@part::@p_brand,@part::@p_type,@part::@p_size,@aggr2::@aggfmname1] => ["p_brand","p_type","p_size","supplier_cnt"] : !db.table
        return %50 : !db.table
    }
}

