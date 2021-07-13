module @querymodule{
    func @main (%executionContext: !util.generic_memref<i8>)  -> !db.table{
        %1 = relalg.basetable @partsupp { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=!db.int<64>}),
            ps_suppkey => @ps_suppkey({type=!db.int<64>}),
            ps_availqty => @ps_availqty({type=!db.int<32>}),
            ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
            ps_comment => @ps_comment({type=!db.string})
        }
        %2 = relalg.basetable @supplier { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=!db.int<64>}),
            s_name => @s_name({type=!db.string}),
            s_address => @s_address({type=!db.string}),
            s_nationkey => @s_nationkey({type=!db.int<64>}),
            s_phone => @s_phone({type=!db.string}),
            s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
            s_comment => @s_comment({type=!db.string})
        }
        %3 = relalg.crossproduct %1, %2
        %4 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<64>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<64>}),
            n_comment => @n_comment({type=!db.string<nullable>})
        }
        %5 = relalg.crossproduct %3, %4
        %7 = relalg.selection %5(%6: !relalg.tuple) {
            %8 = relalg.getattr %6 @partsupp::@ps_suppkey : !db.int<64>
            %9 = relalg.getattr %6 @supplier::@s_suppkey : !db.int<64>
            %10 = db.compare eq %8 : !db.int<64>,%9 : !db.int<64>
            %11 = relalg.getattr %6 @supplier::@s_nationkey : !db.int<64>
            %12 = relalg.getattr %6 @nation::@n_nationkey : !db.int<64>
            %13 = db.compare eq %11 : !db.int<64>,%12 : !db.int<64>
            %14 = relalg.getattr %6 @nation::@n_name : !db.string
            %15 = db.constant ("GERMANY") :!db.string
            %16 = db.compare eq %14 : !db.string,%15 : !db.string
            %17 = db.and %10 : !db.bool,%13 : !db.bool,%16 : !db.bool
            relalg.return %17 : !db.bool
        }
        %19 = relalg.map @map1 %7 (%18: !relalg.tuple) {
            %20 = relalg.getattr %18 @partsupp::@ps_supplycost : !db.decimal<15,2>
            %21 = relalg.getattr %18 @partsupp::@ps_availqty : !db.int<32>
            %22 = db.cast %21 : !db.int<32> -> !db.decimal<15,2>
            %23 = db.mul %20 : !db.decimal<15,2>,%22 : !db.decimal<15,2>
            relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %23
            %24 = relalg.getattr %18 @partsupp::@ps_supplycost : !db.decimal<15,2>
            %25 = relalg.getattr %18 @partsupp::@ps_availqty : !db.int<32>
            %26 = db.cast %25 : !db.int<32> -> !db.decimal<15,2>
            %27 = db.mul %24 : !db.decimal<15,2>,%26 : !db.decimal<15,2>
            relalg.addattr @aggfmname3({type=!db.decimal<15,2>}) %27
            relalg.return
        }
        %29 = relalg.aggregation @aggr1 %19 [@partsupp::@ps_partkey] (%28 : !relalg.relation) {
            %30 = relalg.aggrfn sum @map1::@aggfmname1 %28 : !db.decimal<15,2>
            relalg.addattr @aggfmname2({type=!db.decimal<15,2>}) %30
            %31 = relalg.aggrfn sum @map1::@aggfmname3 %28 : !db.decimal<15,2>
            relalg.addattr @aggfmname4({type=!db.decimal<15,2>}) %31
            relalg.return
        }
        %33 = relalg.selection %29(%32: !relalg.tuple) {
            %34 = relalg.getattr %32 @aggr1::@aggfmname2 : !db.decimal<15,2>
            %35 = relalg.basetable @partsupp1 { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=!db.int<64>}),
                ps_suppkey => @ps_suppkey({type=!db.int<64>}),
                ps_availqty => @ps_availqty({type=!db.int<32>}),
                ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
                ps_comment => @ps_comment({type=!db.string})
            }
            %36 = relalg.basetable @supplier1 { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=!db.int<64>}),
                s_name => @s_name({type=!db.string}),
                s_address => @s_address({type=!db.string}),
                s_nationkey => @s_nationkey({type=!db.int<64>}),
                s_phone => @s_phone({type=!db.string}),
                s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
                s_comment => @s_comment({type=!db.string})
            }
            %37 = relalg.crossproduct %35, %36
            %38 = relalg.basetable @nation1 { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<64>}),
                n_name => @n_name({type=!db.string}),
                n_regionkey => @n_regionkey({type=!db.int<64>}),
                n_comment => @n_comment({type=!db.string<nullable>})
            }
            %39 = relalg.crossproduct %37, %38
            %41 = relalg.selection %39(%40: !relalg.tuple) {
                %42 = relalg.getattr %40 @partsupp1::@ps_suppkey : !db.int<64>
                %43 = relalg.getattr %40 @supplier1::@s_suppkey : !db.int<64>
                %44 = db.compare eq %42 : !db.int<64>,%43 : !db.int<64>
                %45 = relalg.getattr %40 @supplier1::@s_nationkey : !db.int<64>
                %46 = relalg.getattr %40 @nation1::@n_nationkey : !db.int<64>
                %47 = db.compare eq %45 : !db.int<64>,%46 : !db.int<64>
                %48 = relalg.getattr %40 @nation1::@n_name : !db.string
                %49 = db.constant ("GERMANY") :!db.string
                %50 = db.compare eq %48 : !db.string,%49 : !db.string
                %51 = db.and %44 : !db.bool,%47 : !db.bool,%50 : !db.bool
                relalg.return %51 : !db.bool
            }
            %53 = relalg.map @map3 %41 (%52: !relalg.tuple) {
                %54 = relalg.getattr %52 @partsupp1::@ps_supplycost : !db.decimal<15,2>
                %55 = relalg.getattr %52 @partsupp1::@ps_availqty : !db.int<32>
                %56 = db.cast %55 : !db.int<32> -> !db.decimal<15,2>
                %57 = db.mul %54 : !db.decimal<15,2>,%56 : !db.decimal<15,2>
                relalg.addattr @aggfmname1({type=!db.decimal<15,2>}) %57
                relalg.return
            }
            %59 = relalg.aggregation @aggr2 %53 [] (%58 : !relalg.relation) {
                %60 = relalg.aggrfn sum @map3::@aggfmname1 %58 : !db.decimal<15,2,nullable>
                relalg.addattr @aggfmname2({type=!db.decimal<15,2,nullable>}) %60
                relalg.return
            }
            %62 = relalg.map @map4 %59 (%61: !relalg.tuple) {
                %63 = relalg.getattr %61 @aggr2::@aggfmname2 : !db.decimal<15,2,nullable>
                %64 = db.constant ("0.0001") :!db.decimal<15,4>
                %65 = db.cast %63 : !db.decimal<15,2,nullable> -> !db.decimal<15,4,nullable>
                %66 = db.mul %65 : !db.decimal<15,4,nullable>,%64 : !db.decimal<15,4>
                relalg.addattr @aggfmname3({type=!db.decimal<15,4,nullable>}) %66
                relalg.return
            }
            %67 = relalg.getscalar @map4::@aggfmname3 %62 : !db.decimal<15,4,nullable>
            %68 = db.cast %34 : !db.decimal<15,2> -> !db.decimal<15,4,nullable>
            %69 = db.compare gt %68 : !db.decimal<15,4,nullable>,%67 : !db.decimal<15,4,nullable>
            relalg.return %69 : !db.bool<nullable>
        }
        %70 = relalg.sort %33 [(@aggr1::@aggfmname4,desc)]
        %71 = relalg.materialize %70 [@partsupp::@ps_partkey,@aggr1::@aggfmname4] => ["ps_partkey","value"] : !db.table
        return %71 : !db.table
    }
}

