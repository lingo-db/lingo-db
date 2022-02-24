module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @partsupp { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=!db.int<32>}),
            ps_suppkey => @ps_suppkey({type=!db.int<32>}),
            ps_availqty => @ps_availqty({type=!db.int<32>}),
            ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
            ps_comment => @ps_comment({type=!db.string})
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
        %4 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<32>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<32>}),
            n_comment => @n_comment({type=!db.nullable<!db.string>})
        }
        %5 = relalg.crossproduct %3, %4
        %7 = relalg.selection %5(%6: !relalg.tuple) {
            %8 = relalg.getattr %6 @partsupp::@ps_suppkey : !db.int<32>
            %9 = relalg.getattr %6 @supplier::@s_suppkey : !db.int<32>
            %10 = db.compare eq %8 : !db.int<32>,%9 : !db.int<32>
            %11 = relalg.getattr %6 @supplier::@s_nationkey : !db.int<32>
            %12 = relalg.getattr %6 @nation::@n_nationkey : !db.int<32>
            %13 = db.compare eq %11 : !db.int<32>,%12 : !db.int<32>
            %14 = relalg.getattr %6 @nation::@n_name : !db.string
            %15 = db.constant ("GERMANY") :!db.string
            %16 = db.compare eq %14 : !db.string,%15 : !db.string
            %17 = db.and %10 : i1,%13 : i1,%16 : i1
            relalg.return %17 : i1
        }
        %19 = relalg.map @map %7 (%18: !relalg.tuple) {
            %20 = relalg.getattr %18 @partsupp::@ps_supplycost : !db.decimal<15,2>
            %21 = relalg.getattr %18 @partsupp::@ps_availqty : !db.int<32>
            %22 = db.cast %21 : !db.int<32> -> !db.decimal<15,2>
            %23 = db.mul %20 : !db.decimal<15,2>,%22 : !db.decimal<15,2>
            %24 = relalg.addattr %18, @aggfmname1({type=!db.decimal<15,2>}) %23
            %25 = relalg.getattr %18 @partsupp::@ps_supplycost : !db.decimal<15,2>
            %26 = relalg.getattr %18 @partsupp::@ps_availqty : !db.int<32>
            %27 = db.cast %26 : !db.int<32> -> !db.decimal<15,2>
            %28 = db.mul %25 : !db.decimal<15,2>,%27 : !db.decimal<15,2>
            %29 = relalg.addattr %24, @aggfmname3({type=!db.decimal<15,2>}) %28
            relalg.return %29 : !relalg.tuple
        }
        %32 = relalg.aggregation @aggr %19 [@partsupp::@ps_partkey] (%30 : !relalg.tuplestream, %31 : !relalg.tuple) {
            %33 = relalg.aggrfn sum @map::@aggfmname1 %30 : !db.decimal<15,2>
            %34 = relalg.addattr %31, @aggfmname2({type=!db.decimal<15,2>}) %33
            %35 = relalg.aggrfn sum @map::@aggfmname3 %30 : !db.decimal<15,2>
            %36 = relalg.addattr %34, @aggfmname4({type=!db.decimal<15,2>}) %35
            relalg.return %36 : !relalg.tuple
        }
        %38 = relalg.selection %32(%37: !relalg.tuple) {
            %39 = relalg.getattr %37 @aggr::@aggfmname2 : !db.decimal<15,2>
            %40 = relalg.basetable @partsupp1 { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=!db.int<32>}),
                ps_suppkey => @ps_suppkey({type=!db.int<32>}),
                ps_availqty => @ps_availqty({type=!db.int<32>}),
                ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
                ps_comment => @ps_comment({type=!db.string})
            }
            %41 = relalg.basetable @supplier1 { table_identifier="supplier", rows=1000 , pkey=["s_suppkey"]} columns: {s_suppkey => @s_suppkey({type=!db.int<32>}),
                s_name => @s_name({type=!db.string}),
                s_address => @s_address({type=!db.string}),
                s_nationkey => @s_nationkey({type=!db.int<32>}),
                s_phone => @s_phone({type=!db.string}),
                s_acctbal => @s_acctbal({type=!db.decimal<15,2>}),
                s_comment => @s_comment({type=!db.string})
            }
            %42 = relalg.crossproduct %40, %41
            %43 = relalg.basetable @nation1 { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<32>}),
                n_name => @n_name({type=!db.string}),
                n_regionkey => @n_regionkey({type=!db.int<32>}),
                n_comment => @n_comment({type=!db.nullable<!db.string>})
            }
            %44 = relalg.crossproduct %42, %43
            %46 = relalg.selection %44(%45: !relalg.tuple) {
                %47 = relalg.getattr %45 @partsupp1::@ps_suppkey : !db.int<32>
                %48 = relalg.getattr %45 @supplier1::@s_suppkey : !db.int<32>
                %49 = db.compare eq %47 : !db.int<32>,%48 : !db.int<32>
                %50 = relalg.getattr %45 @supplier1::@s_nationkey : !db.int<32>
                %51 = relalg.getattr %45 @nation1::@n_nationkey : !db.int<32>
                %52 = db.compare eq %50 : !db.int<32>,%51 : !db.int<32>
                %53 = relalg.getattr %45 @nation1::@n_name : !db.string
                %54 = db.constant ("GERMANY") :!db.string
                %55 = db.compare eq %53 : !db.string,%54 : !db.string
                %56 = db.and %49 : i1,%52 : i1,%55 : i1
                relalg.return %56 : i1
            }
            %58 = relalg.map @map2 %46 (%57: !relalg.tuple) {
                %59 = relalg.getattr %57 @partsupp1::@ps_supplycost : !db.decimal<15,2>
                %60 = relalg.getattr %57 @partsupp1::@ps_availqty : !db.int<32>
                %61 = db.cast %60 : !db.int<32> -> !db.decimal<15,2>
                %62 = db.mul %59 : !db.decimal<15,2>,%61 : !db.decimal<15,2>
                %63 = relalg.addattr %57, @aggfmname1({type=!db.decimal<15,2>}) %62
                relalg.return %63 : !relalg.tuple
            }
            %66 = relalg.aggregation @aggr1 %58 [] (%64 : !relalg.tuplestream, %65 : !relalg.tuple) {
                %67 = relalg.aggrfn sum @map2::@aggfmname1 %64 : !db.nullable<!db.decimal<15,2>>
                %68 = relalg.addattr %65, @aggfmname2({type=!db.nullable<!db.decimal<15,2>>}) %67
                relalg.return %68 : !relalg.tuple
            }
            %70 = relalg.map @map3 %66 (%69: !relalg.tuple) {
                %71 = relalg.getattr %69 @aggr1::@aggfmname2 : !db.nullable<!db.decimal<15,2>>
                %72 = db.constant ("0.0001") :!db.decimal<15,4>
                %73 = db.cast %71 : !db.nullable<!db.decimal<15,2>> -> !db.nullable<!db.decimal<15,4>>
                %74 = db.mul %73 : !db.nullable<!db.decimal<15,4>>,%72 : !db.decimal<15,4>
                %75 = relalg.addattr %69, @aggfmname3({type=!db.nullable<!db.decimal<15,4>>}) %74
                relalg.return %75 : !relalg.tuple
            }
            %76 = relalg.getscalar @map3::@aggfmname3 %70 : !db.nullable<!db.decimal<15,4>>
            %77 = db.cast %39 : !db.decimal<15,2> -> !db.nullable<!db.decimal<15,4>>
            %78 = db.compare gt %77 : !db.nullable<!db.decimal<15,4>>,%76 : !db.nullable<!db.decimal<15,4>>
            relalg.return %78 : !db.nullable<i1>
        }
        %79 = relalg.sort %38 [(@aggr::@aggfmname4,desc)]
        %80 = relalg.materialize %79 [@partsupp::@ps_partkey,@aggr::@aggfmname4] => ["ps_partkey","value"] : !db.table
        return %80 : !db.table
    }
}

