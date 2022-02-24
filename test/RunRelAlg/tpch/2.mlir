//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                     s_acctbal  |                        s_name  |                        n_name  |                     p_partkey  |                        p_mfgr  |                     s_address  |                       s_phone  |                     s_comment  |
//CHECK: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//CHECK: |                       9828.21  |          "Supplier#000000647"  |              "UNITED KINGDOM"  |                         13120  |              "Manufacturer#5"  |                "x5U7MBZmwfG9"  |             "33-258-202-4782"  |"s the slyly even ideas poach fluffily "  |
//CHECK: |                       9508.37  |          "Supplier#000000070"  |                      "FRANCE"  |                          3563  |              "Manufacturer#1"  |"INWNH2w,OOWgNDq0BRCcBwOMQc6PdFDc4"  |             "16-821-608-1166"  |"ests sleep quickly express ideas. ironic ideas haggle about the final T"  |
//CHECK: |                       9508.37  |          "Supplier#000000070"  |                      "FRANCE"  |                         17268  |              "Manufacturer#4"  |"INWNH2w,OOWgNDq0BRCcBwOMQc6PdFDc4"  |             "16-821-608-1166"  |"ests sleep quickly express ideas. ironic ideas haggle about the final T"  |
//CHECK: |                       9453.01  |          "Supplier#000000802"  |                     "ROMANIA"  |                         10021  |              "Manufacturer#5"  |    ",6HYXb4uaHITmtMBj4Ak57Pd"  |             "29-342-882-6463"  |"gular frets. permanently special multipliers believe blithely alongs"  |
//CHECK: |                       9453.01  |          "Supplier#000000802"  |                     "ROMANIA"  |                         13275  |              "Manufacturer#4"  |    ",6HYXb4uaHITmtMBj4Ak57Pd"  |             "29-342-882-6463"  |"gular frets. permanently special multipliers believe blithely alongs"  |
//CHECK: |                       9192.10  |          "Supplier#000000115"  |              "UNITED KINGDOM"  |                         13325  |              "Manufacturer#1"  |"nJ 2t0f7Ve,wL1,6WzGBJLNBUCKlsV"  |             "33-597-248-1220"  |"es across the carefully express accounts boost caref"  |
//CHECK: |                       9032.15  |          "Supplier#000000959"  |                     "GERMANY"  |                          4958  |              "Manufacturer#4"  |              "8grA EHBnwOZhO"  |             "17-108-642-3106"  |"nding dependencies nag furiou"  |
//CHECK: |                       8702.02  |          "Supplier#000000333"  |                      "RUSSIA"  |                         11810  |              "Manufacturer#3"  |"MaVf XgwPdkiX4nfJGOis8Uu2zKiIZH"  |             "32-508-202-6136"  |"oss the deposits cajole carefully even pinto beans. regular foxes detect alo"  |
//CHECK: |                       8615.50  |          "Supplier#000000812"  |                      "FRANCE"  |                         10551  |              "Manufacturer#2"  |"8qh4tezyScl5bidLAysvutB,,ZI2dn6xP"  |             "16-585-724-6633"  |"y quickly regular deposits? quickly pending packages after the caref"  |
//CHECK: |                       8615.50  |          "Supplier#000000812"  |                      "FRANCE"  |                         13811  |              "Manufacturer#4"  |"8qh4tezyScl5bidLAysvutB,,ZI2dn6xP"  |             "16-585-724-6633"  |"y quickly regular deposits? quickly pending packages after the caref"  |
//CHECK: |                       8488.53  |          "Supplier#000000367"  |                      "RUSSIA"  |                          6854  |              "Manufacturer#4"  |             "E Sv9brQVf43Mzz"  |             "32-458-198-9557"  |"ages. carefully final excuses nag finally. carefully ironic deposits abov"  |
//CHECK: |                       8430.52  |          "Supplier#000000646"  |                      "FRANCE"  |                         11384  |              "Manufacturer#3"  |"IUzsmT,2oBgjhWP2TlXTL6IkJH,4h,1SJRt"  |             "16-601-220-5489"  |"ites among the always final ideas kindle according to the theodolites. notornis in"  |
//CHECK: |                       8271.39  |          "Supplier#000000146"  |                      "RUSSIA"  |                          4637  |              "Manufacturer#5"  | "rBDNgCr04x0sfdzD5,gFOutCiG2"  |             "32-792-619-3155"  |"s cajole quickly special requests. quickly enticing theodolites h"  |
//CHECK: |                       8096.98  |          "Supplier#000000574"  |                      "RUSSIA"  |                           323  |              "Manufacturer#4"  | "2O8 sy9g2mlBOuEjzj0pA2pevk,"  |             "32-866-246-8752"  |"ully after the regular requests. slyly final dependencies wake slyly along the busy deposit"  |
//CHECK: |                       7392.78  |          "Supplier#000000170"  |              "UNITED KINGDOM"  |                          7655  |              "Manufacturer#2"  |            "RtsXQ,SunkA XHy9"  |             "33-803-340-5398"  |"ake carefully across the quickly"  |
//CHECK: |                       7205.20  |          "Supplier#000000477"  |                     "GERMANY"  |                         10956  |              "Manufacturer#5"  |    "VtaNKN5Mqui5yh7j2ldd5waf"  |             "17-180-144-7991"  |" excuses wake express deposits. furiously careful asymptotes according to the carefull"  |
//CHECK: |                       6820.35  |          "Supplier#000000007"  |              "UNITED KINGDOM"  |                         13217  |              "Manufacturer#5"  |       "s,4TicNGB4uO6PaSqNBUq"  |             "33-990-965-2201"  |"s unwind silently furiously regular courts. final requests are deposits. requests wake quietly blit"  |
//CHECK: |                       6721.70  |          "Supplier#000000954"  |                      "FRANCE"  |                          4191  |              "Manufacturer#3"  |            "P3O5p UFz1QsLmZX"  |             "16-537-341-8517"  |"ect blithely blithely final acco"  |
//CHECK: |                       6329.90  |          "Supplier#000000996"  |                     "GERMANY"  |                         10735  |              "Manufacturer#2"  |        "Wx4dQwOAwWjfSCGupfrM"  |             "17-447-811-3282"  |" ironic forges cajole blithely agai"  |
//CHECK: |                       6173.87  |          "Supplier#000000408"  |                      "RUSSIA"  |                         18139  |              "Manufacturer#1"  |  "qcor1u,vJXAokjnL5,dilyYNmh"  |             "32-858-724-2950"  |"blithely pending packages cajole furiously slyly pending notornis. slyly final "  |
//CHECK: |                       5364.99  |          "Supplier#000000785"  |                      "RUSSIA"  |                         13784  |              "Manufacturer#4"  |"W VkHBpQyD3qjQjWGpWicOpmILFehmEdWy67kUGY"  |             "32-297-653-2203"  |" packages boost carefully. express ideas along"  |
//CHECK: |                       5069.27  |          "Supplier#000000328"  |                     "GERMANY"  |                         16327  |              "Manufacturer#1"  |                 "SMm24d WG62"  |             "17-231-513-5721"  |"he unusual ideas. slyly final packages a"  |
//CHECK: |                       4941.88  |          "Supplier#000000321"  |                     "ROMANIA"  |                          7320  |              "Manufacturer#5"  |             "pLngFl5yeMcHyov"  |             "29-573-279-1406"  |  "y final requests impress s"  |
//CHECK: |                       4672.25  |          "Supplier#000000239"  |                      "RUSSIA"  |                         12238  |              "Manufacturer#1"  |"XO101kgHrJagK2FL1U6QCaTE ncCsMbeuTgK6o8"  |             "32-396-654-6826"  |"arls wake furiously deposits. even, regular depen"  |
//CHECK: |                       4586.49  |          "Supplier#000000680"  |                      "RUSSIA"  |                          5679  |              "Manufacturer#3"  |"UhvDfdEfJh,Qbe7VZb8uSGO2TU 0jEa6nXZXE"  |             "32-522-382-1620"  |" the regularly regular dependencies. carefully bold excuses under th"  |
//CHECK: |                       4518.31  |          "Supplier#000000149"  |                      "FRANCE"  |                         18344  |              "Manufacturer#5"  |    "pVyWsjOidpHKp4NfKU4yLeym"  |             "16-660-553-2456"  |"ts detect along the foxes. final Tiresias are. idly pending deposits haggle; even, blithe pin"  |
//CHECK: |                       4315.15  |          "Supplier#000000509"  |                      "FRANCE"  |                         18972  |              "Manufacturer#2"  |                  "SF7dR8V5pK"  |             "16-298-154-3365"  |"ronic orbits are furiously across the requests. quickly express ideas across the special, bold"  |
//CHECK: |                       3526.53  |          "Supplier#000000553"  |                      "FRANCE"  |                          8036  |              "Manufacturer#4"  |                 "a,liVofXbCJ"  |             "16-599-552-3755"  |   "lar dinos nag slyly brave"  |
//CHECK: |                       3526.53  |          "Supplier#000000553"  |                      "FRANCE"  |                         17018  |              "Manufacturer#3"  |                 "a,liVofXbCJ"  |             "16-599-552-3755"  |   "lar dinos nag slyly brave"  |
//CHECK: |                       3294.68  |          "Supplier#000000350"  |                     "GERMANY"  |                          4841  |              "Manufacturer#4"  |              "KIFxV73eovmwhh"  |             "17-113-181-4017"  |"e slyly special foxes. furiously unusual deposits detect carefully carefully ruthless foxes. quick"  |
//CHECK: |                       2972.26  |          "Supplier#000000016"  |                      "RUSSIA"  |                          1015  |              "Manufacturer#4"  |"YjP5C55zHDXL7LalK27zfQnwejdpin4AMpvh"  |             "32-822-502-4215"  |"ously express ideas haggle quickly dugouts? fu"  |
//CHECK: |                       2963.09  |          "Supplier#000000840"  |                     "ROMANIA"  |                          3080  |              "Manufacturer#2"  |                "iYzUIypKhC0Y"  |             "29-781-337-5584"  |"eep blithely regular dependencies. blithely regular platelets sublate alongside o"  |
//CHECK: |                       2221.25  |          "Supplier#000000771"  |                     "ROMANIA"  |                         13981  |              "Manufacturer#2"  |          "lwZ I15rq9kmZXUNhl"  |             "29-986-304-9006"  |"nal foxes eat slyly about the fluffily permanent id"  |
//CHECK: |                       1381.97  |          "Supplier#000000104"  |                      "FRANCE"  |                         18103  |              "Manufacturer#3"  |"Dcl4yGrzqv3OPeRO49bKh78XmQEDR7PBXIs0m"  |             "16-434-972-6922"  |"gular ideas. bravely bold deposits haggle through the carefully final deposits. slyly unusual idea"  |
//CHECK: |                        906.07  |          "Supplier#000000138"  |                     "ROMANIA"  |                          8363  |              "Manufacturer#4"  |"utbplAm g7RmxVfYoNdhcrQGWuzRqPe0qHSwbKw"  |             "29-533-434-6776"  |"ickly unusual requests cajole. accounts above the furiously special excuses "  |
//CHECK: |                        765.69  |          "Supplier#000000799"  |                      "RUSSIA"  |                         11276  |              "Manufacturer#2"  |               "jwFN7ZB3T9sMF"  |             "32-579-339-1495"  |"nusual requests. furiously unusual epitaphs integrate. slyly "  |
//CHECK: |                        727.89  |          "Supplier#000000470"  |                     "ROMANIA"  |                          6213  |              "Manufacturer#3"  |"XckbzsAgBLbUkdfjgJEPjmUMTM8ebSMEvI"  |             "29-165-289-1523"  |"gular excuses. furiously regular excuses sleep slyly caref"  |
//CHECK: |                        683.07  |          "Supplier#000000651"  |                      "RUSSIA"  |                          4888  |              "Manufacturer#4"  |                "oWekiBV6s,1g"  |             "32-181-426-4490"  |"ly regular requests cajole abou"  |
//CHECK: |                        167.56  |          "Supplier#000000290"  |                      "FRANCE"  |                          2037  |              "Manufacturer#1"  |            "6Bk06GVtwZaKqg01"  |             "16-675-286-5102"  |" the theodolites. ironic, ironic deposits above "  |
//CHECK: |                         91.39  |          "Supplier#000000949"  |              "UNITED KINGDOM"  |                          9430  |              "Manufacturer#2"  |"a,UE,6nRVl2fCphkOoetR1ajIzAEJ1Aa1G1HV"  |             "33-332-697-2768"  |"pinto beans. carefully express requests hagg"  |
//CHECK: |                       -314.06  |          "Supplier#000000510"  |                     "ROMANIA"  |                         17242  |              "Manufacturer#4"  |"VmXQl ,vY8JiEseo8Mv4zscvNCfsY"  |             "29-207-852-3454"  |" bold deposits. carefully even d"  |
//CHECK: |                       -820.89  |          "Supplier#000000409"  |                     "GERMANY"  |                          2156  |              "Manufacturer#5"  |"LyXUYFz7aXrvy65kKAbTatGzGS,NDBcdtD"  |             "17-719-517-9836"  |"y final, slow theodolites. furiously regular req"  |
//CHECK: |                       -845.44  |          "Supplier#000000704"  |                     "ROMANIA"  |                          9926  |              "Manufacturer#5"  |"hQvlBqbqqnA5Dgo1BffRBX78tkkRu"  |             "29-300-896-5991"  |  "ctions. carefully sly requ"  |
//CHECK: |                       -942.73  |          "Supplier#000000563"  |                     "GERMANY"  |                          5797  |              "Manufacturer#1"  |             "Rc7U1cRUhYs03JD"  |             "17-108-537-2691"  |"slyly furiously final decoys; silent, special realms poach f"  |
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


