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
module {
  func @main() -> !db.table {
    %0 = relalg.basetable @part  {table_identifier = "part"} columns: {p_brand => @p_brand({type = !db.string}), p_comment => @p_comment({type = !db.string}), p_container => @p_container({type = !db.string}), p_mfgr => @p_mfgr({type = !db.string}), p_name => @p_name({type = !db.string}), p_partkey => @p_partkey({type = i32}), p_retailprice => @p_retailprice({type = !db.decimal<15, 2>}), p_size => @p_size({type = i32}), p_type => @p_type({type = !db.string})}
    %1 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = i32}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = i32}), ps_suppkey => @ps_suppkey({type = i32}), ps_supplycost => @ps_supplycost({type = !db.decimal<15, 2>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %6 = relalg.crossproduct %4, %5
    %7 = relalg.basetable @region  {table_identifier = "region"} columns: {r_comment => @r_comment({type = !db.nullable<!db.string>}), r_name => @r_name({type = !db.string}), r_regionkey => @r_regionkey({type = i32})}
    %8 = relalg.crossproduct %6, %7
    %9 = relalg.selection %8 (%arg0: !relalg.tuple){
      %13 = relalg.getattr %arg0 @part::@p_partkey : i32
      %14 = relalg.getattr %arg0 @partsupp::@ps_partkey : i32
      %15 = db.compare eq %13 : i32, %14 : i32
      %16 = relalg.getattr %arg0 @supplier::@s_suppkey : i32
      %17 = relalg.getattr %arg0 @partsupp::@ps_suppkey : i32
      %18 = db.compare eq %16 : i32, %17 : i32
      %19 = relalg.getattr %arg0 @part::@p_size : i32
      %20 = db.constant(15 : i32) : i32
      %21 = db.compare eq %19 : i32, %20 : i32
      %22 = relalg.getattr %arg0 @part::@p_type : !db.string
      %23 = db.constant("%BRASS") : !db.string
      %24 = db.compare like %22 : !db.string, %23 : !db.string
      %25 = relalg.getattr %arg0 @supplier::@s_nationkey : i32
      %26 = relalg.getattr %arg0 @nation::@n_nationkey : i32
      %27 = db.compare eq %25 : i32, %26 : i32
      %28 = relalg.getattr %arg0 @nation::@n_regionkey : i32
      %29 = relalg.getattr %arg0 @region::@r_regionkey : i32
      %30 = db.compare eq %28 : i32, %29 : i32
      %31 = relalg.getattr %arg0 @region::@r_name : !db.string
      %32 = db.constant("EUROPE") : !db.string
      %33 = db.compare eq %31 : !db.string, %32 : !db.string
      %34 = relalg.getattr %arg0 @partsupp::@ps_supplycost : !db.decimal<15, 2>
      %35 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = i32}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = i32}), ps_suppkey => @ps_suppkey({type = i32}), ps_supplycost => @ps_supplycost({type = !db.decimal<15, 2>})}
      %36 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
      %37 = relalg.crossproduct %35, %36
      %38 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
      %39 = relalg.crossproduct %37, %38
      %40 = relalg.basetable @region  {table_identifier = "region"} columns: {r_comment => @r_comment({type = !db.nullable<!db.string>}), r_name => @r_name({type = !db.string}), r_regionkey => @r_regionkey({type = i32})}
      %41 = relalg.crossproduct %39, %40
      %42 = relalg.selection %41 (%arg1: !relalg.tuple){
        %47 = relalg.getattr %arg1 @part::@p_partkey : i32
        %48 = relalg.getattr %arg1 @partsupp::@ps_partkey : i32
        %49 = db.compare eq %47 : i32, %48 : i32
        %50 = relalg.getattr %arg1 @supplier::@s_suppkey : i32
        %51 = relalg.getattr %arg1 @partsupp::@ps_suppkey : i32
        %52 = db.compare eq %50 : i32, %51 : i32
        %53 = relalg.getattr %arg1 @supplier::@s_nationkey : i32
        %54 = relalg.getattr %arg1 @nation::@n_nationkey : i32
        %55 = db.compare eq %53 : i32, %54 : i32
        %56 = relalg.getattr %arg1 @nation::@n_regionkey : i32
        %57 = relalg.getattr %arg1 @region::@r_regionkey : i32
        %58 = db.compare eq %56 : i32, %57 : i32
        %59 = relalg.getattr %arg1 @region::@r_name : !db.string
        %60 = db.constant("EUROPE") : !db.string
        %61 = db.compare eq %59 : !db.string, %60 : !db.string
        %62 = db.and %49, %52, %55, %58, %61 : i1, i1, i1, i1, i1
        relalg.return %62 : i1
      }
      %43 = relalg.aggregation @aggr0 %42 [] (%arg1: !relalg.tuplestream,%arg2: !relalg.tuple){
        %47 = relalg.aggrfn min @partsupp::@ps_supplycost %arg1 : !db.nullable<!db.decimal<15, 2>>
        %48 = relalg.addattr %arg2, @tmp_attr0({type = !db.nullable<!db.decimal<15, 2>>}) %47
        relalg.return %48 : !relalg.tuple
      }
      %44 = relalg.getscalar @aggr0::@tmp_attr0 %43 : !db.nullable<!db.decimal<15, 2>>
      %45 = db.compare eq %34 : !db.decimal<15, 2>, %44 : !db.nullable<!db.decimal<15, 2>>
      %46 = db.and %15, %18, %21, %24, %27, %30, %33, %45 : i1, i1, i1, i1, i1, i1, i1, !db.nullable<i1>
      relalg.return %46 : !db.nullable<i1>
    }
    %10 = relalg.sort %9 [(@supplier::@s_acctbal,desc),(@nation::@n_name,asc),(@supplier::@s_name,asc),(@part::@p_partkey,asc)]
    %11 = relalg.limit 100 %10
    %12 = relalg.materialize %11 [@supplier::@s_acctbal,@supplier::@s_name,@nation::@n_name,@part::@p_partkey,@part::@p_mfgr,@supplier::@s_address,@supplier::@s_phone,@supplier::@s_comment] => ["s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr", "s_address", "s_phone", "s_comment"] : !db.table
    return %12 : !db.table
  }
}

