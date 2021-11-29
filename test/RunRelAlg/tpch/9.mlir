//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                        nation  |                        o_year  |                    sum_profit  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                     "ALGERIA"  |                          1998  |                    2321784.90  |
//CHECK: |                     "ALGERIA"  |                          1997  |                    3685016.08  |
//CHECK: |                     "ALGERIA"  |                          1996  |                    4276596.60  |
//CHECK: |                     "ALGERIA"  |                          1995  |                    4418369.67  |
//CHECK: |                     "ALGERIA"  |                          1994  |                    3864849.22  |
//CHECK: |                     "ALGERIA"  |                          1993  |                    3541050.77  |
//CHECK: |                     "ALGERIA"  |                          1992  |                    4310012.59  |
//CHECK: |                   "ARGENTINA"  |                          1998  |                    2685983.20  |
//CHECK: |                   "ARGENTINA"  |                          1997  |                    4242146.98  |
//CHECK: |                   "ARGENTINA"  |                          1996  |                    3907866.14  |
//CHECK: |                   "ARGENTINA"  |                          1995  |                    4605920.67  |
//CHECK: |                   "ARGENTINA"  |                          1994  |                    3542095.44  |
//CHECK: |                   "ARGENTINA"  |                          1993  |                    3949965.16  |
//CHECK: |                   "ARGENTINA"  |                          1992  |                    4521179.61  |
//CHECK: |                      "BRAZIL"  |                          1998  |                    2778729.90  |
//CHECK: |                      "BRAZIL"  |                          1997  |                    4642036.58  |
//CHECK: |                      "BRAZIL"  |                          1996  |                    4530303.72  |
//CHECK: |                      "BRAZIL"  |                          1995  |                    4502344.02  |
//CHECK: |                      "BRAZIL"  |                          1994  |                    4875805.58  |
//CHECK: |                      "BRAZIL"  |                          1993  |                    4687477.72  |
//CHECK: |                      "BRAZIL"  |                          1992  |                    5035199.06  |
//CHECK: |                      "CANADA"  |                          1998  |                    2194508.66  |
//CHECK: |                      "CANADA"  |                          1997  |                    3482197.20  |
//CHECK: |                      "CANADA"  |                          1996  |                    3712230.51  |
//CHECK: |                      "CANADA"  |                          1995  |                    4014814.11  |
//CHECK: |                      "CANADA"  |                          1994  |                    4145303.78  |
//CHECK: |                      "CANADA"  |                          1993  |                    3787068.79  |
//CHECK: |                      "CANADA"  |                          1992  |                    4168008.68  |
//CHECK: |                       "CHINA"  |                          1998  |                    3398577.39  |
//CHECK: |                       "CHINA"  |                          1997  |                    6358958.16  |
//CHECK: |                       "CHINA"  |                          1996  |                    6435157.17  |
//CHECK: |                       "CHINA"  |                          1995  |                    6174775.04  |
//CHECK: |                       "CHINA"  |                          1994  |                    6385749.93  |
//CHECK: |                       "CHINA"  |                          1993  |                    5765033.09  |
//CHECK: |                       "CHINA"  |                          1992  |                    6324033.16  |
//CHECK: |                       "EGYPT"  |                          1998  |                    2333147.89  |
//CHECK: |                       "EGYPT"  |                          1997  |                    3661243.55  |
//CHECK: |                       "EGYPT"  |                          1996  |                    3765370.54  |
//CHECK: |                       "EGYPT"  |                          1995  |                    4094743.58  |
//CHECK: |                       "EGYPT"  |                          1994  |                    3566507.47  |
//CHECK: |                       "EGYPT"  |                          1993  |                    3725283.07  |
//CHECK: |                       "EGYPT"  |                          1992  |                    3373761.72  |
//CHECK: |                    "ETHIOPIA"  |                          1998  |                    1953926.85  |
//CHECK: |                    "ETHIOPIA"  |                          1997  |                    3285785.67  |
//CHECK: |                    "ETHIOPIA"  |                          1996  |                    3525028.08  |
//CHECK: |                    "ETHIOPIA"  |                          1995  |                    3781674.09  |
//CHECK: |                    "ETHIOPIA"  |                          1994  |                    3037408.75  |
//CHECK: |                    "ETHIOPIA"  |                          1993  |                    3008977.66  |
//CHECK: |                    "ETHIOPIA"  |                          1992  |                    2721202.69  |
//CHECK: |                      "FRANCE"  |                          1998  |                    2604373.33  |
//CHECK: |                      "FRANCE"  |                          1997  |                    3982871.24  |
//CHECK: |                      "FRANCE"  |                          1996  |                    3622478.53  |
//CHECK: |                      "FRANCE"  |                          1995  |                    4479938.88  |
//CHECK: |                      "FRANCE"  |                          1994  |                    3531012.40  |
//CHECK: |                      "FRANCE"  |                          1993  |                    4086436.43  |
//CHECK: |                      "FRANCE"  |                          1992  |                    3637791.41  |
//CHECK: |                     "GERMANY"  |                          1998  |                    3291022.61  |
//CHECK: |                     "GERMANY"  |                          1997  |                    5139336.33  |
//CHECK: |                     "GERMANY"  |                          1996  |                    4799809.39  |
//CHECK: |                     "GERMANY"  |                          1995  |                    5405784.81  |
//CHECK: |                     "GERMANY"  |                          1994  |                    4555555.52  |
//CHECK: |                     "GERMANY"  |                          1993  |                    4428194.12  |
//CHECK: |                     "GERMANY"  |                          1992  |                    4656147.50  |
//CHECK: |                       "INDIA"  |                          1998  |                    2591287.60  |
//CHECK: |                       "INDIA"  |                          1997  |                    5159561.70  |
//CHECK: |                       "INDIA"  |                          1996  |                    5307257.20  |
//CHECK: |                       "INDIA"  |                          1995  |                    5148207.76  |
//CHECK: |                       "INDIA"  |                          1994  |                    5164001.05  |
//CHECK: |                       "INDIA"  |                          1993  |                    4321397.61  |
//CHECK: |                       "INDIA"  |                          1992  |                    5297702.59  |
//CHECK: |                   "INDONESIA"  |                          1998  |                    3094899.59  |
//CHECK: |                   "INDONESIA"  |                          1997  |                    5719771.88  |
//CHECK: |                   "INDONESIA"  |                          1996  |                    6037237.56  |
//CHECK: |                   "INDONESIA"  |                          1995  |                    5266782.47  |
//CHECK: |                   "INDONESIA"  |                          1994  |                    5470761.85  |
//CHECK: |                   "INDONESIA"  |                          1993  |                    6189825.57  |
//CHECK: |                   "INDONESIA"  |                          1992  |                    4414622.29  |
//CHECK: |                        "IRAN"  |                          1998  |                    3214863.56  |
//CHECK: |                        "IRAN"  |                          1997  |                    3688048.40  |
//CHECK: |                        "IRAN"  |                          1996  |                    3621648.54  |
//CHECK: |                        "IRAN"  |                          1995  |                    4420782.50  |
//CHECK: |                        "IRAN"  |                          1994  |                    4373983.94  |
//CHECK: |                        "IRAN"  |                          1993  |                    3731301.09  |
//CHECK: |                        "IRAN"  |                          1992  |                    4417132.48  |
//CHECK: |                        "IRAQ"  |                          1998  |                    2338858.87  |
//CHECK: |                        "IRAQ"  |                          1997  |                    3622680.77  |
//CHECK: |                        "IRAQ"  |                          1996  |                    4762290.98  |
//CHECK: |                        "IRAQ"  |                          1995  |                    4558091.71  |
//CHECK: |                        "IRAQ"  |                          1994  |                    4951603.17  |
//CHECK: |                        "IRAQ"  |                          1993  |                    3830077.26  |
//CHECK: |                        "IRAQ"  |                          1992  |                    3938635.59  |
//CHECK: |                       "JAPAN"  |                          1998  |                    1849534.67  |
//CHECK: |                       "JAPAN"  |                          1997  |                    4068687.99  |
//CHECK: |                       "JAPAN"  |                          1996  |                    4044773.90  |
//CHECK: |                       "JAPAN"  |                          1995  |                    4793004.78  |
//CHECK: |                       "JAPAN"  |                          1994  |                    4114716.24  |
//CHECK: |                       "JAPAN"  |                          1993  |                    3614468.06  |
//CHECK: |                       "JAPAN"  |                          1992  |                    4266693.60  |
//CHECK: |                      "JORDAN"  |                          1998  |                    1811487.67  |
//CHECK: |                      "JORDAN"  |                          1997  |                    2951297.27  |
//CHECK: |                      "JORDAN"  |                          1996  |                    3302527.69  |
//CHECK: |                      "JORDAN"  |                          1995  |                    3221813.31  |
//CHECK: |                      "JORDAN"  |                          1994  |                    2417891.51  |
//CHECK: |                      "JORDAN"  |                          1993  |                    3107641.10  |
//CHECK: |                      "JORDAN"  |                          1992  |                    3316378.42  |
//CHECK: |                       "KENYA"  |                          1998  |                    2579074.95  |
//CHECK: |                       "KENYA"  |                          1997  |                    2929193.58  |
//CHECK: |                       "KENYA"  |                          1996  |                    3569128.85  |
//CHECK: |                       "KENYA"  |                          1995  |                    3542888.41  |
//CHECK: |                       "KENYA"  |                          1994  |                    3983094.77  |
//CHECK: |                       "KENYA"  |                          1993  |                    3713988.29  |
//CHECK: |                       "KENYA"  |                          1992  |                    3304641.19  |
//CHECK: |                     "MOROCCO"  |                          1998  |                    1815334.37  |
//CHECK: |                     "MOROCCO"  |                          1997  |                    3693214.13  |
//CHECK: |                     "MOROCCO"  |                          1996  |                    4116175.11  |
//CHECK: |                     "MOROCCO"  |                          1995  |                    3515126.39  |
//CHECK: |                     "MOROCCO"  |                          1994  |                    4003071.26  |
//CHECK: |                     "MOROCCO"  |                          1993  |                    3599198.96  |
//CHECK: |                     "MOROCCO"  |                          1992  |                    3958334.57  |
//CHECK: |                  "MOZAMBIQUE"  |                          1998  |                    1620428.39  |
//CHECK: |                  "MOZAMBIQUE"  |                          1997  |                    2802166.11  |
//CHECK: |                  "MOZAMBIQUE"  |                          1996  |                    2409954.55  |
//CHECK: |                  "MOZAMBIQUE"  |                          1995  |                    2771602.02  |
//CHECK: |                  "MOZAMBIQUE"  |                          1994  |                    2548225.66  |
//CHECK: |                  "MOZAMBIQUE"  |                          1993  |                    2843748.27  |
//CHECK: |                  "MOZAMBIQUE"  |                          1992  |                    2556500.51  |
//CHECK: |                        "PERU"  |                          1998  |                    2036429.88  |
//CHECK: |                        "PERU"  |                          1997  |                    4064141.72  |
//CHECK: |                        "PERU"  |                          1996  |                    4068677.86  |
//CHECK: |                        "PERU"  |                          1995  |                    4657694.07  |
//CHECK: |                        "PERU"  |                          1994  |                    4731958.62  |
//CHECK: |                        "PERU"  |                          1993  |                    4144005.94  |
//CHECK: |                        "PERU"  |                          1992  |                    3754634.20  |
//CHECK: |                     "ROMANIA"  |                          1998  |                    1992773.35  |
//CHECK: |                     "ROMANIA"  |                          1997  |                    2854639.25  |
//CHECK: |                     "ROMANIA"  |                          1996  |                    3139336.73  |
//CHECK: |                     "ROMANIA"  |                          1995  |                    3222152.81  |
//CHECK: |                     "ROMANIA"  |                          1994  |                    3222843.79  |
//CHECK: |                     "ROMANIA"  |                          1993  |                    3488993.37  |
//CHECK: |                     "ROMANIA"  |                          1992  |                    3029273.78  |
//CHECK: |                      "RUSSIA"  |                          1998  |                    2339865.15  |
//CHECK: |                      "RUSSIA"  |                          1997  |                    4153618.71  |
//CHECK: |                      "RUSSIA"  |                          1996  |                    3772066.64  |
//CHECK: |                      "RUSSIA"  |                          1995  |                    4704988.04  |
//CHECK: |                      "RUSSIA"  |                          1994  |                    4479082.00  |
//CHECK: |                      "RUSSIA"  |                          1993  |                    4767719.13  |
//CHECK: |                      "RUSSIA"  |                          1992  |                    4533464.62  |
//CHECK: |                "SAUDI ARABIA"  |                          1998  |                    3386948.27  |
//CHECK: |                "SAUDI ARABIA"  |                          1997  |                    5425979.51  |
//CHECK: |                "SAUDI ARABIA"  |                          1996  |                    5227606.12  |
//CHECK: |                "SAUDI ARABIA"  |                          1995  |                    4506730.75  |
//CHECK: |                "SAUDI ARABIA"  |                          1994  |                    4698657.86  |
//CHECK: |                "SAUDI ARABIA"  |                          1993  |                    5493625.62  |
//CHECK: |                "SAUDI ARABIA"  |                          1992  |                    4573559.08  |
//CHECK: |              "UNITED KINGDOM"  |                          1998  |                    2252021.09  |
//CHECK: |              "UNITED KINGDOM"  |                          1997  |                    4343925.92  |
//CHECK: |              "UNITED KINGDOM"  |                          1996  |                    4189475.57  |
//CHECK: |              "UNITED KINGDOM"  |                          1995  |                    4469569.04  |
//CHECK: |              "UNITED KINGDOM"  |                          1994  |                    4410093.80  |
//CHECK: |              "UNITED KINGDOM"  |                          1993  |                    4054676.34  |
//CHECK: |              "UNITED KINGDOM"  |                          1992  |                    3978688.08  |
//CHECK: |               "UNITED STATES"  |                          1998  |                    2238771.10  |
//CHECK: |               "UNITED STATES"  |                          1997  |                    4135580.82  |
//CHECK: |               "UNITED STATES"  |                          1996  |                    3624012.54  |
//CHECK: |               "UNITED STATES"  |                          1995  |                    3892243.78  |
//CHECK: |               "UNITED STATES"  |                          1994  |                    3289223.44  |
//CHECK: |               "UNITED STATES"  |                          1993  |                    3626169.57  |
//CHECK: |               "UNITED STATES"  |                          1992  |                    3993972.74  |
//CHECK: |                     "VIETNAM"  |                          1998  |                    1924313.13  |
//CHECK: |                     "VIETNAM"  |                          1997  |                    3436194.75  |
//CHECK: |                     "VIETNAM"  |                          1996  |                    4017288.09  |
//CHECK: |                     "VIETNAM"  |                          1995  |                    3644053.40  |
//CHECK: |                     "VIETNAM"  |                          1994  |                    4141276.98  |
//CHECK: |                     "VIETNAM"  |                          1993  |                    2556113.72  |
//CHECK: |                     "VIETNAM"  |                          1992  |                    4090523.78  |
module @querymodule{
    func  @main ()  -> !db.table{
        %1 = relalg.basetable @part { table_identifier="part", rows=20000 , pkey=["p_partkey"]} columns: {p_partkey => @p_partkey({type=!db.int<64>}),
            p_name => @p_name({type=!db.string}),
            p_mfgr => @p_mfgr({type=!db.string}),
            p_brand => @p_brand({type=!db.string}),
            p_type => @p_type({type=!db.string}),
            p_size => @p_size({type=!db.int<32>}),
            p_container => @p_container({type=!db.string}),
            p_retailprice => @p_retailprice({type=!db.decimal<15,2>}),
            p_comment => @p_comment({type=!db.string})
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
        %4 = relalg.basetable @lineitem { table_identifier="lineitem", rows=600572 , pkey=["l_orderkey","l_linenumber"]} columns: {l_orderkey => @l_orderkey({type=!db.int<64>}),
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
        %5 = relalg.crossproduct %3, %4
        %6 = relalg.basetable @partsupp { table_identifier="partsupp", rows=80000 , pkey=["ps_partkey","ps_suppkey"]} columns: {ps_partkey => @ps_partkey({type=!db.int<64>}),
            ps_suppkey => @ps_suppkey({type=!db.int<64>}),
            ps_availqty => @ps_availqty({type=!db.int<32>}),
            ps_supplycost => @ps_supplycost({type=!db.decimal<15,2>}),
            ps_comment => @ps_comment({type=!db.string})
        }
        %7 = relalg.crossproduct %5, %6
        %8 = relalg.basetable @orders { table_identifier="orders", rows=150000 , pkey=["o_orderkey"]} columns: {o_orderkey => @o_orderkey({type=!db.int<64>}),
            o_custkey => @o_custkey({type=!db.int<64>}),
            o_orderstatus => @o_orderstatus({type=!db.string}),
            o_totalprice => @o_totalprice({type=!db.decimal<15,2>}),
            o_orderdate => @o_orderdate({type=!db.date<day>}),
            o_orderpriority => @o_orderpriority({type=!db.string}),
            o_clerk => @o_clerk({type=!db.string}),
            o_shippriority => @o_shippriority({type=!db.int<32>}),
            o_comment => @o_comment({type=!db.string})
        }
        %9 = relalg.crossproduct %7, %8
        %10 = relalg.basetable @nation { table_identifier="nation", rows=25 , pkey=["n_nationkey"]} columns: {n_nationkey => @n_nationkey({type=!db.int<64>}),
            n_name => @n_name({type=!db.string}),
            n_regionkey => @n_regionkey({type=!db.int<64>}),
            n_comment => @n_comment({type=!db.string<nullable>})
        }
        %11 = relalg.crossproduct %9, %10
        %13 = relalg.selection %11(%12: !relalg.tuple) {
            %14 = relalg.getattr %12 @supplier::@s_suppkey : !db.int<64>
            %15 = relalg.getattr %12 @lineitem::@l_suppkey : !db.int<64>
            %16 = db.compare eq %14 : !db.int<64>,%15 : !db.int<64>
            %17 = relalg.getattr %12 @partsupp::@ps_suppkey : !db.int<64>
            %18 = relalg.getattr %12 @lineitem::@l_suppkey : !db.int<64>
            %19 = db.compare eq %17 : !db.int<64>,%18 : !db.int<64>
            %20 = relalg.getattr %12 @partsupp::@ps_partkey : !db.int<64>
            %21 = relalg.getattr %12 @lineitem::@l_partkey : !db.int<64>
            %22 = db.compare eq %20 : !db.int<64>,%21 : !db.int<64>
            %23 = relalg.getattr %12 @part::@p_partkey : !db.int<64>
            %24 = relalg.getattr %12 @lineitem::@l_partkey : !db.int<64>
            %25 = db.compare eq %23 : !db.int<64>,%24 : !db.int<64>
            %26 = relalg.getattr %12 @orders::@o_orderkey : !db.int<64>
            %27 = relalg.getattr %12 @lineitem::@l_orderkey : !db.int<64>
            %28 = db.compare eq %26 : !db.int<64>,%27 : !db.int<64>
            %29 = relalg.getattr %12 @supplier::@s_nationkey : !db.int<64>
            %30 = relalg.getattr %12 @nation::@n_nationkey : !db.int<64>
            %31 = db.compare eq %29 : !db.int<64>,%30 : !db.int<64>
            %32 = relalg.getattr %12 @part::@p_name : !db.string
            %33 = db.constant ("%green%") :!db.string
            %34 = db.compare like %32 : !db.string,%33 : !db.string
            %35 = db.and %16 : !db.bool,%19 : !db.bool,%22 : !db.bool,%25 : !db.bool,%28 : !db.bool,%31 : !db.bool,%34 : !db.bool
            relalg.return %35 : !db.bool
        }
        %37 = relalg.map @map1 %13 (%36: !relalg.tuple) {
            %38 = relalg.getattr %36 @orders::@o_orderdate : !db.date<day>
            %39 = db.date_extract year, %38 : !db.date<day>
            %40 = relalg.addattr %36, @aggfmname1({type=!db.int<64>}) %39
            %41 = relalg.getattr %36 @lineitem::@l_extendedprice : !db.decimal<15,2>
            %42 = db.constant (1) :!db.decimal<15,2>
            %43 = relalg.getattr %36 @lineitem::@l_discount : !db.decimal<15,2>
            %44 = db.sub %42 : !db.decimal<15,2>,%43 : !db.decimal<15,2>
            %45 = db.mul %41 : !db.decimal<15,2>,%44 : !db.decimal<15,2>
            %46 = relalg.getattr %36 @partsupp::@ps_supplycost : !db.decimal<15,2>
            %47 = relalg.getattr %36 @lineitem::@l_quantity : !db.decimal<15,2>
            %48 = db.mul %46 : !db.decimal<15,2>,%47 : !db.decimal<15,2>
            %49 = db.sub %45 : !db.decimal<15,2>,%48 : !db.decimal<15,2>
            %50 = relalg.addattr %40, @aggfmname2({type=!db.decimal<15,2>}) %49
            relalg.return %50 : !relalg.tuple
        }
        %53 = relalg.aggregation @aggr1 %37 [@nation::@n_name,@map1::@aggfmname1] (%51 : !relalg.tuplestream, %52 : !relalg.tuple) {
            %54 = relalg.aggrfn sum @map1::@aggfmname2 %51 : !db.decimal<15,2>
            %55 = relalg.addattr %52, @aggfmname1({type=!db.decimal<15,2>}) %54
            relalg.return %55 : !relalg.tuple
        }
        %56 = relalg.sort %53 [(@nation::@n_name,asc),(@map1::@aggfmname1,desc)]
        %57 = relalg.materialize %56 [@nation::@n_name,@map1::@aggfmname1,@aggr1::@aggfmname1] => ["nation","o_year","sum_profit"] : !db.table
        return %57 : !db.table
    }
}


