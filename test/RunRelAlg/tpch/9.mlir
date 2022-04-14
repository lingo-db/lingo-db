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
module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @part  {table_identifier = "part"} columns: {p_brand => @p_brand({type = !db.string}), p_comment => @p_comment({type = !db.string}), p_container => @p_container({type = !db.string}), p_mfgr => @p_mfgr({type = !db.string}), p_name => @p_name({type = !db.string}), p_partkey => @p_partkey({type = i32}), p_retailprice => @p_retailprice({type = !db.decimal<15, 2>}), p_size => @p_size({type = i32}), p_type => @p_type({type = !db.string})}
    %1 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = i32}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = i32}), ps_suppkey => @ps_suppkey({type = i32}), ps_supplycost => @ps_supplycost({type = !db.decimal<15, 2>})}
    %6 = relalg.crossproduct %4, %5
    %7 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %8 = relalg.crossproduct %6, %7
    %9 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %10 = relalg.crossproduct %8, %9
    %11 = relalg.selection %10 (%arg0: !relalg.tuple){
      %16 = relalg.getcol %arg0 @supplier::@s_suppkey : i32
      %17 = relalg.getcol %arg0 @lineitem::@l_suppkey : i32
      %18 = db.compare eq %16 : i32, %17 : i32
      %19 = relalg.getcol %arg0 @partsupp::@ps_suppkey : i32
      %20 = relalg.getcol %arg0 @lineitem::@l_suppkey : i32
      %21 = db.compare eq %19 : i32, %20 : i32
      %22 = relalg.getcol %arg0 @partsupp::@ps_partkey : i32
      %23 = relalg.getcol %arg0 @lineitem::@l_partkey : i32
      %24 = db.compare eq %22 : i32, %23 : i32
      %25 = relalg.getcol %arg0 @part::@p_partkey : i32
      %26 = relalg.getcol %arg0 @lineitem::@l_partkey : i32
      %27 = db.compare eq %25 : i32, %26 : i32
      %28 = relalg.getcol %arg0 @orders::@o_orderkey : i32
      %29 = relalg.getcol %arg0 @lineitem::@l_orderkey : i32
      %30 = db.compare eq %28 : i32, %29 : i32
      %31 = relalg.getcol %arg0 @supplier::@s_nationkey : i32
      %32 = relalg.getcol %arg0 @nation::@n_nationkey : i32
      %33 = db.compare eq %31 : i32, %32 : i32
      %34 = relalg.getcol %arg0 @part::@p_name : !db.string
      %35 = db.constant("%green%") : !db.string
      %36 = db.runtime_call "Like"(%34, %35) : (!db.string, !db.string) -> i1
      %37 = db.and %18, %21, %24, %27, %30, %33, %36 : i1, i1, i1, i1, i1, i1, i1
      relalg.return %37 : i1
    }
    %12 = relalg.map @map0 %11 computes : [@tmp_attr1({type = !db.decimal<15, 2>}),@tmp_attr0({type = i64})] (%arg0: !relalg.tuple){
      %16 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %17 = db.constant(1 : i32) : !db.decimal<15, 2>
      %18 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %19 = db.sub %17 : !db.decimal<15, 2>, %18 : !db.decimal<15, 2>
      %20 = db.mul %16 : !db.decimal<15, 2>, %19 : !db.decimal<15, 2>
      %21 = relalg.getcol %arg0 @partsupp::@ps_supplycost : !db.decimal<15, 2>
      %22 = relalg.getcol %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %23 = db.mul %21 : !db.decimal<15, 2>, %22 : !db.decimal<15, 2>
      %24 = db.sub %20 : !db.decimal<15, 2>, %23 : !db.decimal<15, 2>
      %25 = db.constant("year") : !db.char<4>
      %26 = relalg.getcol %arg0 @orders::@o_orderdate : !db.date<day>
      %27 = db.runtime_call "ExtractFromDate"(%25, %26) : (!db.char<4>, !db.date<day>) -> i64
      relalg.return %24, %27 : !db.decimal<15, 2>, i64
    }
    %13 = relalg.aggregation @aggr0 %12 [@nation::@n_name,@map0::@tmp_attr0] computes : [@tmp_attr2({type = !db.decimal<15, 2>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %16 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.decimal<15, 2>
      relalg.return %16 : !db.decimal<15, 2>
    }
    %14 = relalg.sort %13 [(@nation::@n_name,asc),(@map0::@tmp_attr0,desc)]
    %15 = relalg.materialize %14 [@nation::@n_name,@map0::@tmp_attr0,@aggr0::@tmp_attr2] => ["nation", "o_year", "sum_profit"] : !dsa.table
    return %15 : !dsa.table
  }
}

