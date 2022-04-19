//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                        nation  |                        o_year  |                    sum_profit  |
//CHECK: ----------------------------------------------------------------------------------------------------
//CHECK: |                     "ALGERIA"  |                          1998  |                  2321785.3682  |
//CHECK: |                     "ALGERIA"  |                          1997  |                  3685016.8589  |
//CHECK: |                     "ALGERIA"  |                          1996  |                  4276597.4253  |
//CHECK: |                     "ALGERIA"  |                          1995  |                  4418370.4154  |
//CHECK: |                     "ALGERIA"  |                          1994  |                  3864849.9521  |
//CHECK: |                     "ALGERIA"  |                          1993  |                  3541051.3865  |
//CHECK: |                     "ALGERIA"  |                          1992  |                  4310013.3482  |
//CHECK: |                   "ARGENTINA"  |                          1998  |                  2685983.8005  |
//CHECK: |                   "ARGENTINA"  |                          1997  |                  4242147.8124  |
//CHECK: |                   "ARGENTINA"  |                          1996  |                  3907867.0103  |
//CHECK: |                   "ARGENTINA"  |                          1995  |                  4605921.5011  |
//CHECK: |                   "ARGENTINA"  |                          1994  |                  3542096.1564  |
//CHECK: |                   "ARGENTINA"  |                          1993  |                  3949965.9388  |
//CHECK: |                   "ARGENTINA"  |                          1992  |                  4521180.4695  |
//CHECK: |                      "BRAZIL"  |                          1998  |                  2778730.3931  |
//CHECK: |                      "BRAZIL"  |                          1997  |                  4642037.4687  |
//CHECK: |                      "BRAZIL"  |                          1996  |                  4530304.6034  |
//CHECK: |                      "BRAZIL"  |                          1995  |                  4502344.8657  |
//CHECK: |                      "BRAZIL"  |                          1994  |                  4875806.5015  |
//CHECK: |                      "BRAZIL"  |                          1993  |                  4687478.6531  |
//CHECK: |                      "BRAZIL"  |                          1992  |                  5035200.0464  |
//CHECK: |                      "CANADA"  |                          1998  |                  2194509.0465  |
//CHECK: |                      "CANADA"  |                          1997  |                  3482197.9521  |
//CHECK: |                      "CANADA"  |                          1996  |                  3712231.2814  |
//CHECK: |                      "CANADA"  |                          1995  |                  4014814.8476  |
//CHECK: |                      "CANADA"  |                          1994  |                  4145304.4855  |
//CHECK: |                      "CANADA"  |                          1993  |                  3787069.6045  |
//CHECK: |                      "CANADA"  |                          1992  |                  4168009.4201  |
//CHECK: |                       "CHINA"  |                          1998  |                  3398578.0001  |
//CHECK: |                       "CHINA"  |                          1997  |                  6358959.3338  |
//CHECK: |                       "CHINA"  |                          1996  |                  6435158.3229  |
//CHECK: |                       "CHINA"  |                          1995  |                  6174776.2113  |
//CHECK: |                       "CHINA"  |                          1994  |                  6385751.0812  |
//CHECK: |                       "CHINA"  |                          1993  |                  5765034.1194  |
//CHECK: |                       "CHINA"  |                          1992  |                  6324034.2379  |
//CHECK: |                       "EGYPT"  |                          1998  |                  2333148.3334  |
//CHECK: |                       "EGYPT"  |                          1997  |                  3661244.2731  |
//CHECK: |                       "EGYPT"  |                          1996  |                  3765371.2368  |
//CHECK: |                       "EGYPT"  |                          1995  |                  4094744.2925  |
//CHECK: |                       "EGYPT"  |                          1994  |                  3566508.0818  |
//CHECK: |                       "EGYPT"  |                          1993  |                  3725283.7747  |
//CHECK: |                       "EGYPT"  |                          1992  |                  3373762.3335  |
//CHECK: |                    "ETHIOPIA"  |                          1998  |                  1953927.2682  |
//CHECK: |                    "ETHIOPIA"  |                          1997  |                  3285786.3266  |
//CHECK: |                    "ETHIOPIA"  |                          1996  |                  3525028.7952  |
//CHECK: |                    "ETHIOPIA"  |                          1995  |                  3781674.8911  |
//CHECK: |                    "ETHIOPIA"  |                          1994  |                  3037409.4360  |
//CHECK: |                    "ETHIOPIA"  |                          1993  |                  3008978.2677  |
//CHECK: |                    "ETHIOPIA"  |                          1992  |                  2721203.2355  |
//CHECK: |                      "FRANCE"  |                          1998  |                  2604373.8805  |
//CHECK: |                      "FRANCE"  |                          1997  |                  3982872.0488  |
//CHECK: |                      "FRANCE"  |                          1996  |                  3622479.2413  |
//CHECK: |                      "FRANCE"  |                          1995  |                  4479939.7020  |
//CHECK: |                      "FRANCE"  |                          1994  |                  3531013.1981  |
//CHECK: |                      "FRANCE"  |                          1993  |                  4086437.3102  |
//CHECK: |                      "FRANCE"  |                          1992  |                  3637792.1333  |
//CHECK: |                     "GERMANY"  |                          1998  |                  3291023.2965  |
//CHECK: |                     "GERMANY"  |                          1997  |                  5139337.3443  |
//CHECK: |                     "GERMANY"  |                          1996  |                  4799810.4577  |
//CHECK: |                     "GERMANY"  |                          1995  |                  5405785.7978  |
//CHECK: |                     "GERMANY"  |                          1994  |                  4555556.4592  |
//CHECK: |                     "GERMANY"  |                          1993  |                  4428195.1019  |
//CHECK: |                     "GERMANY"  |                          1992  |                  4656148.4204  |
//CHECK: |                       "INDIA"  |                          1998  |                  2591288.1874  |
//CHECK: |                       "INDIA"  |                          1997  |                  5159562.7033  |
//CHECK: |                       "INDIA"  |                          1996  |                  5307258.3049  |
//CHECK: |                       "INDIA"  |                          1995  |                  5148208.7902  |
//CHECK: |                       "INDIA"  |                          1994  |                  5164001.9582  |
//CHECK: |                       "INDIA"  |                          1993  |                  4321398.4388  |
//CHECK: |                       "INDIA"  |                          1992  |                  5297703.6935  |
//CHECK: |                   "INDONESIA"  |                          1998  |                  3094900.1597  |
//CHECK: |                   "INDONESIA"  |                          1997  |                  5719773.0358  |
//CHECK: |                   "INDONESIA"  |                          1996  |                  6037238.5993  |
//CHECK: |                   "INDONESIA"  |                          1995  |                  5266783.4899  |
//CHECK: |                   "INDONESIA"  |                          1994  |                  5470762.8729  |
//CHECK: |                   "INDONESIA"  |                          1993  |                  6189826.6613  |
//CHECK: |                   "INDONESIA"  |                          1992  |                  4414623.1549  |
//CHECK: |                        "IRAN"  |                          1998  |                  3214864.1209  |
//CHECK: |                        "IRAN"  |                          1997  |                  3688049.0691  |
//CHECK: |                        "IRAN"  |                          1996  |                  3621649.2247  |
//CHECK: |                        "IRAN"  |                          1995  |                  4420783.4205  |
//CHECK: |                        "IRAN"  |                          1994  |                  4373984.6523  |
//CHECK: |                        "IRAN"  |                          1993  |                  3731301.7814  |
//CHECK: |                        "IRAN"  |                          1992  |                  4417133.3662  |
//CHECK: |                        "IRAQ"  |                          1998  |                  2338859.4099  |
//CHECK: |                        "IRAQ"  |                          1997  |                  3622681.5643  |
//CHECK: |                        "IRAQ"  |                          1996  |                  4762291.8722  |
//CHECK: |                        "IRAQ"  |                          1995  |                  4558092.7359  |
//CHECK: |                        "IRAQ"  |                          1994  |                  4951604.1699  |
//CHECK: |                        "IRAQ"  |                          1993  |                  3830077.9911  |
//CHECK: |                        "IRAQ"  |                          1992  |                  3938636.4874  |
//CHECK: |                       "JAPAN"  |                          1998  |                  1849535.0802  |
//CHECK: |                       "JAPAN"  |                          1997  |                  4068688.8537  |
//CHECK: |                       "JAPAN"  |                          1996  |                  4044774.7597  |
//CHECK: |                       "JAPAN"  |                          1995  |                  4793005.8027  |
//CHECK: |                       "JAPAN"  |                          1994  |                  4114717.0568  |
//CHECK: |                       "JAPAN"  |                          1993  |                  3614468.7485  |
//CHECK: |                       "JAPAN"  |                          1992  |                  4266694.4700  |
//CHECK: |                      "JORDAN"  |                          1998  |                  1811488.0719  |
//CHECK: |                      "JORDAN"  |                          1997  |                  2951297.8678  |
//CHECK: |                      "JORDAN"  |                          1996  |                  3302528.3067  |
//CHECK: |                      "JORDAN"  |                          1995  |                  3221813.9990  |
//CHECK: |                      "JORDAN"  |                          1994  |                  2417892.0921  |
//CHECK: |                      "JORDAN"  |                          1993  |                  3107641.7661  |
//CHECK: |                      "JORDAN"  |                          1992  |                  3316379.0585  |
//CHECK: |                       "KENYA"  |                          1998  |                  2579075.4190  |
//CHECK: |                       "KENYA"  |                          1997  |                  2929194.2317  |
//CHECK: |                       "KENYA"  |                          1996  |                  3569129.5619  |
//CHECK: |                       "KENYA"  |                          1995  |                  3542889.1087  |
//CHECK: |                       "KENYA"  |                          1994  |                  3983095.3994  |
//CHECK: |                       "KENYA"  |                          1993  |                  3713988.9708  |
//CHECK: |                       "KENYA"  |                          1992  |                  3304641.8340  |
//CHECK: |                     "MOROCCO"  |                          1998  |                  1815334.8180  |
//CHECK: |                     "MOROCCO"  |                          1997  |                  3693214.8447  |
//CHECK: |                     "MOROCCO"  |                          1996  |                  4116175.9230  |
//CHECK: |                     "MOROCCO"  |                          1995  |                  3515127.1402  |
//CHECK: |                     "MOROCCO"  |                          1994  |                  4003072.1120  |
//CHECK: |                     "MOROCCO"  |                          1993  |                  3599199.6679  |
//CHECK: |                     "MOROCCO"  |                          1992  |                  3958335.4224  |
//CHECK: |                  "MOZAMBIQUE"  |                          1998  |                  1620428.7346  |
//CHECK: |                  "MOZAMBIQUE"  |                          1997  |                  2802166.6473  |
//CHECK: |                  "MOZAMBIQUE"  |                          1996  |                  2409955.1755  |
//CHECK: |                  "MOZAMBIQUE"  |                          1995  |                  2771602.6274  |
//CHECK: |                  "MOZAMBIQUE"  |                          1994  |                  2548226.2158  |
//CHECK: |                  "MOZAMBIQUE"  |                          1993  |                  2843748.9053  |
//CHECK: |                  "MOZAMBIQUE"  |                          1992  |                  2556501.0943  |
//CHECK: |                        "PERU"  |                          1998  |                  2036430.3602  |
//CHECK: |                        "PERU"  |                          1997  |                  4064142.4091  |
//CHECK: |                        "PERU"  |                          1996  |                  4068678.5671  |
//CHECK: |                        "PERU"  |                          1995  |                  4657694.8412  |
//CHECK: |                        "PERU"  |                          1994  |                  4731959.4655  |
//CHECK: |                        "PERU"  |                          1993  |                  4144006.6610  |
//CHECK: |                        "PERU"  |                          1992  |                  3754635.0078  |
//CHECK: |                     "ROMANIA"  |                          1998  |                  1992773.6811  |
//CHECK: |                     "ROMANIA"  |                          1997  |                  2854639.8680  |
//CHECK: |                     "ROMANIA"  |                          1996  |                  3139337.3029  |
//CHECK: |                     "ROMANIA"  |                          1995  |                  3222153.3776  |
//CHECK: |                     "ROMANIA"  |                          1994  |                  3222844.3190  |
//CHECK: |                     "ROMANIA"  |                          1993  |                  3488994.0288  |
//CHECK: |                     "ROMANIA"  |                          1992  |                  3029274.4420  |
//CHECK: |                      "RUSSIA"  |                          1998  |                  2339865.6635  |
//CHECK: |                      "RUSSIA"  |                          1997  |                  4153619.5424  |
//CHECK: |                      "RUSSIA"  |                          1996  |                  3772067.4041  |
//CHECK: |                      "RUSSIA"  |                          1995  |                  4704988.8607  |
//CHECK: |                      "RUSSIA"  |                          1994  |                  4479082.8694  |
//CHECK: |                      "RUSSIA"  |                          1993  |                  4767719.9791  |
//CHECK: |                      "RUSSIA"  |                          1992  |                  4533465.5590  |
//CHECK: |                "SAUDI ARABIA"  |                          1998  |                  3386948.9564  |
//CHECK: |                "SAUDI ARABIA"  |                          1997  |                  5425980.3373  |
//CHECK: |                "SAUDI ARABIA"  |                          1996  |                  5227607.1677  |
//CHECK: |                "SAUDI ARABIA"  |                          1995  |                  4506731.6411  |
//CHECK: |                "SAUDI ARABIA"  |                          1994  |                  4698658.7425  |
//CHECK: |                "SAUDI ARABIA"  |                          1993  |                  5493626.5285  |
//CHECK: |                "SAUDI ARABIA"  |                          1992  |                  4573560.0150  |
//CHECK: |              "UNITED KINGDOM"  |                          1998  |                  2252021.5137  |
//CHECK: |              "UNITED KINGDOM"  |                          1997  |                  4343926.8026  |
//CHECK: |              "UNITED KINGDOM"  |                          1996  |                  4189476.3065  |
//CHECK: |              "UNITED KINGDOM"  |                          1995  |                  4469569.8829  |
//CHECK: |              "UNITED KINGDOM"  |                          1994  |                  4410094.6264  |
//CHECK: |              "UNITED KINGDOM"  |                          1993  |                  4054677.1050  |
//CHECK: |              "UNITED KINGDOM"  |                          1992  |                  3978688.8831  |
//CHECK: |               "UNITED STATES"  |                          1998  |                  2238771.5581  |
//CHECK: |               "UNITED STATES"  |                          1997  |                  4135581.5734  |
//CHECK: |               "UNITED STATES"  |                          1996  |                  3624013.2660  |
//CHECK: |               "UNITED STATES"  |                          1995  |                  3892244.5172  |
//CHECK: |               "UNITED STATES"  |                          1994  |                  3289224.1138  |
//CHECK: |               "UNITED STATES"  |                          1993  |                  3626170.2028  |
//CHECK: |               "UNITED STATES"  |                          1992  |                  3993973.4997  |
//CHECK: |                     "VIETNAM"  |                          1998  |                  1924313.4862  |
//CHECK: |                     "VIETNAM"  |                          1997  |                  3436195.3709  |
//CHECK: |                     "VIETNAM"  |                          1996  |                  4017288.8927  |
//CHECK: |                     "VIETNAM"  |                          1995  |                  3644054.1372  |
//CHECK: |                     "VIETNAM"  |                          1994  |                  4141277.6665  |
//CHECK: |                     "VIETNAM"  |                          1993  |                  2556114.1693  |
//CHECK: |                     "VIETNAM"  |                          1992  |                  4090524.4905  |
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
    %12 = relalg.map @map0 %11 computes : [@tmp_attr1({type = !db.decimal<15, 4>}),@tmp_attr0({type = i64})] (%arg0: !relalg.tuple){
      %16 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %17 = db.constant(1 : i32) : !db.decimal<15, 2>
      %18 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %19 = db.sub %17 : !db.decimal<15, 2>, %18 : !db.decimal<15, 2>
      %20 = db.mul %16 : !db.decimal<15, 2>, %19 : !db.decimal<15, 2>
      %21 = relalg.getcol %arg0 @partsupp::@ps_supplycost : !db.decimal<15, 2>
      %22 = relalg.getcol %arg0 @lineitem::@l_quantity : !db.decimal<15, 2>
      %23 = db.mul %21 : !db.decimal<15, 2>, %22 : !db.decimal<15, 2>
      %24 = db.sub %20 : !db.decimal<15, 4>, %23 : !db.decimal<15, 4>
      %25 = db.constant("year") : !db.char<4>
      %26 = relalg.getcol %arg0 @orders::@o_orderdate : !db.date<day>
      %27 = db.runtime_call "ExtractFromDate"(%25, %26) : (!db.char<4>, !db.date<day>) -> i64
      relalg.return %24, %27 : !db.decimal<15, 4>, i64
    }
    %13 = relalg.aggregation @aggr0 %12 [@nation::@n_name,@map0::@tmp_attr0] computes : [@tmp_attr2({type = !db.decimal<15, 4>})] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %16 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.decimal<15, 4>
      relalg.return %16 : !db.decimal<15, 4>
    }
    %14 = relalg.sort %13 [(@nation::@n_name,asc),(@map0::@tmp_attr0,desc)]
    %15 = relalg.materialize %14 [@nation::@n_name,@map0::@tmp_attr0,@aggr0::@tmp_attr2] => ["nation", "o_year", "sum_profit"] : !dsa.table
    return %15 : !dsa.table
  }
}

