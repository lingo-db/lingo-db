//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                     c_custkey  |                        c_name  |                       revenue  |                     c_acctbal  |                        n_name  |                     c_address  |                       c_phone  |                     c_comment  |
//CHECK: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//CHECK: |                          8242  |          "Customer#000008242"  |                     622786.68  |                       6322.09  |                    "ETHIOPIA"  |"P2n4nJhy,UqSo2s43YfSvYJDZ6lk"  |             "15-792-676-1184"  |"slyly regular packages haggle carefully ironic ideas. courts are furiously. furiously unusual theodolites cajole. i"  |
//CHECK: |                          7714  |          "Customer#000007714"  |                     557400.24  |                       9799.98  |                        "IRAN"  |             "SnnIGB,SkmnWpX3"  |             "20-922-418-6024"  |"arhorses according to the blithely express re"  |
//CHECK: |                         11032  |          "Customer#000011032"  |                     512500.91  |                       8496.93  |              "UNITED KINGDOM"  |"WIKHC7K3Cn7156iNOyfVG3cZ7YqkgsR,Ly"  |             "33-102-772-3533"  |"posits-- furiously ironic accounts are again"  |
//CHECK: |                          2455  |          "Customer#000002455"  |                     481592.37  |                       2070.99  |                     "GERMANY"  |  "RVn1ZSRtLqPlJLIZxvpmsbgC02"  |             "17-946-225-9977"  |"al asymptotes. finally ironic accounts cajole furiously. permanently unusual theodolites aro"  |
//CHECK: |                         12106  |          "Customer#000012106"  |                     479414.15  |                       5342.11  |               "UNITED STATES"  |                "wth3twOmu6vy"  |             "34-905-346-4472"  |"ly after the blithely regular foxes. accounts haggle carefully alongside of the blithely even ideas."  |
//CHECK: |                          8530  |          "Customer#000008530"  |                     457855.90  |                       9734.95  |                     "MOROCCO"  |"GMQyte94oDM7eD7exnkj 4hH9yq3"  |             "25-736-932-5850"  |"slyly asymptotes. quickly final deposits in"  |
//CHECK: |                         13984  |          "Customer#000013984"  |                     446316.43  |                       3482.28  |                        "IRAN"  |               "qZXwuapCHvxbX"  |             "20-981-264-2952"  |"y unusual courts could wake furiously"  |
//CHECK: |                          1966  |          "Customer#000001966"  |                     444058.99  |                       1937.72  |                     "ALGERIA"  |"jPv1 UHra5JLALR5Isci5u0636RoAu7t vH"  |             "10-973-269-8886"  |"the blithely even accounts. final deposits cajole around the blithely final packages. "  |
//CHECK: |                         11026  |          "Customer#000011026"  |                     417913.37  |                       7738.76  |                     "ALGERIA"  |          "XorIktoJOAEJkpNNMx"  |             "10-184-163-4632"  |"ly even dolphins eat along the blithely even instructions. express attainments cajole slyly. busy dolphins in"  |
//CHECK: |                          8501  |          "Customer#000008501"  |                     412797.48  |                       6906.70  |                   "ARGENTINA"  |          "776af4rOa mZ66hczs"  |             "11-317-552-5840"  |"y final deposits after the fluffily even accounts are slyly final, regular"  |
//CHECK: |                          1565  |          "Customer#000001565"  |                     412505.99  |                       1820.03  |                      "BRAZIL"  |"EWQO5Ck,nMuHVQimqL8dLrixRP6QKveXcz9QgorW"  |             "12-402-178-2007"  |"ously regular accounts wake slyly ironic idea"  |
//CHECK: |                         14398  |          "Customer#000014398"  |                     408575.32  |                       -602.24  |               "UNITED STATES"  |"GWRCgIPHajtU21vICVvbJJerFu2cUk"  |             "34-814-111-5424"  |"s. blithely even accounts cajole blithely. even foxes doubt-- "  |
//CHECK: |                          1465  |          "Customer#000001465"  |                     405055.30  |                       9365.93  |                       "INDIA"  |    "tDRaTC7UgFbBX7VF6cVXYQA0"  |             "18-807-487-1074"  |"s lose blithely ironic, regular packages. regular, final foxes haggle c"  |
//CHECK: |                         12595  |          "Customer#000012595"  |                     401402.20  |                         -6.92  |                       "INDIA"  |     "LmeaX5cR,w9NqKugl yRm98"  |             "18-186-132-3352"  |"o the busy accounts. blithely special gifts maintain a"  |
//CHECK: |                           961  |          "Customer#000000961"  |                     401198.14  |                       6963.68  |                       "JAPAN"  |"5,81YDLFuRR47KKzv8GXdmi3zyP37PlPn"  |             "22-989-463-6089"  |"e final requests: busily final accounts believe a"  |
//CHECK: |                         14299  |          "Customer#000014299"  |                     400968.35  |                       6595.97  |                      "RUSSIA"  |           "7lFczTya0iM1bhEWT"  |             "32-156-618-1224"  |" carefully regular requests. quickly ironic accounts against the ru"  |
//CHECK: |                           623  |          "Customer#000000623"  |                     399883.40  |                       7887.60  |                   "INDONESIA"  |"HXiFb9oWlgqZXrJPUCEJ6zZIPxAM4m6"  |             "19-113-202-7085"  |" requests. dolphins above the busily regular dependencies cajole after"  |
//CHECK: |                          9151  |          "Customer#000009151"  |                     396561.99  |                       5691.95  |                        "IRAQ"  |  "7gIdRdaxB91EVdyx8DyPjShpMD"  |             "21-834-147-4906"  |"ajole fluffily. furiously regular accounts are special, silent account"  |
//CHECK: |                         14819  |          "Customer#000014819"  |                     396271.07  |                       7308.39  |                      "FRANCE"  |"w8StIbymUXmLCcUag6sx6LUIp8E3pA,Ux"  |             "16-769-398-7926"  |"ss, final asymptotes use furiously slyly ironic dependencies. special, express dugouts according to the dep"  |
//CHECK: |                         13478  |          "Customer#000013478"  |                     395513.11  |                       -778.11  |                       "KENYA"  |"9VIsvIeZrJpC6OOdYheMC2vdtq8Ai0Rt"  |             "24-983-202-8240"  |"r theodolites. slyly unusual pinto beans sleep fluffily against the asymptotes. quickly r"  |
module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @customer  {table_identifier = "customer"} columns: {c_acctbal => @c_acctbal({type = !db.decimal<15, 2>}), c_address => @c_address({type = !db.string}), c_comment => @c_comment({type = !db.string}), c_custkey => @c_custkey({type = i32}), c_mktsegment => @c_mktsegment({type = !db.string}), c_name => @c_name({type = !db.string}), c_nationkey => @c_nationkey({type = i32}), c_phone => @c_phone({type = !db.string})}
    %1 = relalg.basetable @orders  {table_identifier = "orders"} columns: {o_clerk => @o_clerk({type = !db.string}), o_comment => @o_comment({type = !db.string}), o_custkey => @o_custkey({type = i32}), o_orderdate => @o_orderdate({type = !db.date<day>}), o_orderkey => @o_orderkey({type = i32}), o_orderpriority => @o_orderpriority({type = !db.string}), o_orderstatus => @o_orderstatus({type = !db.char<1>}), o_shippriority => @o_shippriority({type = i32}), o_totalprice => @o_totalprice({type = !db.decimal<15, 2>})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @lineitem  {table_identifier = "lineitem"} columns: {l_comment => @l_comment({type = !db.string}), l_commitdate => @l_commitdate({type = !db.date<day>}), l_discount => @l_discount({type = !db.decimal<15, 2>}), l_extendedprice => @l_extendedprice({type = !db.decimal<15, 2>}), l_linenumber => @l_linenumber({type = i32}), l_linestatus => @l_linestatus({type = !db.char<1>}), l_orderkey => @l_orderkey({type = i32}), l_partkey => @l_partkey({type = i32}), l_quantity => @l_quantity({type = !db.decimal<15, 2>}), l_receiptdate => @l_receiptdate({type = !db.date<day>}), l_returnflag => @l_returnflag({type = !db.char<1>}), l_shipdate => @l_shipdate({type = !db.date<day>}), l_shipinstruct => @l_shipinstruct({type = !db.string}), l_shipmode => @l_shipmode({type = !db.string}), l_suppkey => @l_suppkey({type = i32}), l_tax => @l_tax({type = !db.decimal<15, 2>})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %6 = relalg.crossproduct %4, %5
    %7 = relalg.selection %6 (%arg0: !relalg.tuple){
      %13 = relalg.getcol %arg0 @customer::@c_custkey : i32
      %14 = relalg.getcol %arg0 @orders::@o_custkey : i32
      %15 = db.compare eq %13 : i32, %14 : i32
      %16 = relalg.getcol %arg0 @lineitem::@l_orderkey : i32
      %17 = relalg.getcol %arg0 @orders::@o_orderkey : i32
      %18 = db.compare eq %16 : i32, %17 : i32
      %19 = relalg.getcol %arg0 @orders::@o_orderdate : !db.date<day>
      %20 = db.constant("1993-10-01") : !db.date<day>
      %21 = db.compare gte %19 : !db.date<day>, %20 : !db.date<day>
      %22 = relalg.getcol %arg0 @orders::@o_orderdate : !db.date<day>
      %23 = db.constant("1994-01-01") : !db.date<day>
      %24 = db.compare lt %22 : !db.date<day>, %23 : !db.date<day>
      %25 = relalg.getcol %arg0 @lineitem::@l_returnflag : !db.char<1>
      %26 = db.constant("R") : !db.char<1>
      %27 = db.compare eq %25 : !db.char<1>, %26 : !db.char<1>
      %28 = relalg.getcol %arg0 @customer::@c_nationkey : i32
      %29 = relalg.getcol %arg0 @nation::@n_nationkey : i32
      %30 = db.compare eq %28 : i32, %29 : i32
      %31 = db.and %15, %18, %21, %24, %27, %30 : i1, i1, i1, i1, i1, i1
      relalg.return %31 : i1
    }
    %8 = relalg.map @map0 %7 (%arg0: !relalg.tuple){
      %13 = relalg.getcol %arg0 @lineitem::@l_extendedprice : !db.decimal<15, 2>
      %14 = db.constant(1 : i32) : !db.decimal<15, 2>
      %15 = relalg.getcol %arg0 @lineitem::@l_discount : !db.decimal<15, 2>
      %16 = db.sub %14 : !db.decimal<15, 2>, %15 : !db.decimal<15, 2>
      %17 = db.mul %13 : !db.decimal<15, 2>, %16 : !db.decimal<15, 2>
      %18 = relalg.addcol %arg0, @tmp_attr1({type = !db.decimal<15, 2>}) %17
      relalg.return %18 : !relalg.tuple
    }
    %9 = relalg.aggregation @aggr0 %8 [@customer::@c_custkey,@customer::@c_name,@customer::@c_acctbal,@customer::@c_phone,@nation::@n_name,@customer::@c_address,@customer::@c_comment] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %13 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.decimal<15, 2>
      %14 = relalg.addcol %arg1, @tmp_attr0({type = !db.decimal<15, 2>}) %13
      relalg.return %14 : !relalg.tuple
    }
    %10 = relalg.sort %9 [(@aggr0::@tmp_attr0,desc)]
    %11 = relalg.limit 20 %10
    %12 = relalg.materialize %11 [@customer::@c_custkey,@customer::@c_name,@aggr0::@tmp_attr0,@customer::@c_acctbal,@nation::@n_name,@customer::@c_address,@customer::@c_phone,@customer::@c_comment] => ["c_custkey", "c_name", "revenue", "c_acctbal", "n_name", "c_address", "c_phone", "c_comment"] : !dsa.table
    return %12 : !dsa.table
  }
}

