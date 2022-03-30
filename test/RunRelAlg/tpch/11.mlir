//RUN: db-run-query %s %S/../../../resources/data/tpch | FileCheck %s
//CHECK: |                    ps_partkey  |                         value  |
//CHECK: -------------------------------------------------------------------
//CHECK: |                         12098  |                   16227681.21  |
//CHECK: |                          5134  |                   15709338.52  |
//CHECK: |                         13334  |                   15023662.41  |
//CHECK: |                         17052  |                   14351644.20  |
//CHECK: |                          3452  |                   14070870.14  |
//CHECK: |                         12552  |                   13332469.18  |
//CHECK: |                          1084  |                   13170428.29  |
//CHECK: |                          5797  |                   13038622.72  |
//CHECK: |                         12633  |                   12892561.61  |
//CHECK: |                           403  |                   12856217.34  |
//CHECK: |                          1833  |                   12024581.72  |
//CHECK: |                          2084  |                   11502875.36  |
//CHECK: |                         17349  |                   11354213.05  |
//CHECK: |                         18427  |                   11282385.24  |
//CHECK: |                          2860  |                   11262529.95  |
//CHECK: |                         17852  |                   10934711.93  |
//CHECK: |                          9871  |                   10889253.68  |
//CHECK: |                         12231  |                   10841131.39  |
//CHECK: |                          6366  |                   10759786.81  |
//CHECK: |                         12146  |                   10257362.66  |
//CHECK: |                          5043  |                   10226395.88  |
//CHECK: |                         12969  |                   10125777.93  |
//CHECK: |                          1504  |                   10004397.08  |
//CHECK: |                         14327  |                    9981697.08  |
//CHECK: |                           134  |                    9965150.66  |
//CHECK: |                          6860  |                    9805871.26  |
//CHECK: |                         10624  |                    9776138.40  |
//CHECK: |                         15819  |                    9775705.31  |
//CHECK: |                          3293  |                    9674928.12  |
//CHECK: |                         19865  |                    9653766.83  |
//CHECK: |                          8870  |                    9648981.87  |
//CHECK: |                         15778  |                    9636332.82  |
//CHECK: |                         12360  |                    9635023.92  |
//CHECK: |                         14389  |                    9475588.34  |
//CHECK: |                          3257  |                    9451029.24  |
//CHECK: |                          9476  |                    9435207.28  |
//CHECK: |                         19629  |                    9391236.40  |
//CHECK: |                          7179  |                    9386222.25  |
//CHECK: |                         15723  |                    9383900.80  |
//CHECK: |                          4054  |                    9313810.02  |
//CHECK: |                          2380  |                    9307751.22  |
//CHECK: |                         19084  |                    9302916.80  |
//CHECK: |                          4703  |                    9280804.80  |
//CHECK: |                         18791  |                    9267017.97  |
//CHECK: |                         19994  |                    9235972.92  |
//CHECK: |                          9149  |                    9121803.90  |
//CHECK: |                         15118  |                    9120819.50  |
//CHECK: |                          6116  |                    9079369.20  |
//CHECK: |                          7052  |                    9077468.92  |
//CHECK: |                         14147  |                    9069193.78  |
//CHECK: |                          7305  |                    9035228.53  |
//CHECK: |                          9130  |                    9024379.25  |
//CHECK: |                         16698  |                    8991337.95  |
//CHECK: |                          1553  |                    8977226.10  |
//CHECK: |                         16777  |                    8961355.62  |
//CHECK: |                          1402  |                    8953779.12  |
//CHECK: |                         18963  |                    8934063.40  |
//CHECK: |                          8358  |                    8930611.48  |
//CHECK: |                         17547  |                    8860117.00  |
//CHECK: |                          5128  |                    8844222.75  |
//CHECK: |                         17063  |                    8840649.60  |
//CHECK: |                         15490  |                    8833581.40  |
//CHECK: |                         14761  |                    8817240.56  |
//CHECK: |                         19601  |                    8791341.02  |
//CHECK: |                         16160  |                    8740262.76  |
//CHECK: |                         13597  |                    8702669.82  |
//CHECK: |                         13653  |                    8693170.16  |
//CHECK: |                         16383  |                    8691505.92  |
//CHECK: |                           325  |                    8667741.28  |
//CHECK: |                          8879  |                    8667584.38  |
//CHECK: |                         10564  |                    8667098.22  |
//CHECK: |                         17429  |                    8661827.90  |
//CHECK: |                         17403  |                    8643350.30  |
//CHECK: |                         18294  |                    8616583.43  |
//CHECK: |                          4181  |                    8592684.66  |
//CHECK: |                         13008  |                    8567480.64  |
//CHECK: |                         13211  |                    8537000.01  |
//CHECK: |                          1884  |                    8532644.34  |
//CHECK: |                         11101  |                    8530945.32  |
//CHECK: |                         11562  |                    8528028.57  |
//CHECK: |                         15878  |                    8523591.84  |
//CHECK: |                           834  |                    8522135.27  |
//CHECK: |                          2423  |                    8517902.85  |
//CHECK: |                         15383  |                    8513433.11  |
//CHECK: |                         18119  |                    8507611.80  |
//CHECK: |                          7389  |                    8506099.20  |
//CHECK: |                          5016  |                    8489784.15  |
//CHECK: |                         17473  |                    8444766.24  |
//CHECK: |                          6669  |                    8428618.46  |
//CHECK: |                           384  |                    8418472.27  |
//CHECK: |                         12052  |                    8411519.28  |
//CHECK: |                         17562  |                    8409022.83  |
//CHECK: |                          8128  |                    8379149.47  |
//CHECK: |                         13813  |                    8374830.84  |
//CHECK: |                         12800  |                    8318626.78  |
//CHECK: |                         10887  |                    8315019.36  |
//CHECK: |                          1644  |                    8285453.08  |
//CHECK: |                         16638  |                    8274568.00  |
//CHECK: |                          1394  |                    8255140.60  |
//CHECK: |                          7219  |                    8254985.30  |
//CHECK: |                           ...  |                           ...  |
//CHECK: |                           929  |                    1121383.92  |
//CHECK: |                         11599  |                    1119307.27  |
//CHECK: |                          3765  |                    1119093.50  |
//CHECK: |                         17635  |                    1118420.16  |
//CHECK: |                          7119  |                    1118285.08  |
//CHECK: |                         15121  |                    1117715.34  |
//CHECK: |                         11858  |                    1116963.54  |
//CHECK: |                         16963  |                    1116929.45  |
//CHECK: |                         16356  |                    1113648.98  |
//CHECK: |                          6924  |                    1112198.40  |
//CHECK: |                         16223  |                    1111257.00  |
//CHECK: |                         18091  |                    1110043.02  |
//CHECK: |                         12628  |                    1108954.80  |
//CHECK: |                         16043  |                    1108831.05  |
//CHECK: |                          9402  |                    1108290.48  |
//CHECK: |                           708  |                    1107084.00  |
//CHECK: |                          4078  |                    1105993.96  |
//CHECK: |                         17593  |                    1104713.40  |
//CHECK: |                         12776  |                    1104362.59  |
//CHECK: |                          7583  |                    1102813.53  |
//CHECK: |                         14619  |                    1102675.80  |
//CHECK: |                          8842  |                    1100110.26  |
//CHECK: |                          4196  |                    1099726.55  |
//CHECK: |                          2019  |                    1098178.64  |
//CHECK: |                          6863  |                    1097246.36  |
//CHECK: |                          6489  |                    1096503.07  |
//CHECK: |                          2459  |                    1094813.04  |
//CHECK: |                         11964  |                    1094485.02  |
//CHECK: |                          3236  |                    1093969.80  |
//CHECK: |                         17647  |                    1093809.15  |
//CHECK: |                         17648  |                    1093114.62  |
//CHECK: |                           119  |                    1092687.48  |
//CHECK: |                          9626  |                    1092080.00  |
//CHECK: |                          9124  |                    1091569.68  |
//CHECK: |                         13175  |                    1089851.76  |
//CHECK: |                          2532  |                    1088706.35  |
//CHECK: |                         16083  |                    1088295.39  |
//CHECK: |                          8874  |                    1086011.34  |
//CHECK: |                         12872  |                    1082970.30  |
//CHECK: |                         19821  |                    1082520.84  |
//CHECK: |                          4800  |                    1080389.70  |
//CHECK: |                         18696  |                    1079685.36  |
//CHECK: |                         19545  |                    1079184.33  |
//CHECK: |                         13120  |                    1077742.28  |
//CHECK: |                         10588  |                    1076203.83  |
//CHECK: |                         17696  |                    1075092.72  |
//CHECK: |                         14651  |                    1073222.23  |
//CHECK: |                           903  |                    1071146.76  |
//CHECK: |                          5858  |                    1070259.48  |
//CHECK: |                          8302  |                    1069504.80  |
//CHECK: |                         18728  |                    1069225.51  |
//CHECK: |                         18026  |                    1068569.00  |
//CHECK: |                         19383  |                    1066907.58  |
//CHECK: |                         18690  |                    1065930.90  |
//CHECK: |                          5924  |                    1065143.12  |
//CHECK: |                          4880  |                    1065011.75  |
//CHECK: |                         12439  |                    1064381.19  |
//CHECK: |                         16529  |                    1062371.70  |
//CHECK: |                         19653  |                    1057683.56  |
//CHECK: |                          3136  |                    1056810.44  |
//CHECK: |                         18932  |                    1056193.65  |
//CHECK: |                          2124  |                    1054160.52  |
//CHECK: |                         16851  |                    1052646.84  |
//CHECK: |                         10123  |                    1051624.00  |
//CHECK: |                          5618  |                    1048447.93  |
//CHECK: |                         19851  |                    1045187.85  |
//CHECK: |                         16278  |                    1044808.38  |
//CHECK: |                         11479  |                    1044276.22  |
//CHECK: |                         13263  |                    1042046.20  |
//CHECK: |                          6041  |                    1041123.38  |
//CHECK: |                          7193  |                    1040455.32  |
//CHECK: |                         19408  |                    1039430.01  |
//CHECK: |                         11260  |                    1036828.52  |
//CHECK: |                          5179  |                    1035633.44  |
//CHECK: |                          1331  |                    1034398.00  |
//CHECK: |                          7706  |                    1034249.40  |
//CHECK: |                          8436  |                    1033549.35  |
//CHECK: |                          1801  |                    1031886.00  |
//CHECK: |                          4170  |                    1031642.90  |
//CHECK: |                         11827  |                    1031139.39  |
//CHECK: |                         17114  |                    1027985.88  |
//CHECK: |                         18278  |                    1026583.11  |
//CHECK: |                          1995  |                    1025165.68  |
//CHECK: |                          7667  |                    1022980.15  |
//CHECK: |                          6559  |                    1021635.45  |
//CHECK: |                         17488  |                    1021612.13  |
//CHECK: |                         16059  |                    1019781.19  |
//CHECK: |                          7633  |                    1018782.57  |
//CHECK: |                         10032  |                    1016809.50  |
//CHECK: |                          2899  |                    1016438.76  |
//CHECK: |                         14628  |                    1016033.20  |
//CHECK: |                         10126  |                    1015846.78  |
//CHECK: |                          3884  |                    1014413.50  |
//CHECK: |                         16913  |                    1013604.40  |
//CHECK: |                         18644  |                    1010288.10  |
//CHECK: |                         19870  |                    1007919.36  |
//CHECK: |                         18564  |                    1007416.20  |
//CHECK: |                         10179  |                    1004920.00  |
//CHECK: |                           883  |                    1004650.68  |
//CHECK: |                          3627  |                    1004461.04  |
module {
  func @main() -> !dsa.table {
    %0 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = i32}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = i32}), ps_suppkey => @ps_suppkey({type = i32}), ps_supplycost => @ps_supplycost({type = !db.decimal<15, 2>})}
    %1 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
    %2 = relalg.crossproduct %0, %1
    %3 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
    %4 = relalg.crossproduct %2, %3
    %5 = relalg.selection %4 (%arg0: !relalg.tuple){
      %11 = relalg.getcol %arg0 @partsupp::@ps_suppkey : i32
      %12 = relalg.getcol %arg0 @supplier::@s_suppkey : i32
      %13 = db.compare eq %11 : i32, %12 : i32
      %14 = relalg.getcol %arg0 @supplier::@s_nationkey : i32
      %15 = relalg.getcol %arg0 @nation::@n_nationkey : i32
      %16 = db.compare eq %14 : i32, %15 : i32
      %17 = relalg.getcol %arg0 @nation::@n_name : !db.string
      %18 = db.constant("GERMANY") : !db.string
      %19 = db.compare eq %17 : !db.string, %18 : !db.string
      %20 = db.and %13, %16, %19 : i1, i1, i1
      relalg.return %20 : i1
    }
    %6 = relalg.map @map0 %5 (%arg0: !relalg.tuple){
      %11 = relalg.getcol %arg0 @partsupp::@ps_supplycost : !db.decimal<15, 2>
      %12 = relalg.getcol %arg0 @partsupp::@ps_availqty : i32
      %13 = db.cast %12 : i32 -> !db.decimal<15, 2>
      %14 = db.mul %11 : !db.decimal<15, 2>, %13 : !db.decimal<15, 2>
      %15 = relalg.addcol %arg0, @tmp_attr3({type = !db.decimal<15, 2>}) %14
      %16 = relalg.getcol %15 @partsupp::@ps_supplycost : !db.decimal<15, 2>
      %17 = relalg.getcol %15 @partsupp::@ps_availqty : i32
      %18 = db.cast %17 : i32 -> !db.decimal<15, 2>
      %19 = db.mul %16 : !db.decimal<15, 2>, %18 : !db.decimal<15, 2>
      %20 = relalg.addcol %15, @tmp_attr1({type = !db.decimal<15, 2>}) %19
      relalg.return %20 : !relalg.tuple
    }
    %7 = relalg.aggregation @aggr0 %6 [@partsupp::@ps_partkey] (%arg0: !relalg.tuplestream,%arg1: !relalg.tuple){
      %11 = relalg.aggrfn sum @map0::@tmp_attr3 %arg0 : !db.decimal<15, 2>
      %12 = relalg.addcol %arg1, @tmp_attr2({type = !db.decimal<15, 2>}) %11
      %13 = relalg.aggrfn sum @map0::@tmp_attr1 %arg0 : !db.decimal<15, 2>
      %14 = relalg.addcol %12, @tmp_attr0({type = !db.decimal<15, 2>}) %13
      relalg.return %14 : !relalg.tuple
    }
    %8 = relalg.selection %7 (%arg0: !relalg.tuple){
      %11 = relalg.getcol %arg0 @aggr0::@tmp_attr2 : !db.decimal<15, 2>
      %12 = relalg.basetable @partsupp  {table_identifier = "partsupp"} columns: {ps_availqty => @ps_availqty({type = i32}), ps_comment => @ps_comment({type = !db.string}), ps_partkey => @ps_partkey({type = i32}), ps_suppkey => @ps_suppkey({type = i32}), ps_supplycost => @ps_supplycost({type = !db.decimal<15, 2>})}
      %13 = relalg.basetable @supplier  {table_identifier = "supplier"} columns: {s_acctbal => @s_acctbal({type = !db.decimal<15, 2>}), s_address => @s_address({type = !db.string}), s_comment => @s_comment({type = !db.string}), s_name => @s_name({type = !db.string}), s_nationkey => @s_nationkey({type = i32}), s_phone => @s_phone({type = !db.string}), s_suppkey => @s_suppkey({type = i32})}
      %14 = relalg.crossproduct %12, %13
      %15 = relalg.basetable @nation  {table_identifier = "nation"} columns: {n_comment => @n_comment({type = !db.nullable<!db.string>}), n_name => @n_name({type = !db.string}), n_nationkey => @n_nationkey({type = i32}), n_regionkey => @n_regionkey({type = i32})}
      %16 = relalg.crossproduct %14, %15
      %17 = relalg.selection %16 (%arg1: !relalg.tuple){
        %24 = relalg.getcol %arg1 @partsupp::@ps_suppkey : i32
        %25 = relalg.getcol %arg1 @supplier::@s_suppkey : i32
        %26 = db.compare eq %24 : i32, %25 : i32
        %27 = relalg.getcol %arg1 @supplier::@s_nationkey : i32
        %28 = relalg.getcol %arg1 @nation::@n_nationkey : i32
        %29 = db.compare eq %27 : i32, %28 : i32
        %30 = relalg.getcol %arg1 @nation::@n_name : !db.string
        %31 = db.constant("GERMANY") : !db.string
        %32 = db.compare eq %30 : !db.string, %31 : !db.string
        %33 = db.and %26, %29, %32 : i1, i1, i1
        relalg.return %33 : i1
      }
      %18 = relalg.map @map1 %17 (%arg1: !relalg.tuple){
        %24 = relalg.getcol %arg1 @partsupp::@ps_supplycost : !db.decimal<15, 2>
        %25 = relalg.getcol %arg1 @partsupp::@ps_availqty : i32
        %26 = db.cast %25 : i32 -> !db.decimal<15, 2>
        %27 = db.mul %24 : !db.decimal<15, 2>, %26 : !db.decimal<15, 2>
        %28 = relalg.addcol %arg1, @tmp_attr5({type = !db.decimal<15, 2>}) %27
        relalg.return %28 : !relalg.tuple
      }
      %19 = relalg.aggregation @aggr1 %18 [] (%arg1: !relalg.tuplestream,%arg2: !relalg.tuple){
        %24 = relalg.aggrfn sum @map1::@tmp_attr5 %arg1 : !db.nullable<!db.decimal<15, 2>>
        %25 = relalg.addcol %arg2, @tmp_attr4({type = !db.nullable<!db.decimal<15, 2>>}) %24
        relalg.return %25 : !relalg.tuple
      }
      %20 = relalg.map @map2 %19 (%arg1: !relalg.tuple){
        %24 = relalg.getcol %arg1 @aggr1::@tmp_attr4 : !db.nullable<!db.decimal<15, 2>>
        %25 = db.constant("0.0001") : !db.decimal<15, 4>
        %26 = db.cast %24 : !db.nullable<!db.decimal<15, 2>> -> !db.nullable<!db.decimal<15, 4>>
        %27 = db.mul %26 : !db.nullable<!db.decimal<15, 4>>, %25 : !db.decimal<15, 4>
        %28 = relalg.addcol %arg1, @tmp_attr6({type = !db.nullable<!db.decimal<15, 4>>}) %27
        relalg.return %28 : !relalg.tuple
      }
      %21 = relalg.getscalar @map2::@tmp_attr6 %20 : !db.nullable<!db.decimal<15, 4>>
      %22 = db.cast %11 : !db.decimal<15, 2> -> !db.decimal<15, 4>
      %23 = db.compare gt %22 : !db.decimal<15, 4>, %21 : !db.nullable<!db.decimal<15, 4>>
      relalg.return %23 : !db.nullable<i1>
    }
    %9 = relalg.sort %8 [(@aggr0::@tmp_attr0,desc)]
    %10 = relalg.materialize %9 [@partsupp::@ps_partkey,@aggr0::@tmp_attr0] => ["ps_partkey", "value"] : !dsa.table
    return %10 : !dsa.table
  }
}

