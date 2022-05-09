--//RUN: run-sql %s %S/../../../../resources/data/ssb | FileCheck %s
--//CHECK: |                           sum  |                        d_year  |                      p_brand1  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                   29165996.00  |                          1992  |                    "MFGR#121"  |
--//CHECK: |                   23120066.00  |                          1992  |                   "MFGR#1210"  |
--//CHECK: |                   52982362.00  |                          1992  |                   "MFGR#1211"  |
--//CHECK: |                   30954680.00  |                          1992  |                   "MFGR#1212"  |
--//CHECK: |                   15288453.00  |                          1992  |                   "MFGR#1213"  |
--//CHECK: |                    7655070.00  |                          1992  |                   "MFGR#1214"  |
--//CHECK: |                   22246540.00  |                          1992  |                   "MFGR#1215"  |
--//CHECK: |                   19716439.00  |                          1992  |                   "MFGR#1216"  |
--//CHECK: |                   43666251.00  |                          1992  |                   "MFGR#1217"  |
--//CHECK: |                   22759602.00  |                          1992  |                   "MFGR#1218"  |
--//CHECK: |                   23318799.00  |                          1992  |                   "MFGR#1219"  |
--//CHECK: |                   74056106.00  |                          1992  |                    "MFGR#122"  |
--//CHECK: |                   51050565.00  |                          1992  |                   "MFGR#1220"  |
--//CHECK: |                   38878674.00  |                          1992  |                   "MFGR#1221"  |
--//CHECK: |                   16558051.00  |                          1992  |                   "MFGR#1222"  |
--//CHECK: |                   26690787.00  |                          1992  |                   "MFGR#1223"  |
--//CHECK: |                   76498594.00  |                          1992  |                   "MFGR#1224"  |
--//CHECK: |                   32608903.00  |                          1992  |                   "MFGR#1225"  |
--//CHECK: |                   47636685.00  |                          1992  |                   "MFGR#1226"  |
--//CHECK: |                   27691433.00  |                          1992  |                   "MFGR#1227"  |
--//CHECK: |                   32513490.00  |                          1992  |                   "MFGR#1228"  |
--//CHECK: |                   35514258.00  |                          1992  |                   "MFGR#1229"  |
--//CHECK: |                   17199862.00  |                          1992  |                    "MFGR#123"  |
--//CHECK: |                   24678908.00  |                          1992  |                   "MFGR#1230"  |
--//CHECK: |                   26231337.00  |                          1992  |                   "MFGR#1231"  |
--//CHECK: |                   36330900.00  |                          1992  |                   "MFGR#1232"  |
--//CHECK: |                   24946678.00  |                          1992  |                   "MFGR#1233"  |
--//CHECK: |                   36431683.00  |                          1992  |                   "MFGR#1234"  |
--//CHECK: |                   39368479.00  |                          1992  |                   "MFGR#1235"  |
--//CHECK: |                   44456974.00  |                          1992  |                   "MFGR#1236"  |
--//CHECK: |                   31443810.00  |                          1992  |                   "MFGR#1237"  |
--//CHECK: |                   49003021.00  |                          1992  |                   "MFGR#1238"  |
--//CHECK: |                   31379822.00  |                          1992  |                   "MFGR#1239"  |
--//CHECK: |                   24245603.00  |                          1992  |                    "MFGR#124"  |
--//CHECK: |                   49870826.00  |                          1992  |                   "MFGR#1240"  |
--//CHECK: |                   28194770.00  |                          1992  |                    "MFGR#125"  |
--//CHECK: |                   40503844.00  |                          1992  |                    "MFGR#126"  |
--//CHECK: |                   36027836.00  |                          1992  |                    "MFGR#127"  |
--//CHECK: |                   35881895.00  |                          1992  |                    "MFGR#128"  |
--//CHECK: |                   21732451.00  |                          1992  |                    "MFGR#129"  |
select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_category = 'MFGR#12'
and s_region = 'AMERICA'
group by d_year, p_brand1
order by d_year, p_brand1

