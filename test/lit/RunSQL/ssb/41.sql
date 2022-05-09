--//RUN: run-sql %s %S/../../../../resources/data/ssb | FileCheck %s
--//CHECK: |                        d_year  |                      c_nation  |                        profit  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                          1992  |                   "ARGENTINA"  |                  527880802.00  |
--//CHECK: |                          1992  |                      "BRAZIL"  |                  588644387.00  |
--//CHECK: |                          1992  |                      "CANADA"  |                  604043992.00  |
--//CHECK: |                          1992  |                        "PERU"  |                  559686277.00  |
--//CHECK: |                          1992  |               "UNITED STATES"  |                  564004151.00  |
select d_year, c_nation,
sum(lo_revenue - lo_supplycost) as profit
from date, customer, supplier, part, lineorder
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 'AMERICA'
and s_region = 'AMERICA'
and (p_mfgr = 'MFGR#1'
   or p_mfgr = 'MFGR#2')
group by d_year, c_nation
order by d_year, c_nation;

