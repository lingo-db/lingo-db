--//RUN: run-sql %s %S/../../../../resources/data/ssb | FileCheck %s
--//CHECK: |                           sum  |                        d_year  |                      p_brand1  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                   28235270.00  |                          1992  |                   "MFGR#2221"  |
--//CHECK: |                   64071827.00  |                          1992  |                   "MFGR#2222"  |
--//CHECK: |                   48591160.00  |                          1992  |                   "MFGR#2223"  |
--//CHECK: |                   20416501.00  |                          1992  |                   "MFGR#2224"  |
--//CHECK: |                   74950776.00  |                          1992  |                   "MFGR#2225"  |
--//CHECK: |                   60628045.00  |                          1992  |                   "MFGR#2226"  |
--//CHECK: |                   39273349.00  |                          1992  |                   "MFGR#2227"  |
--//CHECK: |                   66658087.00  |                          1992  |                   "MFGR#2228"  |
select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1 between 'MFGR#2221'
and 'MFGR#2228'
and s_region = 'ASIA'
group by d_year, p_brand1
order by d_year, p_brand1;

