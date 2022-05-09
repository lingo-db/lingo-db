--//RUN: run-sql %s %S/../../../../resources/data/ssb | FileCheck %s
--//CHECK: |                           sum  |                        d_year  |                      p_brand1  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                   89380397.00  |                          1992  |                   "MFGR#2239"  |
select sum(lo_revenue), d_year, p_brand1
from lineorder, date, part, supplier
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1= 'MFGR#2239'
and s_region = 'EUROPE'
group by d_year, p_brand1
order by d_year, p_brand1;

