select d_year,s_nation,p_category,sum(lo_revenue-lo_supplycost) as profit
from lineorder,customer,supplier,part,date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 1
and s_region = 1
and (d_year = 1997 or d_year = 1998)
and (p_mfgr = 0 or p_mfgr = 1)
group by d_year,s_nation, p_category;

