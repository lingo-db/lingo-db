select d_year,s_city,p_brand1,sum(lo_revenue-lo_supplycost) as profit
from lineorder,supplier,customer,part,date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_partkey = p_partkey
and lo_orderdate = d_datekey
and c_region = 1
and s_nation = 24
and (d_year = 1997 or d_year = 1998)
and p_category = 3
group by d_year,s_city,p_brand1;

