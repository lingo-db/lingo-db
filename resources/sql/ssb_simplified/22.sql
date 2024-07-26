select sum(lo_revenue),d_year,p_brand1
from lineorder, part, supplier,date
where lo_orderdate = d_datekey
and lo_partkey = p_partkey
and lo_suppkey = s_suppkey
and p_brand1 >= 260
and p_brand1 <= 267
and s_region = 2
group by d_year,p_brand1;

