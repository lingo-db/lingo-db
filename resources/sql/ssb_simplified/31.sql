select c_nation,s_nation,d_year,sum(lo_revenue) as revenue
from lineorder,customer, supplier,date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_region = 2
and s_region = 2
and d_year >= 1992 and d_year <= 1997
group by c_nation,s_nation,d_year;

