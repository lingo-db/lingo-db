select c_city,s_city,d_year,sum(lo_revenue) as revenue
from lineorder,customer,supplier,date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and (c_city = 231 or c_city = 235)
and (s_city = 231 or s_city = 235)
and d_yearmonthnum = 199712
group by c_city,s_city,d_year;

