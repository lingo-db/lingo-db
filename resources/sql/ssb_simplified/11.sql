select sum(lo_extendedprice * lo_discount) as revenue
from lineorder
where lo_orderdate >= 19930101 and lo_orderdate <= 19940101 and lo_discount>=1
and lo_discount<=3
and lo_quantity<25;

