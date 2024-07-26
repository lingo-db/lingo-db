select sum(lo_extendedprice * lo_discount) as revenue
from lineorder
where lo_orderdate >= 19940101 and lo_orderdate <= 19940131
and lo_discount>=4 and lo_discount<=6
and lo_quantity>=26
and lo_quantity<=35;


