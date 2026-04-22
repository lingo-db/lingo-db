-- TPC-H Query 2

select
    s_acctbal,
    p_partkey
from
    part,
    supplier,
    partsupp,
    nation,
    region
where
    p_partkey = ps_partkey
  and s_suppkey = ps_suppkey
  and p_size = 15
  and p_type like '%BRASS'
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'EUROPE'
order by
    s_acctbal,
    p_partkey
    limit 100
