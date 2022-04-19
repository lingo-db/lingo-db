--//RUN: run-sql %s %S/../../../resources/data/tpch | FileCheck %s
--//CHECK: |                        n_name  |                       revenue  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                       "CHINA"  |                  7822103.0000  |
--//CHECK: |                       "INDIA"  |                  6376121.5085  |
--//CHECK: |                       "JAPAN"  |                  6000077.2184  |
--//CHECK: |                   "INDONESIA"  |                  5580475.4027  |
--//CHECK: |                     "VIETNAM"  |                  4497840.5466  |
-- TPC-H Query 5

select
        n_name,
        sum(l_extendedprice * (1 - l_discount)) as revenue
from
        customer,
        orders,
        lineitem,
        supplier,
        nation,
        region
where
        c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and l_suppkey = s_suppkey
        and c_nationkey = s_nationkey
        and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey
        and r_name = 'ASIA'
        and o_orderdate >= date '1994-01-01'
        and o_orderdate < date '1995-01-01'
group by
        n_name
order by
        revenue desc

