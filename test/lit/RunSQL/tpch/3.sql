--//RUN: run-sql %s %S/../../../../resources/data/tpch | FileCheck %s
--//CHECK: |                    l_orderkey  |                       revenue  |                   o_orderdate  |                o_shippriority  |
--//CHECK: -------------------------------------------------------------------------------------------------------------------------------------
--//CHECK: |                        223140  |                   355369.0698  |                    1995-03-14  |                             0  |
--//CHECK: |                        584291  |                   354494.7318  |                    1995-02-21  |                             0  |
--//CHECK: |                        405063  |                   353125.4577  |                    1995-03-03  |                             0  |
--//CHECK: |                        573861  |                   351238.2770  |                    1995-03-09  |                             0  |
--//CHECK: |                        554757  |                   349181.7426  |                    1995-03-14  |                             0  |
--//CHECK: |                        506021  |                   321075.5810  |                    1995-03-10  |                             0  |
--//CHECK: |                        121604  |                   318576.4154  |                    1995-03-07  |                             0  |
--//CHECK: |                        108514  |                   314967.0754  |                    1995-02-20  |                             0  |
--//CHECK: |                        462502  |                   312604.5420  |                    1995-03-08  |                             0  |
--//CHECK: |                        178727  |                   309728.9306  |                    1995-02-25  |                             0  |
-- TPC-H Query 3

select
        l_orderkey,
        sum(l_extendedprice * (1 - l_discount)) as revenue,
        o_orderdate,
        o_shippriority
from
        customer,
        orders,
        lineitem
where
        c_mktsegment = 'BUILDING'
        and c_custkey = o_custkey
        and l_orderkey = o_orderkey
        and o_orderdate < date '1995-03-15'
        and l_shipdate > date '1995-03-15'
group by
        l_orderkey,
        o_orderdate,
        o_shippriority
order by
        revenue desc,
        o_orderdate
limit 10

