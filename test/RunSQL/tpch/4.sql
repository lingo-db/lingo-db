--//RUN: run-sql %s %S/../../../resources/data/tpch | FileCheck %s
--//CHECK: |               o_orderpriority  |                   order_count  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                    "1-URGENT"  |                           999  |
--//CHECK: |                      "2-HIGH"  |                           997  |
--//CHECK: |                    "3-MEDIUM"  |                          1031  |
--//CHECK: |             "4-NOT SPECIFIED"  |                           989  |
--//CHECK: |                       "5-LOW"  |                          1077  |
-- TPC-H Query 4

select
        o_orderpriority,
        count(*) as order_count
from
        orders
where
        o_orderdate >= date '1993-07-01'
        and o_orderdate < date '1993-10-01'
        and exists (
                select
                        *
                from
                        lineitem
                where
                        l_orderkey = o_orderkey
                        and l_commitdate < l_receiptdate
        )
group by
        o_orderpriority
order by
        o_orderpriority

