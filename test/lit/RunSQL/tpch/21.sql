--//RUN: run-sql %s %S/../../../../resources/data/tpch | FileCheck %s
--//CHECK: |                        s_name  |                       numwait  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |          "Supplier#000000445"  |                            16  |
--//CHECK: |          "Supplier#000000825"  |                            16  |
--//CHECK: |          "Supplier#000000709"  |                            15  |
--//CHECK: |          "Supplier#000000762"  |                            15  |
--//CHECK: |          "Supplier#000000357"  |                            14  |
--//CHECK: |          "Supplier#000000399"  |                            14  |
--//CHECK: |          "Supplier#000000496"  |                            14  |
--//CHECK: |          "Supplier#000000977"  |                            13  |
--//CHECK: |          "Supplier#000000144"  |                            12  |
--//CHECK: |          "Supplier#000000188"  |                            12  |
--//CHECK: |          "Supplier#000000415"  |                            12  |
--//CHECK: |          "Supplier#000000472"  |                            12  |
--//CHECK: |          "Supplier#000000633"  |                            12  |
--//CHECK: |          "Supplier#000000708"  |                            12  |
--//CHECK: |          "Supplier#000000889"  |                            12  |
--//CHECK: |          "Supplier#000000380"  |                            11  |
--//CHECK: |          "Supplier#000000602"  |                            11  |
--//CHECK: |          "Supplier#000000659"  |                            11  |
--//CHECK: |          "Supplier#000000821"  |                            11  |
--//CHECK: |          "Supplier#000000929"  |                            11  |
--//CHECK: |          "Supplier#000000262"  |                            10  |
--//CHECK: |          "Supplier#000000460"  |                            10  |
--//CHECK: |          "Supplier#000000486"  |                            10  |
--//CHECK: |          "Supplier#000000669"  |                            10  |
--//CHECK: |          "Supplier#000000718"  |                            10  |
--//CHECK: |          "Supplier#000000778"  |                            10  |
--//CHECK: |          "Supplier#000000167"  |                             9  |
--//CHECK: |          "Supplier#000000578"  |                             9  |
--//CHECK: |          "Supplier#000000673"  |                             9  |
--//CHECK: |          "Supplier#000000687"  |                             9  |
--//CHECK: |          "Supplier#000000074"  |                             8  |
--//CHECK: |          "Supplier#000000565"  |                             8  |
--//CHECK: |          "Supplier#000000648"  |                             8  |
--//CHECK: |          "Supplier#000000918"  |                             8  |
--//CHECK: |          "Supplier#000000427"  |                             7  |
--//CHECK: |          "Supplier#000000503"  |                             7  |
--//CHECK: |          "Supplier#000000610"  |                             7  |
--//CHECK: |          "Supplier#000000670"  |                             7  |
--//CHECK: |          "Supplier#000000811"  |                             7  |
--//CHECK: |          "Supplier#000000114"  |                             6  |
--//CHECK: |          "Supplier#000000379"  |                             6  |
--//CHECK: |          "Supplier#000000436"  |                             6  |
--//CHECK: |          "Supplier#000000500"  |                             6  |
--//CHECK: |          "Supplier#000000660"  |                             6  |
--//CHECK: |          "Supplier#000000788"  |                             6  |
--//CHECK: |          "Supplier#000000846"  |                             6  |
--//CHECK: |          "Supplier#000000920"  |                             4  |
-- TPC-H Query 21

select
        s_name,
        count(*) as numwait
from
        supplier,
        lineitem l1,
        orders,
        nation
where
        s_suppkey = l1.l_suppkey
        and o_orderkey = l1.l_orderkey
        and o_orderstatus = 'F'
        and l1.l_receiptdate > l1.l_commitdate
        and exists (
                select
                        *
                from
                        lineitem l2
                where
                        l2.l_orderkey = l1.l_orderkey
                        and l2.l_suppkey <> l1.l_suppkey
        )
        and not exists (
                select
                        *
                from
                        lineitem l3
                where
                        l3.l_orderkey = l1.l_orderkey
                        and l3.l_suppkey <> l1.l_suppkey
                        and l3.l_receiptdate > l3.l_commitdate
        )
        and s_nationkey = n_nationkey
        and n_name = 'SAUDI ARABIA'
group by
        s_name
order by
        numwait desc,
        s_name
limit 100

