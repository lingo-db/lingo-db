--//RUN: run-sql %s %S/../../../resources/data/tpch | FileCheck %s
--//CHECK: |                       c_count  |                      custdist  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                             0  |                          5000  |
--//CHECK: |                            10  |                           665  |
--//CHECK: |                             9  |                           657  |
--//CHECK: |                            11  |                           621  |
--//CHECK: |                            12  |                           567  |
--//CHECK: |                             8  |                           564  |
--//CHECK: |                            13  |                           492  |
--//CHECK: |                            18  |                           482  |
--//CHECK: |                             7  |                           480  |
--//CHECK: |                            20  |                           456  |
--//CHECK: |                            14  |                           456  |
--//CHECK: |                            16  |                           449  |
--//CHECK: |                            19  |                           447  |
--//CHECK: |                            15  |                           432  |
--//CHECK: |                            17  |                           423  |
--//CHECK: |                            21  |                           412  |
--//CHECK: |                            22  |                           371  |
--//CHECK: |                             6  |                           337  |
--//CHECK: |                            23  |                           323  |
--//CHECK: |                            24  |                           256  |
--//CHECK: |                            25  |                           204  |
--//CHECK: |                             5  |                           204  |
--//CHECK: |                            26  |                           155  |
--//CHECK: |                            27  |                           141  |
--//CHECK: |                            28  |                            97  |
--//CHECK: |                             4  |                            94  |
--//CHECK: |                            29  |                            64  |
--//CHECK: |                             3  |                            48  |
--//CHECK: |                            30  |                            27  |
--//CHECK: |                            31  |                            26  |
--//CHECK: |                            32  |                            14  |
--//CHECK: |                            33  |                            11  |
--//CHECK: |                             2  |                            11  |
--//CHECK: |                            34  |                             6  |
--//CHECK: |                            35  |                             5  |
--//CHECK: |                             1  |                             2  |
--//CHECK: |                            36  |                             1  |
-- TPC-H Query 13

select
        c_count,
        count(*) as custdist
from
        (
                select
                        c_custkey,
                        count(o_orderkey) c_count
                from
                        customer left outer join orders on
                                c_custkey = o_custkey
                                and o_comment not like '%special%requests%'
                group by
                        c_custkey
        ) as c_orders
group by
        c_count
order by
        custdist desc,
        c_count desc

