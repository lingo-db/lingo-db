--//RUN: run-sql %s %S/../../../resources/data/tpch | FileCheck %s
--//CHECK: |                  l_returnflag  |                  l_linestatus  |                       sum_qty  |                sum_base_price  |                sum_disc_price  |                    sum_charge  |                       avg_qty  |                     avg_price  |                      avg_disc  |                   count_order  |
--//CHECK: -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--//CHECK: |                             A  |                             F  |                    3774200.00  |                 5320753880.69  |               5054096266.6828  |             5256751331.449234  |                         25.53  |                      36002.12  |                          0.05  |                        147790  |
--//CHECK: |                             N  |                             F  |                      95257.00  |                  133737795.84  |                127132372.6512  |              132286291.229445  |                         25.30  |                      35521.32  |                          0.04  |                          3765  |
--//CHECK: |                             N  |                             O  |                    7459297.00  |                10512270008.90  |               9986238338.3847  |            10385578376.585467  |                         25.54  |                      36000.92  |                          0.05  |                        292000  |
--//CHECK: |                             R  |                             F  |                    3785523.00  |                 5337950526.47  |               5071818532.9420  |             5274405503.049367  |                         25.52  |                      35994.02  |                          0.04  |                        148301  |
-- TPC-H Query 1

select
        l_returnflag,
        l_linestatus,
        sum(l_quantity) as sum_qty,
        sum(l_extendedprice) as sum_base_price,
        sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
        sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
        avg(l_quantity) as avg_qty,
        avg(l_extendedprice) as avg_price,
        avg(l_discount) as avg_disc,
        count(*) as count_order
from
        lineitem
where
        l_shipdate <= date '1998-12-01' - interval '90' day
group by
        l_returnflag,
        l_linestatus
order by
        l_returnflag,
        l_linestatus

