--//RUN: run-sql %s %S/../../../resources/data/tpch | FileCheck %s
--//CHECK: |                     cntrycode  |                       numcust  |                    totacctbal  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                          "13"  |                            94  |                     714035.05  |
--//CHECK: |                          "17"  |                            96  |                     722560.15  |
--//CHECK: |                          "18"  |                            99  |                     738012.52  |
--//CHECK: |                          "23"  |                            93  |                     708285.25  |
--//CHECK: |                          "29"  |                            85  |                     632693.46  |
--//CHECK: |                          "30"  |                            87  |                     646748.02  |
--//CHECK: |                          "31"  |                            87  |                     647372.50  |
-- TPC-H Query 22

select
        cntrycode,
        count(*) as numcust,
        sum(c_acctbal) as totacctbal
from
        (
                select
                        substring(c_phone from 1 for 2) as cntrycode,
                        c_acctbal
                from
                        customer
                where
                        substring(c_phone from 1 for 2) in
                                ('13', '31', '23', '29', '30', '18', '17')
                        and c_acctbal > (
                                select
                                        avg(c_acctbal)
                                from
                                        customer
                                where
                                        c_acctbal > 0.00
                                        and substring(c_phone from 1 for 2) in
                                                ('13', '31', '23', '29', '30', '18', '17')
                        )
                        and not exists (
                                select
                                        *
                                from
                                        orders
                                where
                                        o_custkey = c_custkey
                        )
        ) as custsale
group by
        cntrycode
order by
        cntrycode


