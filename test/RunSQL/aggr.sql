--//RUN: sql-to-mlir %s | mlir-db-opt -relalg-to-db -canonicalize | db-run "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                      s.matrnr  |                           sum  |                           min  |                           max  |                           avg  |                         count  |                         count  |
--//CHECK: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--//CHECK: |                         26120  |                             4  |                             4  |                             4  |                           4.0  |                             1  |                             1  |
--//CHECK: |                         27550  |                             8  |                             4  |                             4  |                           4.0  |                             2  |                             2  |
--//CHECK: |                         29120  |                            10  |                             2  |                             4  |                           3.3  |                             3  |                             3  |
--//CHECK: |                         29555  |                             6  |                             2  |                             4  |                           3.0  |                             2  |                             2  |
--//CHECK: |                         28106  |                            11  |                             2  |                             4  |                           2.7  |                             4  |                             4  |
--//CHECK: |                         25403  |                             2  |                             2  |                             2  |                           2.0  |                             1  |                             1  |


select s.matrnr, sum(v.sws),min(v.sws),max(v.sws),avg(v.sws*1.0),count(*),count(v.sws)
from studenten s, hoeren h, vorlesungen v
where s.matrnr=h.matrnr and h.vorlnr=v.vorlnr
group by s.matrnr
