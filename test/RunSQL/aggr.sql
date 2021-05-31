--//RUN: python3 %S/../../tools/sql-to-mlir/sql-to-mlir.py %s | mlir-db-opt -relalg-to-db -canonicalize | db-run "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                      s.matrnr  |                           sum  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                         25403  |                             2  |
--//CHECK: |                         26120  |                             4  |
--//CHECK: |                         27550  |                             8  |
--//CHECK: |                         28106  |                            11  |
--//CHECK: |                         29120  |                            10  |
--//CHECK: |                         29555  |                             6  |




select s.matrnr, sum(v.sws)
from studenten s, hoeren h, vorlesungen v
where s.matrnr=h.matrnr and h.vorlnr=v.vorlnr
group by s.matrnr
