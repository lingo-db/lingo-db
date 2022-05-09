--//RUN: run-sql %s %S/../../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |
--//CHECK: ----------------------------------
--//CHECK: |                  "Xenokrates"  |
--//CHECK: |                 "Aristoxenos"  |



select s.name
from studenten s
where not exists( select * from hoeren h where h.matrnr=s.matrnr)