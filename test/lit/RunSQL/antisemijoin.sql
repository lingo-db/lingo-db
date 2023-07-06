--//RUN: run-sql %s %S/../../../resources/data/uni | FileCheck %s

--//CHECK: |                        name  |
--//CHECK: ----------------------------------
--//CHECK: |                  "Xenokrates"  |
--//CHECK: |                 "Aristoxenos"  |



select s.name
from studenten s
where not exists( select * from hoeren h where h.matrnr=s.matrnr)
order by s.name desc