--//RUN: run-sql %s %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |
--//CHECK: ----------------------------------
--//CHECK: |                      "Carnap"  |
--//CHECK: |                   "Feuerbach"  |
--//CHECK: |                      "Fichte"  |
--//CHECK: |                       "Jonas"  |
--//CHECK: |                "Schopenhauer"  |
--//CHECK: |                "Theophrastos"  |


select s.name
from studenten s
where exists( select * from hoeren h where h.matrnr=s.matrnr)
order by s.name