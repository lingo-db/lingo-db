--//RUN: run-sql %s %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |
--//CHECK: ----------------------------------
--//CHECK: |                       "Jonas"  |
--//CHECK: |                      "Fichte"  |
--//CHECK: |                "Schopenhauer"  |
--//CHECK: |                      "Carnap"  |
--//CHECK: |                "Theophrastos"  |
--//CHECK: |                   "Feuerbach"  |


select s.name
from studenten s
where exists( select * from hoeren h where h.matrnr=s.matrnr)