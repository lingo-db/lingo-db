--//RUN: run-sql %s %S/../../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |                      h.vorlnr  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                 "Aristoxenos"  |                          null  |
--//CHECK: |                      "Carnap"  |                          5041  |
--//CHECK: |                      "Carnap"  |                          5052  |
--//CHECK: |                      "Carnap"  |                          5216  |
--//CHECK: |                      "Carnap"  |                          5259  |
--//CHECK: |                   "Feuerbach"  |                          5001  |
--//CHECK: |                   "Feuerbach"  |                          5022  |
--//CHECK: |                      "Fichte"  |                          5001  |
--//CHECK: |                       "Jonas"  |                          5022  |
--//CHECK: |                "Schopenhauer"  |                          4052  |
--//CHECK: |                "Schopenhauer"  |                          5001  |
--//CHECK: |                "Theophrastos"  |                          5001  |
--//CHECK: |                "Theophrastos"  |                          5041  |
--//CHECK: |                "Theophrastos"  |                          5049  |
--//CHECK: |                  "Xenokrates"  |                          null  |


select s.name, h.vorlnr
from studenten s left outer join hoeren h on
                                s.matrnr = h.matrnr
order by s.name, h.vorlnr
