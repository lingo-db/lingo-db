--//RUN: run-sql %s %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |                      h.vorlnr  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                  "Xenokrates"  |                          null  |
--//CHECK: |                       "Jonas"  |                          5022  |
--//CHECK: |                      "Fichte"  |                          5001  |
--//CHECK: |                 "Aristoxenos"  |                          null  |
--//CHECK: |                "Schopenhauer"  |                          4052  |
--//CHECK: |                "Schopenhauer"  |                          5001  |
--//CHECK: |                      "Carnap"  |                          5259  |
--//CHECK: |                      "Carnap"  |                          5216  |
--//CHECK: |                      "Carnap"  |                          5052  |
--//CHECK: |                      "Carnap"  |                          5041  |
--//CHECK: |                "Theophrastos"  |                          5049  |
--//CHECK: |                "Theophrastos"  |                          5041  |
--//CHECK: |                "Theophrastos"  |                          5001  |
--//CHECK: |                   "Feuerbach"  |                          5001  |
--//CHECK: |                   "Feuerbach"  |                          5022  |



select s.name, h.vorlnr
from studenten s left outer join hoeren h on
                                s.matrnr = h.matrnr
