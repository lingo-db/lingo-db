--//RUN: sql-to-mlir %s | mlir-db-opt --relalg-extract-nested-operators --relalg-decompose-lambdas --relalg-implicit-to-explicit-joins --relalg-pushdown --relalg-unnesting | mlir-db-opt -relalg-to-db -canonicalize | run-mlir "-" %S/../../resources/data/uni | FileCheck %s

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