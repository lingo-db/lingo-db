--//RUN: python3 %S/../../tools/sql-to-mlir/sql-to-mlir.py %s | mlir-db-opt --relalg-extract-nested-operators --relalg-decompose-lambdas --relalg-implicit-to-explicit-joins --relalg-pushdown --relalg-unnesting | mlir-db-opt -relalg-to-db -canonicalize | db-run "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |                      h.matrnr  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                  "Xenokrates"  |                          null  |
--//CHECK: |                       "Jonas"  |                         25403  |
--//CHECK: |                      "Fichte"  |                         26120  |
--//CHECK: |                 "Aristoxenos"  |                          null  |
--//CHECK: |                "Schopenhauer"  |                         27550  |
--//CHECK: |                "Schopenhauer"  |                         27550  |
--//CHECK: |                      "Carnap"  |                         28106  |
--//CHECK: |                      "Carnap"  |                         28106  |
--//CHECK: |                      "Carnap"  |                         28106  |
--//CHECK: |                      "Carnap"  |                         28106  |
--//CHECK: |                "Theophrastos"  |                         29120  |
--//CHECK: |                "Theophrastos"  |                         29120  |
--//CHECK: |                "Theophrastos"  |                         29120  |
--//CHECK: |                   "Feuerbach"  |                         29555  |
--//CHECK: |                   "Feuerbach"  |                         29555  |



select s.name, h.matrnr
from studenten s left outer join hoeren h on
                                s.matrnr = h.matrnr
