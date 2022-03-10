--//RUN: sql-to-mlir  %s | mlir-db-opt -relalg-to-db -canonicalize | db-run "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        matrnr  |                          name  |                      semester  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                         24002  |                  "Xenokrates"  |                            18  |
--//CHECK: |                         25403  |                       "Jonas"  |                            12  |
--//CHECK: |                         26120  |                      "Fichte"  |                            10  |
--//CHECK: |                         26830  |                 "Aristoxenos"  |                             8  |
--//CHECK: |                         27550  |                "Schopenhauer"  |                             6  |
--//CHECK: |                         28106  |                      "Carnap"  |                             3  |
--//CHECK: |                         29120  |                "Theophrastos"  |                             2  |
--//CHECK: |                         29555  |                   "Feuerbach"  |                             2  |

select *
from studenten