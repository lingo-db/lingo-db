--//RUN: python3 %S/../../tools/sql-to-mlir/sql-to-mlir.py %s | mlir-db-opt -relalg-to-db -canonicalize | db-run "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        matrnr  |                          name  |                      semester  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                         28106  |                      "Carnap"  |                             3  |


select *
from studenten
where matrnr=28106