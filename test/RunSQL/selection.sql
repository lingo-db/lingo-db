--//RUN: sql-to-mlir %s | mlir-db-opt -relalg-to-db -canonicalize | run-mlir "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        matrnr  |                          name  |                      semester  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                         28106  |                      "Carnap"  |                             3  |


select *
from studenten
where name='Carnap'