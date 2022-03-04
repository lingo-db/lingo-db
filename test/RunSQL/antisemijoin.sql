--//RUN: sql-to-mlir %s | mlir-db-opt --relalg-extract-nested-operators --relalg-decompose-lambdas --relalg-implicit-to-explicit-joins --relalg-pushdown --relalg-unnesting | mlir-db-opt -relalg-to-db -canonicalize | db-run "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |
--//CHECK: ----------------------------------
--//CHECK: |                  "Xenokrates"  |
--//CHECK: |                 "Aristoxenos"  |



select s.name
from studenten s
where not exists( select * from hoeren h where h.matrnr=s.matrnr)