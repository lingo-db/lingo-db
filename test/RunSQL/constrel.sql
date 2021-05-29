--//RUN: python3 %S/../../tools/sql-to-mlir/sql-to-mlir.py %s | mlir-db-opt -relalg-to-db -canonicalize | db-run "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                         t1.c1  |                         t1.c2  |                        power4  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                           "1"  |                             1  |                             1  |
--//CHECK: |                           "2"  |                             4  |                            16  |




select t1.c1,t1.c2, t1.c2*t1.c2 as power4
from values(('1',1),('2',4)) t1 (c1,c2)