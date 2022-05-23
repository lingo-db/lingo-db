--//RUN: run-sql %s .  | FileCheck %s

--//CHECK: |                        const0  |
--//CHECK: ----------------------------------
--//CHECK: |                             1  |
--//CHECK: |                             2  |
(values (1),(2),(2),(2), (3))
EXCEPT ALL
(values (2), (2),(3))