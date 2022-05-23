--//RUN: run-sql %s .  | FileCheck %s

--//CHECK: |                        const0  |
--//CHECK: ----------------------------------
--//CHECK: |                             1  |
(values (1),(2),(2),(2), (3))
EXCEPT
(values (2), (2),(3))