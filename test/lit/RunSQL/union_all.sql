--//RUN: run-sql %s .  | FileCheck %s

--//CHECK: |                        const0  |
--//CHECK: ----------------------------------
--//CHECK: |                             2  |
--//CHECK: |                             3  |
--//CHECK: |                             1  |
--//CHECK: |                             2  |
--//CHECK: |                             2  |
--//CHECK: |                             3  |

(values (1),(2),(2),(3))
UNION ALL
(values (2),(3))