--//RUN: run-sql %s %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                      s.matrnr  |                       nextsem  |                        const4  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                         24002  |                            19  |                             4  |
--//CHECK: |                         25403  |                            13  |                             4  |
--//CHECK: |                         26120  |                            11  |                             4  |
--//CHECK: |                         26830  |                             9  |                             4  |
--//CHECK: |                         27550  |                             7  |                             4  |
--//CHECK: |                         28106  |                             4  |                             4  |
--//CHECK: |                         29120  |                             3  |                             4  |
--//CHECK: |                         29555  |                             3  |                             4  |

select s.matrnr, s.semester+1 as nextsem,1+3 as const4
from studenten s