--//RUN: run-sql %s %S/../../../resources/data/uni | FileCheck %s
--//CHECK: |                        matrnr  |                      semester  |                           avg  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                         24002  |                            18  |                         24002  |
--//CHECK: |                         25403  |                            12  |                         25403  |
--//CHECK: |                         26120  |                            10  |                         26120  |
--//CHECK: |                         26830  |                             8  |                         26830  |
--//CHECK: |                         27550  |                             6  |                         27550  |
--//CHECK: |                         28106  |                             3  |                         28106  |
--//CHECK: |                         29120  |                             2  |                         29337  |
--//CHECK: |                         29555  |                             2  |                         29337  |

select s.matrnr,s.semester, avg(s.matrnr) over (partition by s.semester)
from studenten s