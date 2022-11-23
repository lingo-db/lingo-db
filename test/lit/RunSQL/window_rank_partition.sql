--//RUN: run-sql %s %S/../../../resources/data/uni | FileCheck %s
--//CHECK: |                        matrnr  |                      semester  |                          rank  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                         24002  |                            18  |                             1  |
--//CHECK: |                         25403  |                            12  |                             1  |
--//CHECK: |                         26120  |                            10  |                             1  |
--//CHECK: |                         26830  |                             8  |                             1  |
--//CHECK: |                         27550  |                             6  |                             1  |
--//CHECK: |                         28106  |                             3  |                             1  |
--//CHECK: |                         29120  |                             2  |                             1  |
--//CHECK: |                         29555  |                             2  |                             2  |
select s.matrnr,s.semester, rank() over (partition by s.semester order by s.matrnr)
from studenten s