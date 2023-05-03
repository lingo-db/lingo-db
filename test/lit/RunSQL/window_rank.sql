--//RUN: run-sql %s %S/../../../resources/data/uni | FileCheck %s
--//CHECK: |                        matrnr  |                          rank  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                         24002  |                             1  |
--//CHECK: |                         25403  |                             2  |
--//CHECK: |                         26120  |                             3  |
--//CHECK: |                         26830  |                             4  |
--//CHECK: |                         27550  |                             5  |
--//CHECK: |                         28106  |                             6  |
--//CHECK: |                         29120  |                             7  |
--//CHECK: |                         29555  |                             8  |
select s.matrnr, rank() over (order by s.matrnr)
from studenten s
order by s.matrnr