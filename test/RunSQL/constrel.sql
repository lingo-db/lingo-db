--//RUN: run-sql %s %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                         t1.c1  |                         t1.c2  |                        power4  |
--//CHECK: ----------------------------------------------------------------------------------------------------
--//CHECK: |                           a  |                             1  |                             1  |
--//CHECK: |                           b  |                             4  |                            16  |




select t1.c1,t1.c2, t1.c2*t1.c2 as power4
from (values ('a',1),('b',4)) t1 (c1,c2)