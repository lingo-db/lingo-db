--//RUN: run-sql %s %S/../../../../resources/data/ssb | FileCheck %s
--//CHECK: |                        c_city  |                        s_city  |                        d_year  |                       revenue  |
--//CHECK: -------------------------------------------------------------------------------------------------------------------------------------
--//CHECK: |                  "UNITED ST6"  |                  "UNITED ST9"  |                          1992  |                   25919373.00  |
--//CHECK: |                  "UNITED ST3"  |                  "UNITED ST3"  |                          1992  |                   23830867.00  |
--//CHECK: |                  "UNITED ST4"  |                  "UNITED ST3"  |                          1992  |                   21269805.00  |
--//CHECK: |                  "UNITED ST5"  |                  "UNITED ST0"  |                          1992  |                   19157227.00  |
--//CHECK: |                  "UNITED ST5"  |                  "UNITED ST7"  |                          1992  |                   18699268.00  |
--//CHECK: |                  "UNITED ST4"  |                  "UNITED ST0"  |                          1992  |                   16957558.00  |
--//CHECK: |                  "UNITED ST1"  |                  "UNITED ST9"  |                          1992  |                   14243861.00  |
--//CHECK: |                  "UNITED ST6"  |                  "UNITED ST0"  |                          1992  |                   12567553.00  |
--//CHECK: |                  "UNITED ST4"  |                  "UNITED ST9"  |                          1992  |                   12178066.00  |
--//CHECK: |                  "UNITED ST5"  |                  "UNITED ST3"  |                          1992  |                   12141464.00  |
--//CHECK: |                  "UNITED ST9"  |                  "UNITED ST0"  |                          1992  |                   11520764.00  |
--//CHECK: |                  "UNITED ST2"  |                  "UNITED ST7"  |                          1992  |                   11515729.00  |
--//CHECK: |                  "UNITED ST6"  |                  "UNITED ST6"  |                          1992  |                   10649128.00  |
--//CHECK: |                  "UNITED ST1"  |                  "UNITED ST2"  |                          1992  |                   10397206.00  |
--//CHECK: |                  "UNITED ST5"  |                  "UNITED ST9"  |                          1992  |                   10394304.00  |
--//CHECK: |                  "UNITED ST9"  |                  "UNITED ST3"  |                          1992  |                   10003542.00  |
--//CHECK: |                  "UNITED ST2"  |                  "UNITED ST2"  |                          1992  |                    9750894.00  |
--//CHECK: |                  "UNITED ST9"  |                  "UNITED ST7"  |                          1992  |                    8509746.00  |
--//CHECK: |                  "UNITED ST3"  |                  "UNITED ST9"  |                          1992  |                    8421495.00  |
--//CHECK: |                  "UNITED ST0"  |                  "UNITED ST3"  |                          1992  |                    7724885.00  |
--//CHECK: |                  "UNITED ST0"  |                  "UNITED ST0"  |                          1992  |                    7560881.00  |
--//CHECK: |                  "UNITED ST6"  |                  "UNITED ST3"  |                          1992  |                    7446570.00  |
--//CHECK: |                  "UNITED ST7"  |                  "UNITED ST3"  |                          1992  |                    7439354.00  |
--//CHECK: |                  "UNITED ST9"  |                  "UNITED ST2"  |                          1992  |                    6697568.00  |
--//CHECK: |                  "UNITED ST7"  |                  "UNITED ST0"  |                          1992  |                    6653462.00  |
--//CHECK: |                  "UNITED ST2"  |                  "UNITED ST0"  |                          1992  |                    6212334.00  |
--//CHECK: |                  "UNITED ST2"  |                  "UNITED ST6"  |                          1992  |                    6021797.00  |
--//CHECK: |                  "UNITED ST7"  |                  "UNITED ST6"  |                          1992  |                    5893710.00  |
--//CHECK: |                  "UNITED ST4"  |                  "UNITED ST2"  |                          1992  |                    5825260.00  |
--//CHECK: |                  "UNITED ST1"  |                  "UNITED ST3"  |                          1992  |                    5006460.00  |
--//CHECK: |                  "UNITED ST5"  |                  "UNITED ST2"  |                          1992  |                    4847872.00  |
--//CHECK: |                  "UNITED ST9"  |                  "UNITED ST9"  |                          1992  |                    4792501.00  |
--//CHECK: |                  "UNITED ST2"  |                  "UNITED ST9"  |                          1992  |                    4584633.00  |
--//CHECK: |                  "UNITED ST7"  |                  "UNITED ST2"  |                          1992  |                    3684786.00  |
--//CHECK: |                  "UNITED ST3"  |                  "UNITED ST0"  |                          1992  |                    3655595.00  |
--//CHECK: |                  "UNITED ST7"  |                  "UNITED ST9"  |                          1992  |                    3338403.00  |
--//CHECK: |                  "UNITED ST8"  |                  "UNITED ST3"  |                          1992  |                    3334609.00  |
--//CHECK: |                  "UNITED ST4"  |                  "UNITED ST7"  |                          1992  |                    2916975.00  |
--//CHECK: |                  "UNITED ST8"  |                  "UNITED ST0"  |                          1992  |                    2233583.00  |
--//CHECK: |                  "UNITED ST1"  |                  "UNITED ST0"  |                          1992  |                    2021712.00  |
--//CHECK: |                  "UNITED ST0"  |                  "UNITED ST6"  |                          1992  |                    1540902.00  |
--//CHECK: |                  "UNITED ST2"  |                  "UNITED ST3"  |                          1992  |                    1251472.00  |
--//CHECK: |                  "UNITED ST1"  |                  "UNITED ST7"  |                          1992  |                    1239414.00  |
select c_city, s_city, d_year, sum(lo_revenue)
as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_nation = 'UNITED STATES'
and s_nation = 'UNITED STATES'
and d_year >= 1992 and d_year <= 1997
group by c_city, s_city, d_year
order by d_year asc, revenue desc

