--//RUN: run-sql %s %S/../../../../resources/data/ssb | FileCheck %s
--//CHECK: |                      c_nation  |                      s_nation  |                        d_year  |                       revenue  |
--//CHECK: -------------------------------------------------------------------------------------------------------------------------------------
--//CHECK: |                     "VIETNAM"  |                       "CHINA"  |                          1992  |                  660498421.00  |
--//CHECK: |                       "JAPAN"  |                       "CHINA"  |                          1992  |                  633473889.00  |
--//CHECK: |                   "INDONESIA"  |                       "CHINA"  |                          1992  |                  603961339.00  |
--//CHECK: |                       "JAPAN"  |                       "JAPAN"  |                          1992  |                  537585659.00  |
--//CHECK: |                       "JAPAN"  |                   "INDONESIA"  |                          1992  |                  534147037.00  |
--//CHECK: |                     "VIETNAM"  |                       "JAPAN"  |                          1992  |                  524133484.00  |
--//CHECK: |                     "VIETNAM"  |                   "INDONESIA"  |                          1992  |                  518283208.00  |
--//CHECK: |                   "INDONESIA"  |                       "JAPAN"  |                          1992  |                  497636929.00  |
--//CHECK: |                       "JAPAN"  |                       "INDIA"  |                          1992  |                  447995912.00  |
--//CHECK: |                       "INDIA"  |                       "CHINA"  |                          1992  |                  437475335.00  |
--//CHECK: |                       "JAPAN"  |                     "VIETNAM"  |                          1992  |                  419396398.00  |
--//CHECK: |                     "VIETNAM"  |                       "INDIA"  |                          1992  |                  417731507.00  |
--//CHECK: |                       "CHINA"  |                       "CHINA"  |                          1992  |                  409873756.00  |
--//CHECK: |                       "INDIA"  |                   "INDONESIA"  |                          1992  |                  407538490.00  |
--//CHECK: |                       "INDIA"  |                       "JAPAN"  |                          1992  |                  403338963.00  |
--//CHECK: |                   "INDONESIA"  |                       "INDIA"  |                          1992  |                  389677452.00  |
--//CHECK: |                       "CHINA"  |                   "INDONESIA"  |                          1992  |                  379064659.00  |
--//CHECK: |                     "VIETNAM"  |                     "VIETNAM"  |                          1992  |                  378542607.00  |
--//CHECK: |                       "INDIA"  |                     "VIETNAM"  |                          1992  |                  349350741.00  |
--//CHECK: |                       "CHINA"  |                       "JAPAN"  |                          1992  |                  344806192.00  |
--//CHECK: |                   "INDONESIA"  |                   "INDONESIA"  |                          1992  |                  342784646.00  |
--//CHECK: |                   "INDONESIA"  |                     "VIETNAM"  |                          1992  |                  299620484.00  |
--//CHECK: |                       "CHINA"  |                       "INDIA"  |                          1992  |                  297892205.00  |
--//CHECK: |                       "INDIA"  |                       "INDIA"  |                          1992  |                  279491384.00  |
--//CHECK: |                       "CHINA"  |                     "VIETNAM"  |                          1992  |                  227326902.00  |
select c_nation, s_nation, d_year,
sum(lo_revenue) as revenue
from customer, lineorder, supplier, date
where lo_custkey = c_custkey
and lo_suppkey = s_suppkey
and lo_orderdate = d_datekey
and c_region = 'ASIA'
and s_region = 'ASIA'
and d_year >= 1992 and d_year <= 1997
group by c_nation, s_nation, d_year
order by d_year asc, revenue desc;

