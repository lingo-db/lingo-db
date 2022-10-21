--//RUN: run-sql %s %S/../../../resources/data/uni | FileCheck %s

--//CHECK: |                        name  |                       titel  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                      "Carnap"  |        "Wissenschaftstheorie"  |
--//CHECK: |                      "Carnap"  |                       "Ethik"  |
--//CHECK: |                      "Carnap"  |            "Der Wiener Kreis"  |
--//CHECK: |                      "Carnap"  |                    "Bioethik"  |
--//CHECK: |                   "Feuerbach"  |                  "Grundzuege"  |



select s.name,v.titel
from hoeren h, studenten s,vorlesungen v
where h.matrnr=s.matrnr and h.vorlnr=v.vorlnr
order by s.name asc,v.titel desc
limit 5