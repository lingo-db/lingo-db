--//RUN: run-sql %s %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |                       v.titel  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                      "Carnap"  |                    "Bioethik"  |
--//CHECK: |                      "Carnap"  |            "Der Wiener Kreis"  |
--//CHECK: |                      "Carnap"  |                       "Ethik"  |
--//CHECK: |                      "Carnap"  |        "Wissenschaftstheorie"  |
--//CHECK: |                   "Feuerbach"  |           "Glaube und Wissen"  |
--//CHECK: |                   "Feuerbach"  |                  "Grundzuege"  |
--//CHECK: |                      "Fichte"  |                  "Grundzuege"  |
--//CHECK: |                       "Jonas"  |           "Glaube und Wissen"  |
--//CHECK: |                "Schopenhauer"  |                  "Grundzuege"  |
--//CHECK: |                "Schopenhauer"  |                       "Logik"  |
--//CHECK: |                "Theophrastos"  |                       "Ethik"  |
--//CHECK: |                "Theophrastos"  |                  "Grundzuege"  |
--//CHECK: |                "Theophrastos"  |                    "Maeeutik"  |


select s.name,v.titel
from hoeren h, studenten s,vorlesungen v
where h.matrnr=s.matrnr and h.vorlnr=v.vorlnr
order by s.name,v.titel