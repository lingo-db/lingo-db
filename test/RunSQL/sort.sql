--//RUN: python3 %S/../../tools/sql-to-mlir/sql-to-mlir.py %s | mlir-db-opt -relalg-to-db -canonicalize | db-run "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |                       v.titel  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                      "Carnap"  |        "Wissenschaftstheorie"  |
--//CHECK: |                      "Carnap"  |                       "Ethik"  |
--//CHECK: |                      "Carnap"  |            "Der Wiener Kreis"  |
--//CHECK: |                      "Carnap"  |                    "Bioethik"  |
--//CHECK: |                   "Feuerbach"  |                  "Grundzuege"  |
--//CHECK: |                   "Feuerbach"  |           "Glaube und Wissen"  |
--//CHECK: |                      "Fichte"  |                  "Grundzuege"  |
--//CHECK: |                       "Jonas"  |           "Glaube und Wissen"  |
--//CHECK: |                "Schopenhauer"  |                       "Logik"  |
--//CHECK: |                "Schopenhauer"  |                  "Grundzuege"  |
--//CHECK: |                "Theophrastos"  |                    "Maeeutik"  |
--//CHECK: |                "Theophrastos"  |                  "Grundzuege"  |
--//CHECK: |                "Theophrastos"  |                       "Ethik"  |



select s.name,v.titel
from hoeren h, studenten s,vorlesungen v
where h.matrnr=s.matrnr and h.vorlnr=v.vorlnr
order by s.name asc,v.titel desc