--//RUN: sql-to-mlir %s | mlir-db-opt -relalg-to-db -canonicalize | db-run "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |                       v.titel  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                      "Fichte"  |                  "Grundzuege"  |
--//CHECK: |                "Schopenhauer"  |                  "Grundzuege"  |
--//CHECK: |                "Theophrastos"  |                  "Grundzuege"  |
--//CHECK: |                   "Feuerbach"  |                  "Grundzuege"  |
--//CHECK: |                      "Carnap"  |                       "Ethik"  |
--//CHECK: |                "Theophrastos"  |                       "Ethik"  |
--//CHECK: |                "Theophrastos"  |                    "Maeeutik"  |
--//CHECK: |                "Schopenhauer"  |                       "Logik"  |
--//CHECK: |                      "Carnap"  |        "Wissenschaftstheorie"  |
--//CHECK: |                      "Carnap"  |                    "Bioethik"  |
--//CHECK: |                      "Carnap"  |            "Der Wiener Kreis"  |
--//CHECK: |                       "Jonas"  |           "Glaube und Wissen"  |
--//CHECK: |                   "Feuerbach"  |           "Glaube und Wissen"  |


select s.name,v.titel
from hoeren h, studenten s,vorlesungen v
where h.matrnr=s.matrnr and h.vorlnr=v.vorlnr