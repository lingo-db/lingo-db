--//RUN: python3 %S/../../tools/sql-to-mlir/sql-to-mlir.py %s | mlir-db-opt -relalg-to-db -canonicalize | db-run "-" %S/../../resources/data/uni | FileCheck %s

--//CHECK: |                        s.name  |                       v.titel  |
--//CHECK: -------------------------------------------------------------------
--//CHECK: |                      "Fichte"  |                  "Grundzuege"  |
--//CHECK: |                "Schopenhauer"  |                  "Grundzuege"  |
--//CHECK: |                "Schopenhauer"  |                       "Logik"  |
--//CHECK: |                      "Carnap"  |                       "Ethik"  |
--//CHECK: |                      "Carnap"  |        "Wissenschaftstheorie"  |
--//CHECK: |                      "Carnap"  |                    "Bioethik"  |
--//CHECK: |                      "Carnap"  |            "Der Wiener Kreis"  |
--//CHECK: |                "Theophrastos"  |                  "Grundzuege"  |
--//CHECK: |                "Theophrastos"  |                       "Ethik"  |
--//CHECK: |                "Theophrastos"  |                    "Maeeutik"  |
--//CHECK: |                   "Feuerbach"  |           "Glaube und Wissen"  |
--//CHECK: |                       "Jonas"  |           "Glaube und Wissen"  |
--//CHECK: |                   "Feuerbach"  |                  "Grundzuege"  |


select s.name,v.titel
from hoeren h, studenten s,vorlesungen v
where h.matrnr=s.matrnr and h.vorlnr=v.vorlnr