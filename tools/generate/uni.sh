#!/usr/bin/env bash
set -euo pipefail
rm -rf resources/data/uni
mkdir -p resources/data/uni
"build/lingodb-debug/sql" "resources/data/uni" < resources/sql/uni/initialize.sql

