#!/usr/bin/env bash
set -euo pipefail
TMPDIR=`mktemp --directory`
echo $TMPDIR
cp  resources/sql/job/initialize.sql $TMPDIR/initialize.sql
pushd $TMPDIR
echo '6ec7a92fbd8a61c2d6d26e3aa78e6f84  imdb.tzst' | md5sum --check --status 2>/dev/null || curl -OL https://db.in.tum.de/~fent/dbgen/job/imdb.tzst
echo '6ec7a92fbd8a61c2d6d26e3aa78e6f84  imdb.tzst' | md5sum --check --status
tar --skip-old-files -xf imdb.tzst

for table in ./*.csv; do  sed -i 's/\\\([^"\\]\)/\\\\\1/g' "$table"; done

"$1/sql" $2 < initialize.sql
popd
