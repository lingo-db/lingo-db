#!/usr/bin/env bash
set -euo pipefail
TMPDIR=`mktemp --directory`
echo $TMPDIR
cp  resources/sql/ssb/initialize.sql $TMPDIR/initialize.sql
pushd $TMPDIR
echo 'd37618c646a6918be8ccc4bc79704061  dbgen.zip' | md5sum --check --status 2>/dev/null || curl -OL https://db.in.tum.de/~fent/dbgen/ssb/dbgen.zip
echo 'd37618c646a6918be8ccc4bc79704061  dbgen.zip' | md5sum --check --status
unzip -u dbgen.zip
mv dbgen/* .
rm -rf dbgen
rm dbgen.zip
rm -rf ./*.tbl
sed -i 's/#define  MAXAGG_LEN    10/#define  MAXAGG_LEN    20/' shared.h
make dbgen
SF=$3
./dbgen -f -T c -s "$SF"
./dbgen -qf -T d -s "$SF"
./dbgen -qf -T p -s "$SF"
./dbgen -qf -T s -s "$SF"
./dbgen -q -T l -s "$SF"
for table in ./*.tbl; do  sed -i 's/|$//' "$table"; done

"$1/sql" $2 < initialize.sql
popd
