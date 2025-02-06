#!/usr/bin/env bash
set -euo pipefail
TMPDIR=`mktemp --directory`
echo $TMPDIR
cp  resources/sql/tpcds/initialize.sql $TMPDIR/initialize.sql
pushd $TMPDIR
wget -q https://gitlab.db.in.tum.de/fent/tpcds-kit/-/archive/master/tpcds-kit-master.zip -O tpcds-dbgen.zip
unzip -q tpcds-dbgen.zip
mv tpcds-kit-master/* .
rm tpcds-dbgen.zip
cd tools
ls
make
set -x
./dsdgen -FORCE -SCALE $2
chmod +r *.dat
for table in ./*.dat; do  sed -i 's/|$//' "$table"; mv "$table" "$1/$table";  done
popd
