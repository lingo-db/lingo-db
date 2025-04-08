#!/usr/bin/env bash
set -euo pipefail
TMPDIR=`mktemp --directory`
echo $TMPDIR
cp  resources/sql/tpch/initialize.sql $TMPDIR/initialize.sql
pushd $TMPDIR
wget -q https://github.com/electrum/tpch-dbgen/archive/32f1c1b92d1664dba542e927d23d86ffa57aa253.zip -O tpch-dbgen.zip
unzip -q tpch-dbgen.zip
mv tpch-dbgen-32f1c1b92d1664dba542e927d23d86ffa57aa253/* .
rm tpch-dbgen.zip
make
set -x
./dbgen -f -s $2
ls -la .
chmod +r *.tbl
mkdir -p "$1"  # Ensure the target directory exists
for table in ./*.tbl; do
  # sed behaves differently on macOS and linux. Currently, there is no stable, portable command that works on both.
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/|$//' "$table"  # macOS
  else
    sed -i 's/|$//' "$table"     # Linux
  fi
  mv "$table" "$1/$table"
done
popd
