#!/usr/bin/env bash
set -euo pipefail
TMPDIR=`mktemp --directory`
echo $TMPDIR
cp  tools/scripts/ssb_convert_to_simplified.py $TMPDIR/ssb_convert_to_simplified.py

pushd $TMPDIR
git clone https://github.com/lingo-db/ssb-dbgen.git
cmake -B . ssb-dbgen && cmake --build .
SF=$2
./dbgen -f -T c -s "$SF"
./dbgen -qf -T d -s "$SF"
./dbgen -qf -T p -s "$SF"
./dbgen -qf -T s -s "$SF"
./dbgen -q -T l -s "$SF"
chmod +r *.tbl
for table in ./*.tbl; do  sed -i 's/|$//' "$table"; done
python3 ssb_convert_to_simplified.py ./

for table in ./*.tbl; do mv "$table" "$1/$table"; done
popd
