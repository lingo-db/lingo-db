#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <output_dir> <scale_factor>"
  exit 1
fi

if [[ "$1" = /* ]]; then
  OUTDIR="$1"
else
  OUTDIR="$(realpath "$1")"
fi

TMPDIR=`mktemp --directory`
echo $TMPDIR
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
for table in ./*.tbl; do
  # sed behaves differently on macOS and linux. Currently, there is no stable, portable command that works on both.
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/|$//' "$table"  # macOS
  else
    sed -i 's/|$//' "$table"     # Linux
  fi
done
mkdir -p "$OUTDIR"
for table in ./*.tbl; do mv "$table" "$OUTDIR/$table"; done
popd
