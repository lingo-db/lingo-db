#!/usr/bin/env bash
set -euo pipefail
set -x
TMPDIR=`mktemp --directory`
echo $TMPDIR
cp  resources/sql/tpcds/initialize.sql $TMPDIR/initialize.sql
pushd $TMPDIR
wget -q https://github.com/lingo-db/tpcds-kit/archive/refs/heads/master.zip -O tpcds-dbgen.zip
unzip -q tpcds-dbgen.zip
mv tpcds-kit-master/* .
rm tpcds-dbgen.zip
cd tools
if [[ "$OSTYPE" == "darwin"* ]]; then
  make OS=MACOS MACOS_CFLAGS="-g -Wall --std=c89"  # macOS
else
  make OS=LINUX     # Linux
fi
./dsdgen -FORCE -SCALE $2
chmod +r *.dat
mkdir -p "$1"  # Ensure the target directory exists
for table in ./*.dat; do
  # sed behaves differently on macOS and linux. Currently, there is no stable, portable command that works on both.
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/|$//' "$table"  # macOS
  else
    sed -i 's/|$//' "$table"     # Linux
  fi
  mv "$table" "$1/$table"
done
popd
