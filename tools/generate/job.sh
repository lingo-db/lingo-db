#!/usr/bin/env bash
set -euo pipefail
TMPDIR=`mktemp --directory`
echo $TMPDIR
pushd $TMPDIR

# MD5 check that works on both macOS and Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  [[ "$(md5 -q imdb.tzst 2>/dev/null)" == "6ec7a92fbd8a61c2d6d26e3aa78e6f84" ]] || curl -OL https://db.in.tum.de/~fent/dbgen/job/imdb.tzst
  [[ "$(md5 -q imdb.tzst)" == "6ec7a92fbd8a61c2d6d26e3aa78e6f84" ]] || { echo "MD5 check failed"; exit 1; }
else
  # Linux
  echo '6ec7a92fbd8a61c2d6d26e3aa78e6f84  imdb.tzst' | md5sum --check --status 2>/dev/null || curl -OL https://db.in.tum.de/~fent/dbgen/job/imdb.tzst
  echo '6ec7a92fbd8a61c2d6d26e3aa78e6f84  imdb.tzst' | md5sum --check --status
fi

tar -xf imdb.tzst
chmod +r *.csv

# sed command that works on both macOS and Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS requires extension with -i
  for table in ./*.csv; do sed -i '' 's/\\\([^"\\]\)/\\\\\1/g' "$table"; done
else
  # Linux
  for table in ./*.csv; do sed -i 's/\\\([^"\\]\)/\\\\\1/g' "$table"; done
fi

for table in ./*.csv; do mv "$table" "$1/$table"; done
popd
