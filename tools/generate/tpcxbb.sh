#!/usr/bin/env bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <desired location> <scale factor> <TPCX-BB_Tools zip file>"
    exit 1
fi

set -euo pipefail
TMPDIR=`mktemp --directory`
echo $TMPDIR
pushd $TMPDIR


cp $3 ./tpcxbb.zip
bsdtar -xf tpcxbb.zip # unzip does not work with the tpcxbb.zip file
bsdtar -xf TPCX-BB_Tools_1.6.2.zip

rm tpcxbb.zip
rm TPCX-BB_Tools_1.6.2.zip
rm TPCx-BB_v1.6.2.docx
rm TPCx-BB_v1.6.2.pdf

# sed behaves differently on macOS and linux. Currently, there is no stable, portable command that works on both.
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  sed -i '' 's|^export BIG_BENCH_DATAGEN_DFS_REPLICATION=.*|export BIG_BENCH_DATAGEN_DFS_REPLICATION="2"|' ./tpcx-bb-v1.6.2/conf/userSettings.conf
  sed -i '' "s|^export BIG_BENCH_HDFS_ABSOLUTE_PATH=.*|export BIG_BENCH_HDFS_ABSOLUTE_PATH=\"$TMPDIR\"|" ./tpcx-bb-v1.6.2/conf/userSettings.conf
else
  # Linux
  sed -i 's|^export BIG_BENCH_DATAGEN_DFS_REPLICATION=.*|export BIG_BENCH_DATAGEN_DFS_REPLICATION="2"|' ./tpcx-bb-v1.6.2/conf/userSettings.conf
  sed -i "s|^export BIG_BENCH_HDFS_ABSOLUTE_PATH=.*|export BIG_BENCH_HDFS_ABSOLUTE_PATH=\"$TMPDIR\"|" ./tpcx-bb-v1.6.2/conf/userSettings.conf
fi

(echo; echo YES) | ./tpcx-bb-v1.6.2/bin/bigBench dataGen -f $2 -m 1 -U

popd

mkdir -p "$1"  # Ensure the target directory exists
for table in $TMPDIR/benchmarks/bigbench/data/*/*.dat; do # Note, there is also bigbench/data_refresh
  mv "$table" "$1/$(basename "$table")"
done
