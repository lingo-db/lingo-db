TMPDIR=`mktemp --directory`
pushd $TMPDIR
wget -q https://github.com/electrum/tpch-dbgen/archive/32f1c1b92d1664dba542e927d23d86ffa57aa253.zip -O tpch-dbgen.zip
unzip -q tpch-dbgen.zip
mv tpch-dbgen-32f1c1b92d1664dba542e927d23d86ffa57aa253/* .
rm tpch-dbgen.zip
make
./dbgen -f -s 1
for table in ./*.tbl; do sed 's/|$//' "$table" >"$table.2"; rm "$table";  mv "$table.2" "$table"; done

popd
mkdir -p resources/data/tpch-1
python3 tools/generate/generate.py $TMPDIR ./resources/data/tpch-1
