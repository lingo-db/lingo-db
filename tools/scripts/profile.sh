rm events.parquet
rm operations.parquet
./build/lingodb-debug/tools/sourcemap/sourcemap snapshot-0.mlir snapshot-1.mlir snapshot-2.mlir snapshot-3.mlir > sourcemap.json
perf script  -s tools/scripts/extract-perf-data.py -i perf.data
./tools/venv/bin/python3 tools/profile-viz/create_db.py
./tools/venv/bin/python3 tools/profile-viz/run.py
