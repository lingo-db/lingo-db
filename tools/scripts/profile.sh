rm profile.db
./build/build-debug-llvm-release/tools/sourcemap/sourcemap snapshot-0.mlir > sourcemap-0.json
./build/build-debug-llvm-release/tools/sourcemap/sourcemap snapshot-1.mlir > sourcemap-1.json
./build/build-debug-llvm-release/tools/sourcemap/sourcemap snapshot-2.mlir > sourcemap-2.json
./build/build-debug-llvm-release/tools/sourcemap/sourcemap snapshot-3.mlir > sourcemap-3.json
perf script  -s tools/scripts/extract-perf-data.py -i perf.data
./tools/venv/bin/python3 tools/profile-viz/create_db.py
./tools/venv/bin/python3 tools/profile-viz/analyze.py
