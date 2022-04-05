## To reproduce results with docker
```
make reproduce
```
Note that docker comes with a slight runtime overhead (passing `--privileged` helps a bit). Measurements in the paper were conducted using native builds.

## For developers

1. Ensure you have a machine with sufficient compute power and space (16 GiB RAM, >16 GiB disk space, preferably many cores).
1. Initialize submodules: `git submodule update --init --recursive`
1. Install (system) dependencies:
    - Ubuntu: `libjemalloc-dev libboost-dev libboost-filesystem-dev libboost-system-dev libboost-regex-dev python-dev autoconf flex bison`
    - Fedora: `autoconf cmake ninja jemalloc-devel boost-devel boost-filesystem boost-system python3-devel flex bison lcov`
1. Setup a Python virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   . venv/bin/activate
   pip install -r torch-mlir/requirements.txt
   pip install requests Cython moz_sql_parser numpy pandas pyarrow
   ````
1. Build dependencies: `make dependencies` (this builds LLVM/MLIR/Clang, Arrow and PyArrow)
1. Build & test: `make run-test` (this compiles in debug mode; incremental builds are supported)
1. Run TPC-H (SF=1) benchmarks: `make run-benchmark` (this generates TPC-H data and compiles in release mode; incremental builds are supported)
