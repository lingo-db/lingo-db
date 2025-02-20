set -e
ls -la
/opt/python/$1/bin/python3 -m venv venv
venv/bin/python3 -m pip install build pyarrow===19.0.0
venv/bin/python3 -c "import pyarrow; pyarrow.create_library_symlinks()"
ARROW_LIB_DIR=$(venv/bin/python3 -c "import pyarrow as pa; print(':'.join(pa.get_library_dirs()))")
cmake -G Ninja . -B build/lingodb-release/ -DCMAKE_BUILD_TYPE=Release -DClang_DIR=/built-llvm/lib/cmake/clang -DArrow_DIR=/built-arrow/lib64/cmake/Arrow

cmake --build build/lingodb-release --target pybridge -j$(nproc)
cp -r tools/python/bridge build/pylingodb
cp -r tools/python/bridgelib/custom_dialects.h build/pylingodb/src/extensions/.
cp -r tools/python/bridgelib/bridge.h build/pylingodb/src/extensions/.

BASE_PATH=$(pwd)
cd build/pylingodb
mkdir -p src/lingodbbridge/mlir/dialects
MLIR_BIN_DIR=/built-llvm/bin
MLIR_INCLUDE_DIR=/built-llvm/include
MLIR_PYTHON_BASE=/built-llvm/python_packages/mlir_core
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_ods_common.py src/lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_func_ops_gen.py src/lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/func.py src/lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_arith_ops_gen.py src/lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_arith_enum_gen.py src/lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/arith.py src/lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_scf_ops_gen.py src/lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/scf.py src/lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_builtin_ops_gen.py src/lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/builtin.py src/lingodbbridge/mlir/dialects/.
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=util -I ${MLIR_INCLUDE_DIR} -I ${BASE_PATH}/include/ dialects/UtilOps.td  -I  ${BASE_PATH}/include/lingodb/compiler/Dialect/util > src/lingodbbridge/mlir/dialects/_util_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=tuples -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ dialects/TupleStreamOps.td > src/lingodbbridge/mlir/dialects/_tuples_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=db -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ dialects/DBOps.td > src/lingodbbridge/mlir/dialects/_db_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-enum-bindings -bind-dialect=db -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ dialects/DBOps.td > src/lingodbbridge/mlir/dialects/_db_enum_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=relalg -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/  -I  ${BASE_PATH}/include/lingodb/compiler/Dialect/RelAlg/IR dialects/RelAlgOps.td > src/lingodbbridge/mlir/dialects/_relalg_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-enum-bindings -bind-dialect=relalg -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ -I  ${BASE_PATH}/include/lingodb/compiler/Dialect/RelAlg/IR dialects/RelAlgOps.td > src/lingodbbridge/mlir/dialects/_relalg_enum_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=subop -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/  -I  ${BASE_PATH}/include/lingodb/compiler/Dialect/SubOperator dialects/SubOperatorOps.td > src/lingodbbridge/mlir/dialects/_subop_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-enum-bindings -bind-dialect=subop -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ -I  ${BASE_PATH}/include/lingodb/compiler/Dialect/SubOperator dialects/SubOperatorOps.td > src/lingodbbridge/mlir/dialects/_subop_enum_gen.py

cp -r /llvm-src/mlir/lib/Bindings/Python src/extensions/mlir_vendored
mkdir -p src/lingodbbridge/libs
cp ../lingodb-release/tools/python/bridgelib/libpybridge.so  src/lingodbbridge/libs/.

$BASE_PATH/venv/bin/python3 -m build --wheel
auditwheel repair dist/*.whl --plat "$PLAT" --exclude libarrow_python.so.1900 --exclude libarrow.so.1900 -w /built-packages
