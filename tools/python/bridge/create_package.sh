set -e
ls -la
ln -s "/opt/python/$1/bin/python3" /usr/bin/python3
python3 -v
make venv
MLIR_PYTHON_BASE=$(venv/bin/python3 -c "import lingodbllvm; print(lingodbllvm.get_py_package_dir()+'/mlir_core')")
MLIR_BIN_DIR=$(venv/bin/python3 -c "import lingodbllvm; print(lingodbllvm.get_bin_dir())")
MLIR_INCLUDE_DIR=$(venv/bin/python3 -c "import lingodbllvm; print(lingodbllvm.get_bin_dir()+'/../include/')")
make build/lingodb-release/.stamp
cmake --build build/lingodb-release --target pybridge -j$(nproc)
cp -r tools/python/bridge build/pylingodb
BASE_PATH=$(pwd)
cd build/pylingodb
mkdir -p lingodbbridge/mlir/dialects
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_ods_common.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_func_ops_ext.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_func_ops_gen.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/func.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_arith_ops_ext.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_arith_ops_gen.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_arith_enum_gen.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/arith.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_scf_ops_ext.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_scf_ops_gen.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/scf.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_builtin_ops_ext.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/_builtin_ops_gen.py lingodbbridge/mlir/dialects/.
cp -L ${MLIR_PYTHON_BASE}/mlir/dialects/builtin.py lingodbbridge/mlir/dialects/.
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=util -I ${MLIR_INCLUDE_DIR} -I ${BASE_PATH}/include/ dialects/UtilOps.td > lingodbbridge/mlir/dialects/_util_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=tuples -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ dialects/TupleStreamOps.td > lingodbbridge/mlir/dialects/_tuples_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=db -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ dialects/DBOps.td > lingodbbridge/mlir/dialects/_db_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-enum-bindings -bind-dialect=db -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ dialects/DBOps.td > lingodbbridge/mlir/dialects/_db_enum_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=relalg -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/  -I  ${BASE_PATH}/include/mlir/Dialect/RelAlg/IR dialects/RelAlgOps.td > lingodbbridge/mlir/dialects/_relalg_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-enum-bindings -bind-dialect=relalg -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ -I  ${BASE_PATH}/include/mlir/Dialect/RelAlg/IR dialects/RelAlgOps.td > lingodbbridge/mlir/dialects/_relalg_enum_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=subop -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/  -I  ${BASE_PATH}/include/mlir/Dialect/SubOperator dialects/SubOperatorOps.td > lingodbbridge/mlir/dialects/_subop_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-enum-bindings -bind-dialect=subop -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ -I  ${BASE_PATH}/include/mlir/Dialect/SubOperator dialects/SubOperatorOps.td > lingodbbridge/mlir/dialects/_subop_enum_gen.py

mkdir -p lingodbbridge/libs
cp ../lingodb-release/tools/python/bridge/libpybridge.so  lingodbbridge/libs/.
python3 -m build --wheel
auditwheel repair dist/*.whl --plat "$PLAT" --exclude libarrow_python.so --exclude libarrow.so.1400 -w /built-packages
