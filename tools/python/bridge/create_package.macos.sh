set -ex

BASE_PATH=$(pwd) # this must be the path to the lingoDB repo
LINGO_BUILD_DIR=$(pwd)/build/lingodb-release
LLVM_INSTALL_DIR=$(pwd)/build/llvm-install
LLVM_BUILD_DIR=$(pwd)/build/llvm-build
ARROW_INSTALL_DIR=$(pwd)/build/arrow-install
ARROW_BUILD_DIR=$(pwd)/build/arrow-build

# Build LLVM if not already installed or version is not 20.1.2
if [ ! -d "$LLVM_INSTALL_DIR" ] || [ ! -f "$LLVM_INSTALL_DIR/bin/llvm-config" ] || [ "$($LLVM_INSTALL_DIR/bin/llvm-config --version)" != "20.1.2" ]; then
  echo "LLVM not found. Building LLVM..."
  mkdir -p $LLVM_BUILD_DIR
  cd $LLVM_BUILD_DIR
  /opt/homebrew/bin/python3 -m venv ./venv
  ./venv/bin/pip install numpy pybind11 nanobind
  wget -nc https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.2/llvm-project-20.1.2.src.tar.xz
  tar -xf llvm-project-20.1.2.src.tar.xz
  rm llvm-project-20.1.2.src.tar.xz
  mkdir -p build
  export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
  env VIRTUAL_ENV=$LLVM_BUILD_DIR/venv cmake -B build -DPython3_FIND_VIRTUALENV=ONLY -DPython3_EXECUTABLE=./venv/bin/python3.13 -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang;clang-tools-extra" -DLLVM_TARGETS_TO_BUILD="AArch64" -DLLVM_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release -G Ninja -DLLVM_ENABLE_ASSERTIONS=OFF -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_BUILD_TESTS=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=OFF -DLLVM_ENABLE_DUMP=ON -DLLVM_ENABLE_FFI=ON -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer" -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_PARALLEL_TABLEGEN_JOBS=10 -DBUILD_SHARED_LIBS=OFF -DLLVM_INSTALL_UTILS=ON  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_ZLIB=OFF -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_FIND_FRAMEWORK=LAST -Wno-dev -DBUILD_TESTING=OFF -DCMAKE_OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path) -DLLVM_ENABLE_EH=OFF -DLLVM_ENABLE_FFI=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_INCLUDE_DOCS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_Z3_SOLVER=ON -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_USE_RELATIVE_PATHS_IN_FILES=ON -DLLVM_SOURCE_PREFIX=./llvm-project-20.1.2.src -DLLDB_USE_SYSTEM_DEBUGSERVER=ON -DLIBCXX_INSTALL_MODULES=ON -DLLVM_CREATE_XCODE_TOOLCHAIN=OFF -DCLANG_FORCE_MATCHING_LIBCLANG_SOVERSION=OFF -DFFI_INCLUDE_DIR=$(xcrun --sdk macosx --show-sdk-path)/usr/include/ffi -DFFI_LIBRARY_DIR=/$(xcrun --sdk macosx --show-sdk-path)/usr/lib -DLLVM_ENABLE_LIBCXX=ON -DLIBCXX_PSTL_BACKEND=libdispatch -DLIBCXX_INSTALL_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib/c++ -DLIBUNWIND_INSTALL_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib/unwind -DLIBCXXABI_INSTALL_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib/c++ -DRUNTIMES_CMAKE_ARGS="-DCMAKE_INSTALL_RPATH=@loader_path|@loader_path/../unwind" -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/homebrew ./llvm-project-20.1.2.src/llvm/
  cmake --build build --target install -j$(sysctl -n hw.logicalcpu)
else
  echo "LLVM version 20.1.2 is already installed. Skipping build."
fi

# Build Arrow if not already installed or version is not 20.0.0
if [ ! -d "$ARROW_INSTALL_DIR" ] || [ ! -f "$ARROW_INSTALL_DIR/lib/cmake/Arrow/ArrowConfig.cmake" ] || [ "$(grep -Eo 'PACKAGE_VERSION[[:space:]]+"[0-9]+\.[0-9]+\.[0-9]+"\)' $ARROW_INSTALL_DIR/lib/cmake/Arrow/ArrowConfigVersion.cmake | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+')" != "20.0.0" ]; then
  echo "Arrow not found or incorrect version. Building Arrow..."
  mkdir -p $ARROW_BUILD_DIR
  cd $ARROW_BUILD_DIR
  wget -nc https://github.com/apache/arrow/releases/download/apache-arrow-20.0.0-rc1/apache-arrow-20.0.0.tar.gz
  tar -xf apache-arrow-20.0.0.tar.gz
  rm apache-arrow-20.0.0.tar.gz
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$ARROW_INSTALL_DIR -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_BUILD_STATIC=ON -DARROW_CSV=ON -DARROW_COMPUTE=ON -DCMAKE_PREFIX_PATH=/opt/homebrew/ -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/clang++ -DCMAKE_C_COMPILER=/opt/homebrew/bin/clang apache-arrow-20.0.0/cpp
  cmake --build build --target install -j$(sysctl -n hw.logicalcpu)
else
  echo "Arrow version 20.0.0 is already installed. Skipping build."
fi

/opt/homebrew/bin/python3 -m venv $BASE_PATH/build/venv
$BASE_PATH/build/venv/bin/python3 -m pip install build pyarrow===19.0.0
$BASE_PATH/build/venv/bin/python3 -c "import pyarrow; pyarrow.create_library_symlinks()"

# Build LingoDB
cd $BASE_PATH
cmake -G Ninja . -B build/lingodb-release-py/ -DCMAKE_BUILD_TYPE=Release -DClang_DIR=$LLVM_INSTALL_DIR/lib/cmake/clang -DArrow_DIR=$ARROW_INSTALL_DIR/lib64/cmake/Arrow  -DENABLE_TESTS=OFF -DCMAKE_PREFIX_PATH=/opt/homebrew/ -DCMAKE_OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path)
cmake --build build/lingodb-release-py --target pybridge -j$(sysctl -n hw.logicalcpu)
cp -r tools/python/bridge build/pylingodb
cp -r tools/python/bridgelib/custom_dialects.h build/pylingodb/src/extensions/.
cp -r tools/python/bridgelib/bridge.h build/pylingodb/src/extensions/.

cd build/pylingodb
mkdir -p src/lingodbbridge/mlir/dialects
MLIR_BIN_DIR=$LLVM_INSTALL_DIR/bin
MLIR_INCLUDE_DIR=$LLVM_INSTALL_DIR/include
MLIR_PYTHON_BASE=$LLVM_INSTALL_DIR/python_packages/mlir_core
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

cp -r ${LLVM_BUILD_DIR}/llvm-project-20.1.2.src/mlir/lib/Bindings/Python src/extensions/mlir_vendored
mkdir -p src/lingodbbridge/libs
cp $BASE_PATH/build/lingodb-release-py/tools/python/bridgelib/libpybridge.dylib src/lingodbbridge/libs/.

$BASE_PATH/build/venv/bin/python3 -m build --wheel --config-setting cmake.define.LLVM_DIR=$LLVM_INSTALL_DIR/ --config-setting cmake.define.PYARROW_LIBRARY_DIRS=$BASE_PATH/build/venv/lib/python3.13/site-packages/pyarrow/ --config-setting cmake.define.PYARROW_INCLUDE_DIR=$BASE_PATH/build/venv/lib/python3.13/site-packages/pyarrow/include

# Install delocate if not already installed
$BASE_PATH/build/venv/bin/python3 -m pip install delocate
$BASE_PATH/build/venv/bin/delocate-wheel -v dist/*.whl -e libarrow_python.1900.dylib -e libarrow.1900.dylib -w ./build-packages --ignore-missing-dependencies
