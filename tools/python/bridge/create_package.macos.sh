set -ex

# Usage: create_package.macos.sh [PYVER]
#   PYVER: target Python version for the produced wheel (e.g. 3.13, 3.14).
#          Default 3.13. Re-running with a different PYVER reuses the
#          shared LLVM/Arrow/pybridge builds (which are Python-ABI-free).
PYVER=${1:-3.13}
PYTAG=${PYVER//./}                          # 313, 314 — used as dir suffix
PYTHON_BIN=/opt/homebrew/bin/python${PYVER}

# Python interpreter used while *building* LLVM. LLVM only invokes it for
# codegen scripts (mlir-tblgen helpers etc.); nothing in the install ends up
# linked into the wheel, so this can stay pinned independent of PYVER.
LLVM_BUILD_PYTHON=python3.13

BASE_PATH=$(pwd) # this must be the path to the lingoDB repo
LINGO_BUILD_DIR=$(pwd)/build/lingodb-release-py
LLVM_INSTALL_DIR=$(pwd)/build/llvm-install
LLVM_BUILD_DIR=$(pwd)/build/llvm-build
ARROW_INSTALL_DIR=$(pwd)/build/arrow-install
ARROW_BUILD_DIR=$(pwd)/build/arrow-build

# Build LLVM if not already installed or version is not 20.1.2
if [ ! -d "$LLVM_INSTALL_DIR" ] || [ ! -f "$LLVM_INSTALL_DIR/bin/llvm-config" ] || [ "$($LLVM_INSTALL_DIR/bin/llvm-config --version)" != "20.1.2" ]; then
  echo "LLVM not found. Building LLVM..."
  mkdir -p $LLVM_BUILD_DIR
  cd $LLVM_BUILD_DIR
  /opt/homebrew/bin/${LLVM_BUILD_PYTHON} -m venv ./venv
  ./venv/bin/pip install numpy pybind11 nanobind
  wget -nc https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.2/llvm-project-20.1.2.src.tar.xz
  tar -xf llvm-project-20.1.2.src.tar.xz
  rm llvm-project-20.1.2.src.tar.xz
  mkdir -p build
  export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
  env VIRTUAL_ENV=$LLVM_BUILD_DIR/venv cmake -B build -DPython3_FIND_VIRTUALENV=ONLY -DPython3_EXECUTABLE=./venv/bin/${LLVM_BUILD_PYTHON} -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang;clang-tools-extra" -DLLVM_TARGETS_TO_BUILD="AArch64" -DLLVM_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release -G Ninja -DLLVM_ENABLE_ASSERTIONS=OFF -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_BUILD_TESTS=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=OFF -DLLVM_ENABLE_DUMP=ON -DLLVM_ENABLE_FFI=ON -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer" -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_PARALLEL_TABLEGEN_JOBS=10 -DBUILD_SHARED_LIBS=OFF -DLLVM_INSTALL_UTILS=ON  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_ZLIB=OFF -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_FIND_FRAMEWORK=LAST -Wno-dev -DBUILD_TESTING=OFF -DCMAKE_OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path) -DLLVM_ENABLE_EH=OFF -DLLVM_ENABLE_FFI=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_INCLUDE_DOCS=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INSTALL_UTILS=ON -DLLVM_ENABLE_Z3_SOLVER=ON -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_USE_RELATIVE_PATHS_IN_FILES=ON -DLLVM_SOURCE_PREFIX=./llvm-project-20.1.2.src -DLLDB_USE_SYSTEM_DEBUGSERVER=ON -DLIBCXX_INSTALL_MODULES=ON -DLLVM_CREATE_XCODE_TOOLCHAIN=OFF -DCLANG_FORCE_MATCHING_LIBCLANG_SOVERSION=OFF -DFFI_INCLUDE_DIR=$(xcrun --sdk macosx --show-sdk-path)/usr/include/ffi -DFFI_LIBRARY_DIR=/$(xcrun --sdk macosx --show-sdk-path)/usr/lib -DLLVM_ENABLE_LIBCXX=ON -DLIBCXX_PSTL_BACKEND=libdispatch -DLIBCXX_INSTALL_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib/c++ -DLIBUNWIND_INSTALL_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib/unwind -DLIBCXXABI_INSTALL_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib/c++ -DRUNTIMES_CMAKE_ARGS="-DCMAKE_INSTALL_RPATH=@loader_path|@loader_path/../unwind" -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/homebrew ./llvm-project-20.1.2.src/llvm/
  cmake --build build --target install -j$(sysctl -n hw.logicalcpu)
  # Stash the MLIR Python bindings source inside the install dir so a cache
  # restore of $LLVM_INSTALL_DIR is self-contained — without this the per-wheel
  # cp below fails on cache hits because $LLVM_BUILD_DIR isn't cached.
  cp -r ./llvm-project-20.1.2.src/mlir/lib/Bindings/Python $LLVM_INSTALL_DIR/mlir-python-bindings-src
else
  echo "LLVM version 20.1.2 is already installed. Skipping build."
fi

# Build Arrow if not already installed or version is not 24.0.0
if [ ! -d "$ARROW_INSTALL_DIR" ] || [ ! -f "$ARROW_INSTALL_DIR/lib/cmake/Arrow/ArrowConfig.cmake" ] || [ "$(grep -Eo 'PACKAGE_VERSION[[:space:]]+"[0-9]+\.[0-9]+\.[0-9]+"\)' $ARROW_INSTALL_DIR/lib/cmake/Arrow/ArrowConfigVersion.cmake | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+')" != "24.0.0" ]; then
  echo "Arrow not found or incorrect version. Building Arrow..."
  mkdir -p $ARROW_BUILD_DIR
  cd $ARROW_BUILD_DIR
  wget -nc https://archive.apache.org/dist/arrow/arrow-24.0.0/apache-arrow-24.0.0.tar.gz
  tar -xf apache-arrow-24.0.0.tar.gz
  rm apache-arrow-24.0.0.tar.gz
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$ARROW_INSTALL_DIR -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_BUILD_STATIC=ON -DARROW_CSV=ON -DARROW_COMPUTE=ON -DCMAKE_PREFIX_PATH=/opt/homebrew/ -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang apache-arrow-24.0.0/cpp
  cmake --build build --target install -j$(sysctl -n hw.logicalcpu)
else
  echo "Arrow version 24.0.0 is already installed. Skipping build."
fi

# Per-wheel virtualenv (Python ABI-bound).
WHEEL_VENV=$BASE_PATH/build/venv-${PYTAG}
${PYTHON_BIN} -m venv $WHEEL_VENV
$WHEEL_VENV/bin/python3 -m pip install build pyarrow===24.0.0
$WHEEL_VENV/bin/python3 -c "import pyarrow; pyarrow.create_library_symlinks()"

# Build the C++ pybridge library. Shared across PYVER values — pybridge does
# not link Python itself (see tools/python/bridgelib/CMakeLists.txt), so
# ninja's incremental rebuild makes the second invocation a no-op.
cd $BASE_PATH
cmake -G Ninja . -B $LINGO_BUILD_DIR -DCMAKE_BUILD_TYPE=Release -DClang_DIR=$LLVM_INSTALL_DIR/lib/cmake/clang -DArrow_DIR=$ARROW_INSTALL_DIR/lib64/cmake/Arrow  -DENABLE_TESTS=OFF -DCMAKE_PREFIX_PATH=/opt/homebrew -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ -DCMAKE_OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path)
cmake --build $LINGO_BUILD_DIR --target pybridge -j$(sysctl -n hw.logicalcpu)

# Per-wheel staging directory (rebuild fresh each time so dist/*.whl is
# unambiguous and the previous PYVER's vendored sources are gone).
PYLINGODB_DIR=$BASE_PATH/build/pylingodb-${PYTAG}
rm -rf $PYLINGODB_DIR
cp -r tools/python/bridge $PYLINGODB_DIR
cp -r tools/python/bridgelib/custom_dialects.h $PYLINGODB_DIR/src/extensions/.
cp -r tools/python/bridgelib/bridge.h $PYLINGODB_DIR/src/extensions/.

cd $PYLINGODB_DIR
mkdir -p src/lingodbbridge/mlir/dialects
mkdir -p src/lingodbbridge/mlir/extras
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
cp -L ${MLIR_PYTHON_BASE}/mlir/extras/meta.py src/lingodbbridge/mlir/extras/.
cp -L ${MLIR_PYTHON_BASE}/mlir/extras/types.py src/lingodbbridge/mlir/extras/.
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=util -I ${MLIR_INCLUDE_DIR} -I ${BASE_PATH}/include/ dialects/UtilOps.td  -I  ${BASE_PATH}/include/lingodb/compiler/Dialect/util > src/lingodbbridge/mlir/dialects/_util_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=tuples -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ dialects/TupleStreamOps.td > src/lingodbbridge/mlir/dialects/_tuples_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=db -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ dialects/DBOps.td > src/lingodbbridge/mlir/dialects/_db_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-enum-bindings -bind-dialect=db -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ dialects/DBOps.td > src/lingodbbridge/mlir/dialects/_db_enum_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=relalg -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/  -I  ${BASE_PATH}/include/lingodb/compiler/Dialect/RelAlg/IR dialects/RelAlgOps.td > src/lingodbbridge/mlir/dialects/_relalg_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-enum-bindings -bind-dialect=relalg -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ -I  ${BASE_PATH}/include/lingodb/compiler/Dialect/RelAlg/IR dialects/RelAlgOps.td > src/lingodbbridge/mlir/dialects/_relalg_enum_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-op-bindings -bind-dialect=subop -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/  -I  ${BASE_PATH}/include/lingodb/compiler/Dialect/SubOperator dialects/SubOperatorOps.td > src/lingodbbridge/mlir/dialects/_subop_ops_gen.py
${MLIR_BIN_DIR}/mlir-tblgen -gen-python-enum-bindings -bind-dialect=subop -I ${MLIR_INCLUDE_DIR} -I  ${BASE_PATH}/include/ -I  ${BASE_PATH}/include/lingodb/compiler/Dialect/SubOperator dialects/SubOperatorOps.td > src/lingodbbridge/mlir/dialects/_subop_enum_gen.py

cp -r ${LLVM_INSTALL_DIR}/mlir-python-bindings-src src/extensions/mlir_vendored
mkdir -p src/lingodbbridge/libs
cp $LINGO_BUILD_DIR/tools/python/bridgelib/libpybridge.dylib src/lingodbbridge/libs/.

$WHEEL_VENV/bin/python3 -m build --wheel --config-setting cmake.define.LLVM_DIR=$LLVM_INSTALL_DIR/ --config-setting cmake.define.PYARROW_LIBRARY_DIRS=$WHEEL_VENV/lib/python${PYVER}/site-packages/pyarrow/ --config-setting cmake.define.PYARROW_INCLUDE_DIR=$WHEEL_VENV/lib/python${PYVER}/site-packages/pyarrow/include

# Install delocate if not already installed
$WHEEL_VENV/bin/python3 -m pip install delocate
$WHEEL_VENV/bin/delocate-wheel -v dist/*.whl -e libarrow_python.2400.dylib -e libarrow.2400.dylib -w ./build-packages --ignore-missing-dependencies
