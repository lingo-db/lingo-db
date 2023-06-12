set -e 
mkdir /build/lingodb
cmake -G Ninja . -B /build/lingodb -DMLIR_DIR=/build/llvm/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/build/llvm/bin/llvm-lit -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release 
cmake --build /build/lingodb --target pybridge -j$(nproc)
cp -r tools/python /build/pylingodb
cd /build/pylingodb
mkdir -p pylingodb/libs
cp /build/lingodb/tools/python/libpybridge.so pylingodb/libs/.
"/opt/python/$1/bin/python3" -m build --wheel
auditwheel repair dist/*.whl --plat "$PLAT" --exclude libarrow_python.so --exclude libarrow.so.1200 -w /built-packages
