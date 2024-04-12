set -e
mkdir /build/mlir-package
cp -r /repo/. /build/mlir-package/.
cd /build/mlir-package
ls -l
mkdir lingodbllvm/llvm
mkdir -p lingodbllvm/llvm/src
cp -r /installed/. lingodbllvm/llvm/.
cp -r /llvm-src/llvm-project/mlir/lib/Bindings/Python lingodbllvm/llvm/mlir_python_bindings
cp -r /llvm-src/llvm-project/mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp lingodbllvm/llvm/src/CudaRuntimeWrappers.cpp
"/opt/python/$1/bin/python3" -m build --wheel
cp dist/*.whl /built-packages/.

auditwheel repair dist/*.whl --plat "$PLAT" -w /built-packages
ls -l /built-packages
