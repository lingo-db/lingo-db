set -e
mkdir /build/mlir-package
cp -r /repo/. /build/mlir-package/.
cd /build/mlir-package
ls -l
mkdir lingodbllvm/llvm
cp -r /installed/. lingodbllvm/llvm/.
"/opt/python/$1/bin/python3" -m build --wheel
cp dist/*.whl /built-packages/.

auditwheel repair dist/*.whl --plat "$PLAT" -w /built-packages
ls -l /built-packages
