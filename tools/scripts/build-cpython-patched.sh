BUILD_DIR="$(pwd)/build"
ROOT_DIR="$(pwd)"
INSTALL_DIR="${BUILD_DIR}/cpython-patched"


# Ensure the build directory exists
mkdir -p "${BUILD_DIR}/cpython-patched"
pushd "${BUILD_DIR}/cpython-patched"

# Download the CPython source code
wget https://github.com/python/cpython/archive/17c5d6d58dcb7c94757fdedac500d6e7ac1ce40e.zip
unzip 17c5d6d58dcb7c94757fdedac500d6e7ac1ce40e.zip
rm -f 17c5d6d58dcb7c94757fdedac500d6e7ac1ce40e.zip
cd cpython-17c5d6d58dcb7c94757fdedac500d6e7ac1ce40e
echo $(pwd)
patch -p1 < $ROOT_DIR/tools/scripts/disable_fiber_stack_check.patch

./configure \
    --prefix=$INSTALL_DIR \
    --enable-shared \
    CFLAGS="-fPIC" \
    LDFLAGS="-Wl,-rpath,$INSTALL_DIR"
make -j$(nproc)
make install