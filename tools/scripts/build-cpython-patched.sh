BUILD_DIR="$(pwd)/build"
ROOT_DIR="$(pwd)"
PYTHON_VERSION="3.14.0"
PYTHON_DIR="${BUILD_DIR}/cpython-patched/Python-${PYTHON_VERSION}"
INSTALL_DIR="${BUILD_DIR}/cpython-patched"

# Ensure the correct Python version is installed
REQUIRED_PYTHON_VERSION="3.12"
INSTALLED_PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if [[ $(printf '%s\n' "$REQUIRED_PYTHON_VERSION" "$INSTALLED_PYTHON_VERSION" | sort -V | head -n1) != "$REQUIRED_PYTHON_VERSION" ]]; then
    echo "Error: Python $REQUIRED_PYTHON_VERSION or higher is required. Installed version: $INSTALLED_PYTHON_VERSION"
    exit 1
fi

# Ensure the build directory exists
mkdir -p "${BUILD_DIR}/cpython-patched"
pushd "${BUILD_DIR}/cpython-patched"

# Download the CPython source code
curl -O "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz"
tar -xf "Python-$PYTHON_VERSION.tar.xz"
rm -f "Python-$PYTHON_VERSION.tar.xz"
cd ${PYTHON_DIR}
echo $(pwd)
patch -p1 < $ROOT_DIR/tools/scripts/disable_fiber_stack_check.patch

./configure \
    --prefix=$INSTALL_DIR \
    --enable-shared \
    CFLAGS="-fPIC" \
    LDFLAGS="-Wl,-rpath,$INSTALL_DIR"
make -j$(nproc)
make install