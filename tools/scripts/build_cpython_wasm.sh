#!/bin/bash
set -e
set -x

# Ensure required tools are installed
REQUIRED_TOOLS=("cmake" "clang" "ninja" "python3")
for TOOL in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$TOOL" &> /dev/null; then
        echo "Error: $TOOL is not installed."
        exit 1
    fi
done

# Ensure wasi-sdk is installed
BUILD_DIR="$(pwd)/build"
mkdir -p "${BUILD_DIR}"
export WASI_SDK_PATH="${BUILD_DIR}/wasi-sdk"

if [ ! -d "${WASI_SDK_PATH}" ]; then
  echo "WASI SDK not found. Downloading and installing..."
  if [[ "$(uname -m)" == "arm64" && "$(uname)" == "Darwin" ]]; then
      WASI_OS=apple
      WASI_ARCH=aarch64
  else
      WASI_OS=linux
      WASI_ARCH=x86_64
  fi
  WASI_VERSION=27
  WASI_VERSION_FULL=${WASI_VERSION}.0
  wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_VERSION}/wasi-sdk-${WASI_VERSION_FULL}-${WASI_ARCH}-${WASI_OS}.tar.gz -O wasi-sdk-${WASI_VERSION_FULL}-${WASI_ARCH}-${WASI_OS}.tar.gz
  tar xvf wasi-sdk-${WASI_VERSION_FULL}-${WASI_ARCH}-${WASI_OS}.tar.gz -C ${BUILD_DIR}
  rm -f wasi-sdk-${WASI_VERSION_FULL}-${WASI_ARCH}-${WASI_OS}.tar.gz
  mv ${BUILD_DIR}/wasi-sdk-${WASI_VERSION_FULL}-${WASI_ARCH}-${WASI_OS} ${BUILD_DIR}/wasi-sdk
else
  echo "WASI SDK found at ${WASI_SDK_PATH}"
fi

# Ensure wasmtime is installed
if ! command -v wasmtime &> /dev/null; then
    curl https://wasmtime.dev/install.sh -sSf | bash
    export PATH="$HOME/.wasmtime/bin:$PATH"
fi

# Basis and reference: https://devguide.python.org/getting-started/setup-building/#wasi
PYTHON_VERSION="3.14.0"
PYTHON_DIR="${BUILD_DIR}/cpython-wasm/Python-${PYTHON_VERSION}"

# Ensure the correct Python version is installed
REQUIRED_PYTHON_VERSION="3.12"
INSTALLED_PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if [[ $(printf '%s\n' "$REQUIRED_PYTHON_VERSION" "$INSTALLED_PYTHON_VERSION" | sort -V | head -n1) != "$REQUIRED_PYTHON_VERSION" ]]; then
    echo "Error: Python $REQUIRED_PYTHON_VERSION or higher is required. Installed version: $INSTALLED_PYTHON_VERSION"
    exit 1
fi

# Ensure the build directory exists
mkdir -p "${BUILD_DIR}/cpython-wasm"
pushd "${BUILD_DIR}/cpython-wasm"

# Download the CPython source code
curl -O "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz"
tar -xf "Python-$PYTHON_VERSION.tar.xz"
rm -f "Python-$PYTHON_VERSION.tar.xz"
cd ${PYTHON_DIR}

## Build CPython using wasi-sdk
# 1. Build native python (bootstrap)
mkdir -p cross-build/build
export SOURCE_DATE_EPOCH=1760114616
cd cross-build/build
../../configure
make --jobs $(nproc) all

# 2. Build python wasm through native python
mkdir -p ../wasm32-wasip1
cd ../wasm32-wasip1
# Important: we need to set `--export-all` as the LD flag here, since otherwise we only export the `_start` symbol for CLI use
export CFLAGS="--sysroot=${WASI_SDK_PATH}/share/wasi-sysroot"
export LDFLAGS="--sysroot=${WASI_SDK_PATH}/share/wasi-sysroot -Wl,--export-all"
export AR="${WASI_SDK_PATH}/bin/llvm-ar"
export CC="${WASI_SDK_PATH}/bin/clang --sysroot=/home/doellerer/CLionProjects/lingo-db/build/wasi-sdk/share/wasi-sysroot"
export CONFIG_SITE="${PYTHON_DIR}/Tools/wasm/wasi/config.site-wasm32-wasi"
export CPP="${WASI_SDK_PATH}/bin/clang-cpp --sysroot=/home/doellerer/CLionProjects/lingo-db/build/wasi-sdk/share/wasi-sysroot"
export CXX="${WASI_SDK_PATH}/bin/clang++ --sysroot=/home/doellerer/CLionProjects/lingo-db/build/wasi-sdk/share/wasi-sysroot"
export HOSTRUNNER="wasmtime run --wasm max-wasm-stack=16777216 --dir ${PYTHON_DIR}::/ --env PYTHONPATH=/cross-build/wasm32-wasip1/build/lib.wasi-wasm32-3.14"
export PATH="${WASI_SDK_PATH}/bin:${PATH}"
export PKG_CONFIG_LIBDIR="${WASI_SDK_PATH}/share/wasi-sysroot/lib/pkgconfig:${WASI_SDK_PATH}/share/wasi-sysroot/share/pkgconfig"
export PKG_CONFIG_SYSROOT_DIR=${WASI_SDK_PATH}/share/wasi-sysroot
export PKG_CONFIG_PATH=
export RANLIB=${WASI_SDK_PATH}/bin/ranlib
export SOURCE_DATE_EPOCH=1760114616
export WASI_SDK_PATH=${WASI_SDK_PATH}
export WASI_SYSROOT=${WASI_SDK_PATH}/share/wasi-sysroot

../../configure --host=wasm32-wasip1 --build=x86_64-pc-linux-gnu --with-build-python=${PYTHON_DIR}/cross-build/build/python
make --jobs $(nproc) all

popd
