FROM ubuntu:noble AS devimg
RUN apt-get update && apt-get -y install software-properties-common wget
RUN add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main"
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename  --short).deb && rm *.deb
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main"
RUN apt-get update && apt-get -y install python3 python3-venv python3-pip git g++ cmake ninja-build wget unzip ccache curl lsb-release wget zlib1g-dev lcov clang-20 llvm-20 libclang-20-dev llvm-20-dev libmlir-20-dev mlir-20-tools clang-tidy-20 libarrow-dev=24.* libarrow-compute-dev=24.* libparquet-dev=24.* libboost-context1.83-dev catch2 flex bison libabsl-dev
# Pin lit: lit >= 23 (LLVM-23) dropped support for the bare `VAR=val cmd`
# environment-prefix form used by many test/lit RUN lines (it raises on
# ShTest execute_external=True and its shell no longer parses the prefix).
# 18.1.8 keeps the behavior the tests were written against.
RUN pip3 install --break-system-packages lit==18.1.8
ENV CC=clang-20 CXX=clang++-20
RUN git clone https://github.com/lingo-db/llvmcov2html.git /llvmcov2html && cd /llvmcov2html && git checkout 34603a1 && make && cp bin/llvmcov2html /usr/bin/. && cd / && rm -rf /llvmcov2html

# --- Free-threaded CPython for the embedded Python-UDF runtime ----------------
# The Python-UDF runtime runs every worker in ONE shared interpreter with no GIL
# (src/runtime/ExecutionContext.cpp), so an embedded build (ENABLE_PYTHON=CPYTHON)
# needs a free-threaded interpreter. uv ships a standalone cp314t with headers +
# libpython3.14t.so (the standalone binaries embed the interpreter, so headers
# alone are not enough). /opt/venv314t carries the free-threaded pyarrow/numpy/
# pandas the tabular/numpy UDF tests import; point the embedded interpreter at it
# with PYTHONPATH=/opt/venv314t/lib/python3.14t/site-packages.
ENV UV_PYTHON_INSTALL_DIR=/opt/pythons
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh \
    && uv python install 3.14t \
    && ln -sf "$(uv python find 3.14t)" /usr/local/bin/python3.14t \
    && uv venv --python 3.14t /opt/venv314t \
    && uv pip install --python /opt/venv314t/bin/python pyarrow==24.0.0 numpy pandas
# CMake >= 3.30 is required for free-threaded FindPython (the apt cmake is 3.28
# and can't detect the "t" ABI — it would silently fall back to the
# non-free-threaded system Python). Installed at a private prefix so the default
# apt cmake other jobs rely on is untouched; the CPYTHON UDF build opts in via
# CMAKE=/opt/cmake-ft/bin/cmake.
# uname -m is x86_64 / aarch64, which matches Kitware's tarball naming exactly,
# so this works for both linux/amd64 and linux/arm64 image builds.
RUN CMAKE_ARCH="$(uname -m)" \
    && wget -q "https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-${CMAKE_ARCH}.tar.gz" \
    && tar xzf "cmake-3.31.6-linux-${CMAKE_ARCH}.tar.gz" -C /opt \
    && mv "/opt/cmake-3.31.6-linux-${CMAKE_ARCH}" /opt/cmake-ft \
    && rm "cmake-3.31.6-linux-${CMAKE_ARCH}.tar.gz"
