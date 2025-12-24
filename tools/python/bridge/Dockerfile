FROM quay.io/pypa/manylinux_2_28_x86_64:2025.01.11-1 AS base-amd64
ENV PLAT=manylinux_2_28_x86_64
ENV NINJA_ARCHIVE=ninja-linux.zip
ENV LLVM_TARGET_ARCH=X86

FROM quay.io/pypa/manylinux_2_28_aarch64:2025.01.11-1 AS base-arm64
ENV PLAT=manylinux_2_28_aarch64
ENV NINJA_ARCHIVE=ninja-linux-aarch64.zip
ENV LLVM_TARGET_ARCH=AArch64

ARG TARGETARCH
FROM base-${TARGETARCH} AS baseimg

RUN yum install -y epel-release || sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(cut -d: -f5 /etc/system-release-cpe | cut -d. -f1).noarch.rpm
RUN yum install -y cmake unzip wget flex
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.12.1/${NINJA_ARCHIVE} && unzip ${NINJA_ARCHIVE} -d /usr/bin/ && rm ${NINJA_ARCHIVE}
RUN wget https://mirror.dogado.de/gnu/bison/bison-3.8.2.tar.gz \
    && tar -xf bison-3.8.2.tar.gz \
    && cd bison-3.8.2 \
    && ./configure --prefix=/usr \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf bison-3.8.2 bison-3.8.2.tar.gz

FROM baseimg AS buildllvm
RUN mkdir /built-llvm
RUN yum install -y clang
RUN /opt/python/cp312-cp312/bin/python3 -m venv /venv
RUN /venv/bin/pip install numpy pybind11 nanobind
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.0-rc1/llvm-project-20.1.0-rc1.src.tar.xz && tar -xf llvm-project-20.1.0-rc1.src.tar.xz && mkdir llvm-project-20.1.0-rc1.src/build && \
cd /llvm-project-20.1.0-rc1.src && \
env VIRTUAL_ENV=/venv cmake --log-level=ERROR -B build -DPython3_FIND_VIRTUALENV=ONLY -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang;clang-tools-extra" -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGET_ARCH}" -DLLVM_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release -G Ninja -DLLVM_ENABLE_ASSERTIONS=OFF -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_BUILD_TESTS=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=OFF -DLLVM_ENABLE_DUMP=ON -DLLVM_ENABLE_FFI=ON -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer" -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_PARALLEL_TABLEGEN_JOBS=10 -DBUILD_SHARED_LIBS=OFF -DLLVM_INSTALL_UTILS=ON  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_ZLIB=OFF -DCMAKE_INSTALL_PREFIX=/built-llvm llvm/ && \
cmake  --build build --target install  -- --quiet -j$(nproc) && cd .. && rm -rf llvm-project-20.1.0-rc1.src llvm-project-20.1.0-rc1.src.tar.xz


FROM baseimg AS devimg
COPY --from=buildllvm /built-llvm /built-llvm
RUN wget https://dlcdn.apache.org/arrow/arrow-22.0.0/apache-arrow-22.0.0.tar.gz && tar -xf apache-arrow-22.0.0.tar.gz && rm apache-arrow-22.0.0.tar.gz
RUN mkdir /built-arrow
RUN cd /apache-arrow-22.0.0/cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/built-arrow -DARROW_DEPENDENCY_SOURCE=AUTO -DARROW_BUILD_STATIC=ON -DARROW_CSV=ON -DARROW_COMPUTE=ON  \
      -DARROW_ENABLE_TIMING_TESTS="OFF" \
      -DARROW_BUILD_SHARED="OFF" \
    && cmake --build build --target install -j$(nproc) && cd .. && rm  -r /apache-arrow-22.0.0
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.0-rc1/llvm-project-20.1.0-rc1.src.tar.xz && tar -xf llvm-project-20.1.0-rc1.src.tar.xz && mv /llvm-project-20.1.0-rc1.src /llvm-src
RUN wget https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.gz && tar -xf boost_1_83_0.tar.gz && cd boost_1_83_0 && ./bootstrap.sh --prefix=/usr && ./b2 install --with-context && cd .. && rm -rf boost_1_83_0

FROM devimg AS build
WORKDIR /repo
COPY . /repo
RUN cp /repo/tools/python/bridge/create_package.linux.sh /usr/bin/
RUN chmod +x /usr/bin/create_package.linux.sh
