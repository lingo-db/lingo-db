FROM quay.io/pypa/manylinux_2_28_x86_64 AS baseimg
RUN yum install -y epel-release || sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(cut -d: -f5 /etc/system-release-cpe | cut -d. -f1).noarch.rpm
RUN yum install -y cmake ninja-build
RUN /opt/python/cp310-cp310/bin/python3 -m venv /venv
RUN /venv/bin/pip install numpy pybind11

ENV PLAT=manylinux_2_28_x86_64

FROM baseimg AS buildllvm
RUN mkdir /llvm-src
#COPY .git /llvm-src/.git
RUN git clone https://github.com/lingo-db/llvm-project.git /llvm-src/llvm-project
WORKDIR /llvm-src/llvm-project
RUN git checkout patched
WORKDIR /llvm-src
#RUN git submodule init
#RUN git submodule update llvm-project
RUN mkdir -p /build/llvm
RUN env VIRTUAL_ENV=/venv cmake -G Ninja llvm-project/llvm  -B /build/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
RUN cmake --build /build/llvm -j$(nproc)
RUN mkdir /installed
RUN cmake  --install /build/llvm --prefix /installed
COPY create_package.sh /usr/bin/
RUN chmod +x /usr/bin/create_package.sh
VOLUME [ "/repo" ]
WORKDIR /repo


#FROM baseimg AS buildimg
#COPY --from=buildllvm /llvm-src/llvm-project/llvm /llvm-src/llvm-project/llvm
#COPY --from=buildllvm /llvm-src/llvm-project/mlir /llvm-src/llvm-project/mlir
#COPY --from=buildllvm /llvm-src/llvm-project/clang /llvm-src/llvm-project/clang
#COPY --from=buildllvm /build/llvm/bin/mlir-tblgen /build/llvm/bin/mlir-tblgen
#COPY --from=buildllvm /build/llvm/bin/llvm-lit /build/llvm/bin/llvm-lit
#COPY --from=buildllvm /build/llvm/bin/llvm-link /build/llvm/bin/llvm-link
#COPY --from=buildllvm /build/llvm/bin/clang /build/llvm/bin/clang
#COPY --from=buildllvm /build/llvm/bin/clang-tidy /build/llvm/bin/clang-tidy
#COPY --from=buildllvm /build/llvm/bin/FileCheck /build/llvm/bin/FileCheck
#COPY --from=buildllvm /build/llvm/bin/count /build/llvm/bin/count
#COPY --from=buildllvm /build/llvm/bin/not /build/llvm/bin/not
#COPY --from=buildllvm /build/llvm/lib /build/llvm/lib
#COPY --from=buildllvm /build/llvm/include /build/llvm/include
#COPY --from=buildllvm /build/llvm/tools/mlir/include /build/llvm/tools/mlir/include
#COPY --from=buildllvm /build/llvm/tools/clang/include /build/llvm/tools/clang/include
#COPY create_package.sh /usr/bin/
#RUN chmod +x /usr/bin/create_package.sh
#VOLUME [ "/repo" ]
#WORKDIR /repo