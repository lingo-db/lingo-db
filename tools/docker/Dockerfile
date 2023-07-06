ARG buildImage="buildimg"
ARG builtImage="lingodb"
FROM ubuntu:jammy AS baseimg
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get -y install python3-pip python3-venv libtbb-dev
RUN apt install -y -V ca-certificates lsb-release wget
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt-get update
RUN apt install -y libarrow1200 libarrow-dev
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip3 install numpy pandas Cython pybind11
RUN apt-get install -y git cmake  ninja-build wget unzip ccache curl
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
RUN cargo install --force cbindgen
RUN mkdir /dbgen

FROM baseimg AS buildllvm
COPY .git /llvm-src/.git
WORKDIR /llvm-src
RUN git submodule init
RUN git submodule update llvm-project

RUN mkdir -p /build/llvm

RUN cmake -G Ninja llvm-project/llvm  -B /build/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
RUN cmake --build /build/llvm -j$(nproc)

FROM baseimg AS buildimg
COPY --from=buildllvm /llvm-src/llvm-project/llvm /llvm-src/llvm-project/llvm
COPY --from=buildllvm /llvm-src/llvm-project/mlir /llvm-src/llvm-project/mlir
COPY --from=buildllvm /llvm-src/llvm-project/clang /llvm-src/llvm-project/clang
COPY --from=buildllvm /build/llvm/bin/mlir-tblgen /build/llvm/bin/mlir-tblgen
COPY --from=buildllvm /build/llvm/bin/llvm-lit /build/llvm/bin/llvm-lit
COPY --from=buildllvm /build/llvm/bin/llvm-link /build/llvm/bin/llvm-link
COPY --from=buildllvm /build/llvm/bin/clang /build/llvm/bin/clang
COPY --from=buildllvm /build/llvm/bin/clang-tidy /build/llvm/bin/clang-tidy
COPY --from=buildllvm /build/llvm/bin/FileCheck /build/llvm/bin/FileCheck
COPY --from=buildllvm /build/llvm/bin/count /build/llvm/bin/count
COPY --from=buildllvm /build/llvm/bin/not /build/llvm/bin/not
COPY --from=buildllvm /build/llvm/lib /build/llvm/lib
COPY --from=buildllvm /build/llvm/include /build/llvm/include
COPY --from=buildllvm /build/llvm/tools/mlir/include /build/llvm/tools/mlir/include
COPY --from=buildllvm /build/llvm/tools/clang/include /build/llvm/tools/clang/include

FROM ${buildImage} AS lingodb
COPY . /repo
WORKDIR /repo
Run git submodule init
RUN mkdir /build/lingodb
RUN cmake -G Ninja . -B /build/lingodb\
           -DMLIR_DIR=/build/llvm/lib/cmake/mlir\
           -DLLVM_EXTERNAL_LIT=/build/llvm/bin/llvm-lit\
           -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release 
        
RUN cmake --build /build/lingodb -j$(nproc)


FROM ${builtImage} as reproduce
WORKDIR /repo
RUN mkdir -p resources/data/tpch-1
RUN bash tools/generate/tpch.sh /build/lingodb /repo/resources/data/tpch-1 1