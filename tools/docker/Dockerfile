ARG buildImage="buildimg"
ARG builtImage="mlirdb"
FROM ubuntu:impish AS baseimg
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get -y install python3-pip python3-venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip3 install numpy
RUN pip3 install Cython
RUN pip3 install requests moz_sql_parser numpy pandas pyarrow
RUN apt-get install -y git cmake  ninja-build wget unzip
RUN mkdir /dbgen

FROM baseimg AS tpchgen
RUN cd /dbgen
RUN wget -q https://github.com/electrum/tpch-dbgen/archive/32f1c1b92d1664dba542e927d23d86ffa57aa253.zip -O /dbgen/tpch-dbgen.zip
RUN cd /dbgen; unzip -q /dbgen/tpch-dbgen.zip && mv /dbgen/tpch-dbgen-32f1c1b92d1664dba542e927d23d86ffa57aa253/* . && rm /dbgen/tpch-dbgen.zip
RUN cd /dbgen; make
RUN cd /dbgen; ./dbgen -f -s 1
RUN cd /dbgen; for table in ./*.tbl; do sed 's/|$//' "$table" >"$table.2"; rm "$table";  mv "$table.2" "$table"; done
RUN mkdir /tpch-1
COPY ./tools /opt/tools
RUN python3 /opt/tools/generate/generate.py /dbgen /tpch-1

FROM baseimg AS buildarrow
RUN apt-get -y install autoconf flex bison libjemalloc-dev libboost-dev \
                                 libboost-filesystem-dev \
                                 libboost-system-dev \
                                 libboost-regex-dev \
                                 python-dev libssl-dev
COPY . /arrow-src
WORKDIR /arrow-src
RUN git submodule init
RUN git submodule update arrow
RUN mkdir -p /build/arrow
RUN cmake arrow/cpp  -B /build/arrow -DARROW_PYTHON=ON
RUN cmake --build /build/arrow -j$(nproc)
RUN cmake --install /build/arrow --prefix /build/arrow/install
RUN cd arrow/python; python3 setup.py build_ext --inplace --extra-cmake-args="-DArrow_DIR=/build/arrow/install/lib/cmake/arrow -D ArrowPython_DIR=/build/arrow/install/lib/cmake/arrow"


FROM baseimg AS buildllvm
COPY . /llvm-src
WORKDIR /llvm-src
RUN git submodule init
RUN git submodule update llvm-project
RUN git submodule update torch-mlir
RUN python -m pip install -r torch-mlir/requirements.txt

RUN mkdir -p /build/llvm

RUN cmake -G Ninja llvm-project/llvm  -B /build/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="/llvm-src/torch-mlir" \
    -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR="/llvm-src/torch-mlir/external/llvm-external-projects/torch-mlir-dialects" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
RUN cmake --build /build/llvm -j$(nproc)

FROM baseimg AS buildimg
COPY --from=buildarrow /build/arrow/install /build/arrow/install
COPY --from=buildarrow /arrow-src/arrow/python /arrow-src/arrow/python
COPY --from=buildllvm /llvm-src/torch-mlir /llvm-src/torch-mlir
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




FROM ${buildImage} AS mlirdb
COPY . /repo
WORKDIR /repo
Run git submodule init
Run git submodule update tools/pybind11
RUN rm -r arrow torch-mlir
RUN cp -r /arrow-src/arrow arrow
#RUN cp -r /llvm-src/llvm-project llvm-project
RUN cp -r /llvm-src/torch-mlir torch-mlir
#RUN find /llvm-src | grep "llvm-config" && exit 1
RUN mkdir /build/mlirdb
RUN git submodule update tools/pybind11
RUN cmake -G Ninja . -B /build/mlirdb\
           -DMLIR_DIR=/build/llvm/lib/cmake/mlir\
           -DLLVM_EXTERNAL_LIT=/build/llvm/bin/llvm-lit\
           -DArrow_DIR="/build/arrow/install/lib/cmake/arrow"\
           -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release 
        
RUN cmake --build /build/mlirdb -j$(nproc)


FROM ${builtImage} as reproduce
WORKDIR /repo
COPY --from=tpchgen /tpch-1 ./resources/data/tpch-1
