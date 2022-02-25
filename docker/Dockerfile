ARG buildImage="buildimg"
FROM ubuntu:impish AS baseimg
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get -y install libjemalloc-dev libboost-dev \
                                 libboost-filesystem-dev \
                                 libboost-system-dev \
                                 libboost-regex-dev \
                                 python-dev \
                                 autoconf \
                                 flex \
                                 bison

RUN apt-get -y install python3-pip python3-venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip3 install numpy
RUN pip3 install Cython
RUN pip3 install requests moz_sql_parser numpy pandas pyarrow
RUN apt-get install -y clang-tidy git cmake libssl-dev ninja-build clang wget unzip
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


FROM baseimg AS buildimg
COPY . /repo
WORKDIR /repo



RUN git submodule init
RUN git submodule update arrow
RUN mkdir -p /build/arrow

RUN cmake arrow/cpp  -B /build/arrow -DARROW_PYTHON=ON
RUN cmake --build /build/arrow -j$(nproc)
RUN cmake --install /build/arrow --prefix /build/arrow/install
RUN cd arrow/python; python3 setup.py build_ext --inplace --extra-cmake-args="-DArrow_DIR=/build/arrow/install/lib/cmake/arrow -D ArrowPython_DIR=/build/arrow/install/lib/cmake/arrow"
RUN git submodule update llvm-project
RUN git submodule update torch-mlir
RUN python -m pip install -r torch-mlir/requirements.txt

RUN mkdir /build/llvm

RUN cmake -G Ninja llvm-project/llvm  -B /build/llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="/repo/torch-mlir" \
    -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR="/repo/torch-mlir/external/llvm-external-projects/torch-mlir-dialects" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
RUN cmake --build /build/llvm -j$(nproc)
FROM ${buildImage} AS mlirdb
COPY . /mlirdb-repo
WORKDIR /mlirdb-repo
RUN rm -r arrow pybind11 llvm-project torch-mlir
RUN cp -r /repo/arrow arrow
RUN cp -r /repo/pybind11 pybind11
RUN cp -r /repo/llvm-project llvm-project
RUN cp -r /repo/torch-mlir torch-mlir
RUN mkdir /build/mlirdb
RUN git submodule update pybind11
RUN cmake -G Ninja . -B /build/mlirdb\
           -DMLIR_DIR=/build/llvm/lib/cmake/mlir\
           -DLLVM_EXTERNAL_LIT=/build/llvm/bin/llvm-lit\
           -DArrow_DIR="/build/arrow/install/lib/cmake/arrow"\
           -DARROW_BUILD_DIR="/build/arrow"\
           -DBoost_INCLUDE_DIR="/build/arrow/boost_ep-prefix/src/boost_ep"\
           -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release 
RUN cmake --build /build/mlirdb -j$(nproc)

FROM mlirdb as reproduce
WORKDIR /repo
COPY --from=tpchgen /tpch-1 ./resources/data/tpch-1
