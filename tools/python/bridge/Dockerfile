FROM quay.io/pypa/manylinux_2_28_x86_64:2025.01.11-1 AS baseimg
RUN yum install -y epel-release || sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(cut -d: -f5 /etc/system-release-cpe | cut -d. -f1).noarch.rpm
RUN yum install -y cmake unzip wget
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip && unzip ninja-linux.zip -d /usr/bin/ && rm ninja-linux.zip
FROM baseimg AS buildllvm
RUN mkdir /built-llvm
RUN yum install -y clang
RUN /opt/python/cp312-cp312/bin/python3 -m venv /venv
RUN /venv/bin/pip install numpy pybind11 nanobind
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.0-rc1/llvm-project-20.1.0-rc1.src.tar.xz && tar -xf llvm-project-20.1.0-rc1.src.tar.xz && mkdir llvm-project-20.1.0-rc1.src/build
WORKDIR /llvm-project-20.1.0-rc1.src
RUN env VIRTUAL_ENV=/venv cmake -B build -DPython3_FIND_VIRTUALENV=ONLY -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang;clang-tools-extra" -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release -G Ninja -DLLVM_ENABLE_ASSERTIONS=OFF -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DLLVM_BUILD_TESTS=OFF -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=OFF -DLLVM_ENABLE_DUMP=ON -DLLVM_ENABLE_FFI=ON -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer" -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_PARALLEL_TABLEGEN_JOBS=10 -DBUILD_SHARED_LIBS=OFF -DLLVM_INSTALL_UTILS=ON  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_ZLIB=OFF -DCMAKE_INSTALL_PREFIX=/built-llvm llvm/
RUN  cmake --build build --target install -j$(nproc)

#FROM baseimg AS devimg
#COPY --from=buildllvm /built-llvm /built-llvm
#RUN yum install -y https://apache.jfrog.io/artifactory/arrow/centos/$(cut -d: -f5 /etc/system-release-cpe | cut -d. -f1)/apache-arrow-release-latest.rpm
#RUN yum install -y --enablerepo=epel arrow-devel-19* # For C++
#RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.0-rc1/llvm-project-20.1.0-rc1.src.tar.xz && tar -xf llvm-project-20.1.0-rc1.src.tar.xz && mv /llvm-project-20.1.0-rc1.src /llvm-src
# install boost context library
#RUN wget https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.gz && tar -xf boost_1_83_0.tar.gz && cd boost_1_83_0 && ./bootstrap.sh --prefix=/usr && ./b2 install --with-context && cd .. && rm -rf boost_1_83_0

FROM baseimg AS devimg
COPY --from=buildllvm /built-llvm /built-llvm
RUN wget https://dlcdn.apache.org/arrow/arrow-19.0.1/apache-arrow-19.0.1.tar.gz && tar -xf apache-arrow-19.0.1.tar.gz && rm apache-arrow-19.0.1.tar.gz
RUN mkdir /built-arrow
RUN cd /apache-arrow-19.0.1/cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/built-arrow -DARROW_DEPENDENCY_SOURCE=BUNDLED -DARROW_BUILD_STATIC=ON -DARROW_CSV=ON -DARROW_COMPUTE=ON && cmake --build build --target install -j$(nproc) && cd .. && rm  -r /apache-arrow-19.0.1
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.0-rc1/llvm-project-20.1.0-rc1.src.tar.xz && tar -xf llvm-project-20.1.0-rc1.src.tar.xz && mv /llvm-project-20.1.0-rc1.src /llvm-src
RUN wget https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.gz && tar -xf boost_1_83_0.tar.gz && cd boost_1_83_0 && ./bootstrap.sh --prefix=/usr && ./b2 install --with-context && cd .. && rm -rf boost_1_83_0

ENV PLAT=manylinux_2_28_x86_64
FROM devimg AS build
WORKDIR /repo
COPY . /repo
RUN cp /repo/tools/python/bridge/create_package.sh /usr/bin/
RUN chmod +x /usr/bin/create_package.sh
