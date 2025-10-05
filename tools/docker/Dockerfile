FROM ubuntu:noble AS devimg
RUN apt-get update && apt-get -y install software-properties-common wget
RUN add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main"
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb && apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename  --short).deb && rm *.deb
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main"
RUN apt-get update && apt-get -y install python3 python3-venv python3-pip git g++ cmake ninja-build wget unzip ccache curl lsb-release wget zlib1g-dev lcov clang-20 llvm-20 libclang-20-dev llvm-20-dev libmlir-20-dev mlir-20-tools clang-tidy-20 libarrow-dev=21.* libarrow-compute-dev=21.*  libboost-context1.83-dev catch2 flex bison
RUN pip3 install --break-system-packages lit
ENV CC=clang-20 CXX=clang++-20
RUN git clone https://github.com/lingo-db/llvmcov2html.git /llvmcov2html && cd /llvmcov2html && git checkout 34603a1 && make && cp bin/llvmcov2html /usr/bin/. && cd / && rm -rf /llvmcov2html
