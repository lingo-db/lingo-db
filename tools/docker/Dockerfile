FROM ubuntu:noble AS baseimg
RUN apt-get update && apt-get -y install software-properties-common wget
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main"
RUN apt-get update && apt-get -y install python3 python3-venv git g++ cmake ninja-build wget unzip ccache curl lsb-release wget zlib1g-dev lcov clang-19 llvm-19
ENV CC=clang-19 CXX=clang++-19
RUN git clone https://github.com/lingo-db/llvmcov2html.git /llvmcov2html && cd /llvmcov2html && git checkout 7362ad2 && make && cp bin/llvmcov2html /usr/bin/. && cd / && rm -rf /llvmcov2html

FROM baseimg AS lingodb
COPY . /repo
WORKDIR /repo
RUN make build/lingodb-release/.buildstamp

