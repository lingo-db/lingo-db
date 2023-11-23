FROM ubuntu:jammy AS baseimg
RUN apt-get update
RUN apt-get -y install python3 python3-venv git cmake  g++ ninja-build wget unzip ccache curl lsb-release wget zlib1g-dev
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"
RUN cargo install --force cbindgen

FROM baseimg AS lingodb
COPY . /repo
WORKDIR /repo
RUN make build/lingodb-release/.buildstamp

