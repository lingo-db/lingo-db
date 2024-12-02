FROM ubuntu:jammy AS baseimg
RUN apt-get update
RUN apt-get -y install python3 python3-venv git cmake  g++ ninja-build wget unzip ccache curl lsb-release wget zlib1g-dev lcov


FROM baseimg AS lingodb
COPY . /repo
WORKDIR /repo
RUN make build/lingodb-release/.buildstamp

