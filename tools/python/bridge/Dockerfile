FROM quay.io/pypa/manylinux_2_28_x86_64 AS devimg
RUN yum install -y epel-release || sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(cut -d: -f5 /etc/system-release-cpe | cut -d. -f1).noarch.rpm
RUN yum install -y https://apache.jfrog.io/artifactory/arrow/centos/$(cut -d: -f5 /etc/system-release-cpe | cut -d. -f1)/apache-arrow-release-latest.rpm
RUN yum install -y --enablerepo=epel arrow-devel # For C++
RUN yum install -y cmake ninja-build
ENV PLAT=manylinux_2_28_x86_64
FROM devimg AS build
WORKDIR /repo
COPY . /repo
RUN cp /repo/tools/python/bridge/create_package.sh /usr/bin/
RUN chmod +x /usr/bin/create_package.sh