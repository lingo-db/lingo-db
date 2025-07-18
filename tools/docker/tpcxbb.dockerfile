# docker build -t tpcxbb -f /tools/docker/tpcxbb.dockerfile .
# docker run --rm -v <tpcxbb-zip>:/app/input.zip:Z -v <./datasets/tpcxbb-1>:/app/output:Z tpcxbb /app/output <scale-factor> /app/input.zip
# Note: the output directory (e.g. ./datasets/tpcxbb-1) has exit for the command to run successfully

FROM apache/hadoop:3.4.1
WORKDIR /app

USER root

# https://blog.centos.org/2020/12/future-is-centos-stream/
# https://serverfault.com/questions/1161816/mirrorlist-centos-org-no-longer-resolve
RUN sed -i 's/mirror\.centos\.org/vault.centos.org/g' /etc/yum.repos.d/CentOS-*.repo && \
    sed -i 's/^#.*baseurl=http/baseurl=http/g' /etc/yum.repos.d/CentOS-*.repo && \
    sed -i 's/^mirrorlist=http/#mirrorlist=http/g' /etc/yum.repos.d/CentOS-*.repo


RUN yum update -y && \
    yum install -y bsdtar && \
    yum clean all

COPY tools/generate/tpcxbb.sh .

ENTRYPOINT ["./tpcxbb.sh"]