# docker image with hadoop for generating tpcxbb data
# run the build command from the root dir of lingo-db
# note that the path to the zip file has to be in the build context


FROM apache/hadoop:3.4.1

# note that the path to the zip file has to be in the build context
ARG TPCXBB_ZIP

WORKDIR /app

# become root inside the container
USER root

# https://blog.centos.org/2020/12/future-is-centos-stream/
# https://serverfault.com/questions/1161816/mirrorlist-centos-org-no-longer-resolve
RUN sed -i 's/mirror\.centos\.org/vault.centos.org/g' /etc/yum.repos.d/CentOS-*.repo && \
    sed -i 's/^#.*baseurl=http/baseurl=http/g' /etc/yum.repos.d/CentOS-*.repo && \
    sed -i 's/^mirrorlist=http/#mirrorlist=http/g' /etc/yum.repos.d/CentOS-*.repo


RUN yum update -y && \
    yum install -y bsdtar && \
    yum clean all


COPY tools/generate/generate-tpcxbb.sh .
COPY $TPCXBB_ZIP /app/TPCXBB_INPUT.zip

ENTRYPOINT ["./generate-tpcxbb.sh"]