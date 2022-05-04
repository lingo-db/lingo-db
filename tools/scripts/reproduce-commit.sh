#!/bin/bash
echo $1
DOCKER_BUILDKIT=1 docker build -f "docker/Dockerfile" -t mlirdb-repr:latest --build-arg builtImage=ghcr.io/jungmair/mlirdb:$1 --target reproduce "."
docker run --privileged -it mlirdb-repr /bin/bash -c "python3 tools/scripts/benchmark-tpch.py /build/mlirdb/ tpch-1"
