#!/usr/bin/env bash
# generates tpcxbb benchmark by using the container built by /tools/docker/tpcxbb.dockerfile

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <scale factor>"
    exit 1
fi

IMAGE_NAME="generate-tpcxbb:latest"
OUTPUT_DIR=$1

# Check if the image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Docker image '$IMAGE_NAME' not found."
    echo "Please build the image from the root of lingo-db with:"
    echo ""
    echo "    docker build --build-arg TPCXBB_ZIP=<path_to_TPCXBB_ZIP> -t generate-tpcxbb -f tools/docker/tpcxbb.dockerfile . "
    echo ""
    echo "Hint: the zip files need to be in the current path"
    exit 1
fi

# Generate the data
mkdir -p $OUTPUT_DIR  # Ensure the target directory exists
docker run --rm -v $OUTPUT_DIR:/app/output:Z generate-tpcxbb $2
