#!/bin/bash

# Ensure the script is called with two arguments
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <binary_dir> <source_file> <target_file>"
  exit 1
fi

binary_dir=$1
source_file="$2"
target_file="$3"
${binary_dir}/mlir-db-opt --strip-debuginfo ${source_file} > ${target_file}



# Count lines starting with "#loc" continuously from the beginning
count=$(awk '!/^#loc/{exit} {count++} END {print count}' ${source_file})

# Create a real temporary file for the empty lines
temp_file=$(mktemp)

# Generate the empty lines and write them to the temporary file
for ((i=0; i<count; i++)); do echo ""; done > ${temp_file}

# Prepend the empty lines to the target file
cat ${target_file} >> ${temp_file} && mv ${temp_file} ${target_file}


