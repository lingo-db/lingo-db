#!/bin/bash

# Check if a directory was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <folder>"
  exit 1
fi

# Get the absolute path to the folder
FOLDER="$1"


# Find all .md files and run the Python script on each
find "$FOLDER" -type f -name "*.md" | while read -r file; do
  echo "Processing $file"
  python3 tools/fix-doc/fix-md-doc.py "$file" "$file"
done
