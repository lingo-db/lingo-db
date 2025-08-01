#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <SQL_QUERY_FILE> <DB_DIR> <BIN_DIR> <NEW_WT_DIR> <NEW_BRANCH>"
  exit 1
fi

SQL_FILE="$1"
DB_DIR="$2"
BIN_DIR="$3"
WT_DIR="$4"
BRANCH="$5"

# Normalize db directory path to absolute path
if [[ "$DB_DIR" != /* ]]; then
  DB_DIR="$(cd "$DB_DIR" && pwd)"
fi

# 1) Check that SQL_FILE exists and is a regular file
if [ ! -f "$SQL_FILE" ]; then
  echo "error: SQL query file '$SQL_FILE' does not exist or is not a regular file."
  exit 1
fi

# 2) Check that DB_DIR exists and is a directory
if [ ! -d "$DB_DIR" ]; then
  echo "error: database directory '$DB_DIR' does not exist."
  exit 1
fi

# 3) Check that BIN_DIR exists and contains an executable 'compile-to-cpp'
if [ ! -d "$BIN_DIR" ]; then
  echo "error: binary directory '$BIN_DIR' does not exist."
  exit 1
fi
if [ ! -x "$BIN_DIR/compile-to-cpp" ]; then
  echo "error: '$BIN_DIR/compile-to-cpp' not found or not executable."
  exit 1
fi

# 4) Check that WT_DIR does NOT yet exist
if [ -e "$WT_DIR" ]; then
  echo "error: worktree directory '$WT_DIR' already exists."
  exit 1
fi

# 5) Check that BRANCH does NOT yet exist in this repo
if git rev-parse --verify --quiet "refs/heads/$BRANCH" >/dev/null; then
  echo "error: branch '$BRANCH' already exists."
  exit 1
fi

# All preconditions satisfied — proceed:

# Add a new worktree with no checkout of main branch, create new branch
git worktree add -b "$BRANCH" "$WT_DIR" --no-checkout

pushd "$WT_DIR" >/dev/null

# Set up sparse-checkout (paths are relative to repo root)
git sparse-checkout set --no-cone /include/ /src/runtime/ /src/execution/ResultProcessor.cpp /vendored /src/utility /src/catalog /src/scheduler /tools/standalone-query

# Now check out the new branch
git checkout "$BRANCH"

# Copy in the CMakeLists template
cp tools/standalone-query/CMakeLists.txt.template CMakeLists.txt
# Generate the query.cpp file using the provided SQL file
popd >/dev/null

# Run compile-to-cpp to turn the SQL query into C++ code
"$BIN_DIR/compile-to-cpp" "$SQL_FILE" "$DB_DIR" "$WT_DIR/query.cpp"

echo ""
echo ""
echo "== Standalone query worktree created =="
echo "Branch: $BRANCH"
echo "Worktree directory: $WT_DIR"
echo "You can now experiment with the query in this worktree using your favorite tools."
echo ""
echo ""
echo "== Commands to build the standalone query executable =="
echo "cd '$WT_DIR'"
echo "mkdir build && cd build"
echo "cmake  -DCMAKE_BUILD_TYPE=Release .."
echo 'cmake --build . -j $(nproc)'
echo ""
echo "== Commands to run the standalone query executable =="
echo "./main $DB_DIR"
echo ""
echo "== Commands to clean up the worktree =="
echo "cd [MAIN_REPO_DIR]"
echo "git worktree remove '$WT_DIR' --force"
echo "(optional) git branch -D '$BRANCH'"


