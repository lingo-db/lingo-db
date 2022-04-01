ROOT_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
NPROCS := $(shell echo $$(nproc))
SUBID := $(shell echo "$$(git submodule status)" | cut -c 2-9 | tr '\n' '-')

build/llvm-build:
	mkdir -p build/llvm-build
	cmake -G Ninja llvm-project/llvm  -B build/llvm-build \
	   -DLLVM_ENABLE_PROJECTS="mlir;clang;clang-tools-extra"\
	   -DLLVM_USE_PERF=ON \
	   -DLLVM_BUILD_EXAMPLES=OFF \
	   -DLLVM_TARGETS_TO_BUILD="X86" \
	   -DCMAKE_BUILD_TYPE=Release \
	   -DLLVM_ENABLE_ASSERTIONS=ON \
	   -DPython3_FIND_VIRTUALENV=ONLY \
       -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
       -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="${ROOT_DIR}/torch-mlir" \
       -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR="${ROOT_DIR}/torch-mlir/external/llvm-external-projects/torch-mlir-dialects" \
       -DMLIR_ENABLE_BINDINGS_PYTHON=ON

build/arrow:
	mkdir -p build/arrow
	cmake arrow/cpp  -B build/arrow -DARROW_PYTHON=ON -DCMAKE_BUILD_TYPE=Release

build/arrow-perf:
	mkdir -p build/arrow-perf
	cmake arrow/cpp  -B build/arrow-perf -DARROW_PYTHON=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-ffixed-r15

build-arrow: build/arrow
	cmake --build build/arrow -j${NPROCS}
	cmake --install build/arrow --prefix build/arrow/install

build-arrow-perf: build/arrow-perf
	cmake --build build/arrow-perf -j${NPROCS}
	cmake --install build/arrow-perf --prefix build/arrow-perf/install

build/pyarrow:
	cd arrow/python; python3 setup.py build_ext --inplace --extra-cmake-args="-DArrow_DIR=${ROOT_DIR}/build/arrow/install/lib/cmake/arrow -DArrowPython_DIR=${ROOT_DIR}/build/arrow/install/lib/cmake/arrow"
build-llvm: build/llvm-build
	cmake --build build/llvm-build -j${NPROCS}

build/llvm-build-debug:
	mkdir -p build/llvm-build-debug
	cmake -G Ninja llvm-project/llvm  -B build/llvm-build-debug \
	   -DLLVM_ENABLE_PROJECTS=mlir \
	   -DLLVM_BUILD_EXAMPLES=OFF \
	   -DLLVM_USE_PERF=ON \
	   -DLLVM_TARGETS_TO_BUILD="X86" \
	   -DCMAKE_BUILD_TYPE=Debug \
	   -DLLVM_ENABLE_ASSERTIONS=ON

build-llvm-debug: build/llvm-build-debug
	cmake --build build/llvm-build-debug -j${NPROCS}

build/build-debug-llvm-release:
	cmake -G Ninja .  -B  build/build-debug-llvm-release \
		-DMLIR_DIR=${ROOT_DIR}build/llvm-build/lib/cmake/mlir \
		-DArrow_DIR=${ROOT_DIR}build/arrow/install/lib/cmake/arrow \
		-DBoost_INCLUDE_DIR=${ROOT_DIR}build/arrow/boost_ep-prefix/src/boost_ep \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON

build/build-llvm-release:
	cmake -G Ninja .  -B  build/build-llvm-release \
		-DMLIR_DIR=${ROOT_DIR}build/llvm-build/lib/cmake/mlir \
		-DArrow_DIR=${ROOT_DIR}build/arrow/install/lib/cmake/arrow \
		-DBoost_INCLUDE_DIR=${ROOT_DIR}build/arrow/boost_ep-prefix/src/boost_ep \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release

build/build-debug-llvm-release-coverage:
	cmake -G Ninja .  -B  build/build-debug-llvm-release-coverage \
		-DMLIR_DIR=${ROOT_DIR}build/llvm-build/lib/cmake/mlir \
		-DArrow_DIR=${ROOT_DIR}build/arrow/install/lib/cmake/arrow \
		-DBoost_INCLUDE_DIR=${ROOT_DIR}build/arrow/boost_ep-prefix/src/boost_ep \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_FLAGS=--coverage -DCMAKE_C_FLAGS=--coverage

dependencies: build build-llvm build-arrow build/pyarrow

test-coverage:  build/build-debug-llvm-release-coverage
	cmake --build build/build-debug-llvm-release-coverage --target mlir-db-opt db-run-query db-run pymlirdbext -- -j${NPROCS}
	export LD_LIBRARY_PATH=${ROOT_DIR}/build/arrow/install/lib &&./build/llvm-build/bin/llvm-lit build/build-debug-llvm-release-coverage/test
	lcov --no-external --capture --directory build/build-debug-llvm-release-coverage -b . --output-file build/build-debug-llvm-release-coverage/coverage.info
		lcov --remove build/build-debug-llvm-release-coverage/coverage.info -o build/build-debug-llvm-release-coverage/filtered-coverage.info \
			'**/build/llvm-build/*' '**/llvm-project/*' '*.inc' '**/arrow/*' '**/pybind11/*'
	genhtml  --ignore-errors source build/build-debug-llvm-release-coverage/filtered-coverage.info --legend --title "lcov-test" --output-directory=build/build-debug-llvm-release-coverage/coverage-report
run-test: build/build-debug-llvm-release
	cmake --build build/build-debug-llvm-release --target mlir-db-opt db-run-query db-run pymlirdbext sql-to-mlir -- -j${NPROCS}
	export LD_LIBRARY_PATH=${ROOT_DIR}/build/arrow/install/lib && ./build/llvm-build/bin/llvm-lit -v build/build-debug-llvm-release/test
run-benchmark: build/build-llvm-release
	cmake --build build/build-llvm-release --target db-run-query -- -j${NPROCS}
	python3 tools/benchmark-tpch.py ./build/build-llvm-release tpch-1

coverage-clean:
	rm -rf build/build-debug-llvm-release-coverage/coverage

docker-buildimg:
	DOCKER_BUILDKIT=1 docker build -f "docker/Dockerfile" -t mlirdb-buildimg:${SUBID} --target buildimg "."
build-docker:
	DOCKER_BUILDKIT=1 docker build -f "docker/Dockerfile" -t mlirdb:latest --target mlirdb  "."
build-repr-docker:
	DOCKER_BUILDKIT=1 docker build -f "docker/Dockerfile" -t mlirdb-repr:latest --target reproduce "."

.repr-docker-built:
	$(MAKE) build-repr-docker
	touch .repr-docker-built
clean:
	rm -rf build


reproduce: .repr-docker-built
	 docker run --privileged -it mlirdb-repr /bin/bash -c "python3 tools/benchmark-tpch.py /build/mlirdb/ tpch-1"

lint:
	python3 tools/scripts/run-clang-tidy.py -p build/build-debug-llvm-release -quiet -header-filter="$(shell pwd)/include/.*" -exclude="arrow|vendored" -clang-tidy-binary=./build/llvm-build/bin/clang-tidy
#perf:
#	 perf record -k 1 -F 1000  --call-graph dwarf [cmd]
#	 perf inject -j -i perf.data -o perf.data.jitted
#	perf report --no-children -i perf.data.jitted

test-subid:
	echo ${SUBID}