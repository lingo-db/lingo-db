ROOT_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
NPROCS := $(shell echo $$(nproc))

build:
	mkdir -p $@

build/llvm-build/.stamp:
	mkdir -p $(dir $@)
	cmake -G Ninja llvm-project/llvm  -B $(dir $@) \
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
	touch $@

build/arrow/.stamp:
	mkdir -p $(dir $@)
	cmake arrow/cpp  -B $(dir $@) -DARROW_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib
	touch $@

build/arrow-perf/.stamp:
	mkdir -p $(dir $@)
	cmake arrow/cpp  -B $(dir $@) -DARROW_PYTHON=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-ffixed-r15
	touch $@

build/arrow/.buildstamp: build/arrow/.stamp
	cmake --build build/arrow -j${NPROCS}
	cmake --install build/arrow --prefix build/arrow/install
	touch $@

build/arrow-perf/.buildstamp: build/arrow-perf/.stamp
	cmake --build build/arrow-perf -j${NPROCS}
	cmake --install build/arrow-perf --prefix build/arrow-perf/install
	touch $@

build/arrow/.pyarrowstamp: build/arrow/.buildstamp
	cd arrow/python; python3 setup.py build_ext --inplace --extra-cmake-args="-DArrow_DIR=${ROOT_DIR}/build/arrow/install/lib/cmake/arrow -DArrowPython_DIR=${ROOT_DIR}/build/arrow/install/lib/cmake/arrow"
	touch $@

build/llvm-build/.buildstamp: build/llvm-build/.stamp
	cmake --build build/llvm-build -j${NPROCS}
	touch $@

resources/data/tpch-1/.stamp: tools/generate/generate.sh
	bash $<
	touch $@

LDB_ARGS=-DMLIR_DIR=${ROOT_DIR}build/llvm-build/lib/cmake/mlir \
		 -DArrow_DIR=${ROOT_DIR}build/arrow/install/lib/cmake/arrow \
		 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
LDB_DEPS=build/llvm-build/.buildstamp build/arrow/.buildstamp
dependencies: $(LDB_DEPS)

build/lingodb-debug/.stamp: $(LDB_DEPS)
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS)
	touch $@
build/lingodb-release/.stamp: $(LDB_DEPS)
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_BUILD_TYPE=Release
	touch $@
build/lingodb-debug-coverage/.stamp: $(LDB_DEPS)
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_CXX_FLAGS=--coverage -DCMAKE_C_FLAGS=--coverage
	touch $@

.PHONY: test-coverage
test-coverage: build/lingodb-debug-coverage/.stamp
	cmake --build $(dir $<) --target mlir-db-opt run-mlir run-sql pymlirdbext sql-to-mlir -- -j${NPROCS}
	env LD_LIBRARY_PATH=${ROOT_DIR}/build/arrow/install/lib ./build/llvm-build/bin/llvm-lit $(dir $<)/test
	lcov --capture --directory $(dir $<)  --output-file $(dir $<)/coverage.info
	lcov --remove $(dir $<)/coverage.info -o $(dir $<)/filtered-coverage.info \
			'**/build/llvm-build/*' '**/llvm-project/*' '*.inc' '**/arrow/*' '**/pybind11/*' '**/vendored/*' '/usr/*'
	genhtml  --ignore-errors source $(dir $<)/filtered-coverage.info --legend --title "lcov-test" --output-directory=$(dir $<)/coverage-report
.PHONY: run-test
run-test: build/lingodb-debug/.stamp
	cmake --build $(dir $<) --target mlir-db-opt run-mlir run-sql pymlirdbext sql-to-mlir -- -j${NPROCS}
	env LD_LIBRARY_PATH=${ROOT_DIR}/build/arrow/install/lib ./build/llvm-build/bin/llvm-lit -v $(dir $<)/test
.PHONY: run-benchmark
run-benchmark: build/lingodb-release/.stamp resources/data/tpch-1/.stamp
	cmake --build $(dir $<) --target run-sql -- -j${NPROCS}
	env QUERY_RUNS=5 python3 tools/scripts/benchmark-tpch.py $(dir $<) tpch-1

docker-buildimg:
	DOCKER_BUILDKIT=1 docker build -f "docker/Dockerfile" -t mlirdb-buildimg:$(shell echo "$$(git submodule status)" | cut -c 2-9 | tr '\n' '-') --target buildimg "."
build-docker:
	DOCKER_BUILDKIT=1 docker build -f "docker/Dockerfile" -t mlirdb:latest --target mlirdb  "."
build-repr-docker:
	DOCKER_BUILDKIT=1 docker build -f "docker/Dockerfile" -t mlirdb-repr:latest --target reproduce "."

.repr-docker-built:
	$(MAKE) build-repr-docker
	touch .repr-docker-built

.PHONY: clean
clean:
	rm -rf build

reproduce: .repr-docker-built
	 docker run --privileged -it mlirdb-repr /bin/bash -c "python3 tools/benchmark-tpch.py /build/mlirdb/ tpch-1"

lint: build/lingodb-debug/.stamp
	python3 tools/scripts/run-clang-tidy.py -p $(dir $<) -quiet -header-filter="$(shell pwd)/include/.*" -exclude="arrow|vendored" -clang-tidy-binary=./build/llvm-build/bin/clang-tidy
