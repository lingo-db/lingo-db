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
	   -DMLIR_ENABLE_BINDINGS_PYTHON=ON
	touch $@

build/llvm-build/.buildstamp: build/llvm-build/.stamp
	cmake --build build/llvm-build -j16
	touch $@

resources/data/tpch-1/.stamp: tools/generate/tpch.sh
	mkdir -p resources/data/tpch-1

	bash $< $(CURDIR)/build/lingodb-debug $(dir $(CURDIR)/$@) 1
	touch $@

resources/data/tpcds-1/.stamp: tools/generate/tpcds.sh
	mkdir -p resources/data/tpcds-1
	bash $< $(CURDIR)/build/lingodb-debug $(dir $(CURDIR)/$@) 1
	touch $@

resources/data/job/.stamp: tools/generate/job.sh
	mkdir -p resources/data/job
	bash $< $(CURDIR)/build/lingodb-debug $(dir $(CURDIR)/$@) 1
	touch $@

LDB_ARGS=-DMLIR_DIR=${ROOT_DIR}build/llvm-build/lib/cmake/mlir \
		 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	   	 -DCMAKE_BUILD_TYPE=Debug
LDB_DEPS=build/llvm-build/.buildstamp
dependencies: $(LDB_DEPS)

build_llvm: build/llvm-build/.buildstamp
build/lingodb-debug/.stamp: $(LDB_DEPS)
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS)
	touch $@

build/lingodb-debug/.buildstamp: build/lingodb-debug/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@


build/lingodb-release/.buildstamp: build/lingodb-release/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@

build/lingodb-release/.stamp: $(LDB_DEPS)
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_BUILD_TYPE=Release
	touch $@
build/lingodb-debug-coverage/.stamp: $(LDB_DEPS)
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_CXX_FLAGS=--coverage -DCMAKE_C_FLAGS=--coverage
	touch $@

.PHONY: run-test
run-test: build/lingodb-debug/.stamp
	cmake --build $(dir $<) --target mlir-db-opt run-mlir run-sql mlir-doc -- -j${NPROCS}
	env LD_LIBRARY_PATH=${ROOT_DIR}/build/arrow/install/lib ./build/llvm-build/bin/llvm-lit -v $(dir $<)/test/lit -j 1
.PHONY: run-benchmark
run-benchmark: build/lingodb-release/.stamp build/lingodb-debug/.stamp resources/data/tpch-1/.stamp
	cmake --build $(dir $<) --target run-sql -- -j${NPROCS}
	env QUERY_RUNS=5 env LINGODB_EXECUTION_MODE=SPEED python3 tools/scripts/benchmark-tpch.py $(dir $<) tpch-1

run-benchmarks: build/lingodb-release/.stamp build/lingodb-debug/.stamp resources/data/tpch-1/.stamp resources/data/tpcds-1/.stamp
	cmake --build $(dir $<) --target run-sql -- -j${NPROCS}
	env QUERY_RUNS=5 env LINGODB_EXECUTION_MODE=SPEED python3 tools/scripts/benchmark-tpch.py $(dir $<) tpch-1
	env QUERY_RUNS=5 env LINGODB_EXECUTION_MODE=SPEED python3 tools/scripts/benchmark-tpcds.py $(dir $<) tpcds-1

docker-buildimg:
	DOCKER_BUILDKIT=1 docker build -f "tools/docker/Dockerfile" -t lingodb-buildimg:$(shell echo "$$(git submodule status)" | cut -c 2-9 | tr '\n' '-') --target buildimg "."
build-docker:
	DOCKER_BUILDKIT=1 docker build -f "tools/docker/Dockerfile" -t lingo-db:latest --target lingodb  "."
build-repr-docker:
	DOCKER_BUILDKIT=1 docker build -f "tools/docker/Dockerfile" -t lingodb-repr:latest --target reproduce "."

build-release: build/lingodb-release/.buildstamp
build-debug: build/lingodb-debug/.buildstamp

.repr-docker-built:
	$(MAKE) build-repr-docker
	touch .repr-docker-built

.PHONY: clean
clean:
	rm -rf build

reproduce: .repr-docker-built
	 docker run --privileged -it lingodb-repr /bin/bash -c "python3 tools/scripts/benchmark-tpch.py /build/lingodb/ tpch-1"

lint: build/lingodb-debug/.stamp
	sed -i 's/-fno-lifetime-dse//g' build/lingodb-debug/compile_commands.json
	python3 tools/scripts/run-clang-tidy.py -p $(dir $<) -quiet -header-filter="$(shell pwd)/include/.*" -exclude="arrow|vendored" -clang-tidy-binary=./build/llvm-build/bin/clang-tidy
