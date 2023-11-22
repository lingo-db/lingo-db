ROOT_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
NPROCS := $(shell echo $$(nproc))
LLVM_LIT := $(shell python3 -c "import pkg_resources; print(pkg_resources.get_distribution('lit').location+'/../../../bin/lit')")
LLVM_BIN_DIR := $(shell python3 -c "import lingodbllvm; print(lingodbllvm.get_bin_dir())")

build:
	mkdir -p $@
venv:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt

resources/data/tpch-1/.stamp: tools/generate/tpch.sh build/lingodb-debug/.buildstamp
	mkdir -p resources/data/tpch-1

	bash $< $(CURDIR)/build/lingodb-debug $(dir $(CURDIR)/$@) 1
	touch $@

resources/data/tpcds-1/.stamp: tools/generate/tpcds.sh build/lingodb-debug/.buildstamp
	mkdir -p resources/data/tpcds-1
	bash $< $(CURDIR)/build/lingodb-debug $(dir $(CURDIR)/$@) 1
	touch $@

resources/data/job/.stamp: tools/generate/job.sh build/lingodb-debug/.buildstamp
	mkdir -p resources/data/job
	bash $< $(CURDIR)/build/lingodb-debug $(dir $(CURDIR)/$@) 1
	touch $@

LDB_ARGS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		 -DPython3_EXECUTABLE="venv/bin/python3" \
	   	 -DCMAKE_BUILD_TYPE=Debug

dependencies: venv build
build/lingodb-debug/.stamp: dependencies
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS)
	touch $@

build/lingodb-debug/.buildstamp:
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@


build/lingodb-release/.buildstamp: build/lingodb-release/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@

build/lingodb-release/.stamp: dependencies
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_BUILD_TYPE=Release
	touch $@
build/lingodb-debug-coverage/.stamp: dependencies
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_CXX_FLAGS=--coverage -DCMAKE_C_FLAGS=--coverage
	touch $@

.PHONY: run-test
run-test: build/lingodb-debug/.stamp
	cmake --build $(dir $<) --target mlir-db-opt run-mlir run-sql sql-to-mlir sqlite-tester -- -j${NPROCS}
	${LLVM_LIT} -v build/lingodb-debug/test/lit -j 1
	find ./test/sqlite-small/ -maxdepth 1 -type f -name '*.test' | xargs -L 1 -P ${NPROCS} ./build/lingodb-debug/sqlite-tester

.PHONY: test-coverage
test-coverage: build/lingodb-debug-coverage/.stamp
	cmake --build $(dir $<) --target mlir-db-opt run-mlir run-sql sql-to-mlir -- -j${NPROCS}
	./build/llvm-build/bin/llvm-lit -v $(dir $<)/test/lit
	lcov --capture --directory $(dir $<)  --output-file $(dir $<)/coverage.info
	lcov --remove $(dir $<)/coverage.info -o $(dir $<)/filtered-coverage.info \
			'**/build/llvm-build/*' '**/llvm-project/*' '*.inc' '**/arrow/*' '**/pybind11/*' '**/vendored/*' '/usr/*' '**/build/lingodb-debug-coverage/*'
	genhtml  --ignore-errors source $(dir $<)/filtered-coverage.info --legend --title "lcov-test" --output-directory=$(dir $<)/coverage-report


.PHONY: run-benchmark
run-benchmark: build/lingodb-release/.stamp resources/data/tpch-1/.stamp
	cmake --build $(dir $<) --target run-sql -- -j${NPROCS}
	env QUERY_RUNS=5 env LINGODB_EXECUTION_MODE=SPEED python3 tools/scripts/benchmark-tpch.py $(dir $<) tpch-1

run-benchmarks: build/lingodb-release/.stamp resources/data/tpch-1/.stamp resources/data/tpcds-1/.stamp
	cmake --build $(dir $<) --target run-sql -- -j${NPROCS}
	env QUERY_RUNS=5 env LINGODB_EXECUTION_MODE=SPEED python3 tools/scripts/benchmark-tpch.py $(dir $<) tpch-1
	env QUERY_RUNS=5 env LINGODB_EXECUTION_MODE=SPEED python3 tools/scripts/benchmark-tpcds.py $(dir $<) tpcds-1

docker-buildimg:
	DOCKER_BUILDKIT=1 docker build -f "tools/docker/Dockerfile" -t lingodb-buildimg:$(shell echo "$$(git submodule status)" | cut -c 2-9 | tr '\n' '-') --target buildimg "."
build-docker:
	DOCKER_BUILDKIT=1 docker build -f "tools/docker/Dockerfile" -t lingo-db:latest --target lingodb  "."

build-release: build/lingodb-release/.buildstamp
build-debug: build/lingodb-debug/.buildstamp

.repr-docker-built:
	$(MAKE) build-repr-docker
	touch .repr-docker-built

.PHONY: clean
clean:
	rm -rf build

lint: build/lingodb-debug/.stamp
	sed -i 's/-fno-lifetime-dse//g' build/lingodb-debug/compile_commands.json
	venv/bin/python3 tools/scripts/run-clang-tidy.py -p $(dir $<) -quiet -header-filter="$(shell pwd)/include/.*" -exclude="arrow|vendored" -clang-tidy-binary=${LLVM_BIN_DIR}/clang-tidy
