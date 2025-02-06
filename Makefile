ROOT_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
NPROCS := $(shell echo $$(nproc))
LLVM_LIT_BINARY := lit
#LLVM_BIN_DIR := $(shell venv/bin/python3 -c "import lingodbllvm; print(lingodbllvm.get_bin_dir())")
PYTHON_BINARY := python3
build:
	mkdir -p $@
venv:
	$(PYTHON_BINARY) -m venv venv
	venv/bin/pip install -r requirements.txt


resources/data/%/.rawdata:
	@mkdir -p $@
	@dir_name=$(shell dirname $@) && \
	base_name=$$(basename $$dir_name) && \
	script_name=$$(echo $$base_name | sed -E 's/-[0-9]+$$//') && \
	scale_factor=$$(echo $$base_name | grep -oE '[0-9]+$$' || echo 1) && \
	abs_path=$$(realpath $@) && \
	if [ -f tools/generate/$$script_name.sh ]; then \
		echo "Running bash tools/generate/$$script_name.sh with $$abs_path $$scale_factor"; \
		bash tools/generate/$$script_name.sh $$abs_path $$scale_factor; \
	else \
		echo "Error: Script tools/generate/$$script_name.sh not found!" >&2; \
		exit 1; \
	fi



resources/data/%/.stamp: resources/data/%/.rawdata build/lingodb-debug/.buildstamp
	@dir_name=$(shell dirname $@) && \
	base_name=$$(basename $$dir_name) && \
	dataset_name=$$(echo $$base_name | sed -E 's/-[0-9]+$$//') && \
	cd $(dir $@)/.rawdata && $(ROOT_DIR)/build/lingodb-debug/sql ../ < $(ROOT_DIR)/resources/sql/$$dataset_name/initialize.sql
	touch $@
	rm -rf resources/data/$*/.rawdata



LDB_ARGS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		 -DPython3_EXECUTABLE="${ROOT_DIR}/venv/bin/python3" \
	   	 -DCMAKE_BUILD_TYPE=Debug

build/dependencies: venv build
	touch $@

build/lingodb-debug/.stamp: build/dependencies
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS)
	touch $@

build/lingodb-debug/.buildstamp: build/lingodb-debug/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@


build/lingodb-release/.buildstamp: build/lingodb-release/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@

build/lingodb-release/.stamp: build/dependencies
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_BUILD_TYPE=Release
	touch $@

build/lingodb-relwithdebinfo/.buildstamp: build/lingodb-relwithdebinfo/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@

build/lingodb-relwithdebinfo/.stamp: build/dependencies
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_BUILD_TYPE=RelWithDebInfo
	touch $@
build/lingodb-debug-coverage/.stamp: build/dependencies
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_CXX_FLAGS="-O0 -fprofile-instr-generate -fcoverage-mapping" -DCMAKE_C_FLAGS="-O0 -fprofile-instr-generate -fcoverage-mapping" -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_C_COMPILER=clang-20
	touch $@

.PHONY: run-test
run-test: build/lingodb-debug/.stamp
	cmake --build $(dir $<) --target mlir-db-opt run-mlir run-sql sql-to-mlir sqlite-tester -- -j${NPROCS}
	$(MAKE) test-no-rebuild

test-no-rebuild: build/lingodb-debug/.buildstamp resources/data/test/.stamp resources/data/uni/.stamp
	${LLVM_LIT_BINARY} -v build/lingodb-debug/test/lit -j 1
	find ./test/sqlite-small/ -maxdepth 1 -type f -name '*.test' | xargs -L 1 -P ${NPROCS} ./build/lingodb-debug/sqlite-tester

sqlite-test-no-rebuild: build/lingodb-release/.buildstamp
	find ./test/sqlite/ -maxdepth 1 -type f -name '*.test' | xargs -L 1 -P ${NPROCS} ./build/lingodb-release/sqlite-tester

.PHONY: test-coverage
test-coverage: build/lingodb-debug-coverage/.stamp resources/data/test/.stamp resources/data/uni/.stamp
	cmake --build $(dir $<) --target mlir-db-opt run-mlir run-sql sql-to-mlir -- -j${NPROCS}
	${LLVM_LIT_BINARY} -v --per-test-coverage  $(dir $<)/test/lit
	find $(dir $<) -type f -name "*.profraw" > $(dir $<)/profraw-files
	llvm-profdata-20 merge -o $(dir $<)/coverage.profdata --input-files=$(dir $<)/profraw-files

coverage: build/lingodb-debug-coverage/.stamp
	$(MAKE) test-coverage
	mkdir -p build/coverage-report
	llvmcov2html --exclude-dir=$(dir $<),venv,vendored build/coverage-report $(dir $<)/run-mlir ./build/lingodb-debug-coverage/run-sql $(dir $<)/mlir-db-opt $(dir $<)/sql-to-mlir $(dir $<)/coverage.profdata


.PHONY: run-benchmark
run-benchmark: build/lingodb-release/.stamp resources/data/tpch-1/.stamp
	cmake --build $(dir $<) --target run-sql -- -j${NPROCS}
	env QUERY_RUNS=5 env LINGODB_EXECUTION_MODE=SPEED python3 tools/scripts/benchmark-tpch.py $(dir $<) tpch-1

run-benchmarks: build/lingodb-release/.stamp resources/data/tpch-1/.stamp resources/data/tpcds-1/.stamp
	cmake --build $(dir $<) --target run-sql -- -j${NPROCS}
	env QUERY_RUNS=5 env LINGODB_EXECUTION_MODE=SPEED python3 tools/scripts/benchmark-tpch.py $(dir $<) tpch-1
	env QUERY_RUNS=5 env LINGODB_EXECUTION_MODE=SPEED python3 tools/scripts/benchmark-tpcds.py $(dir $<) tpcds-1

build-docker-dev:
	DOCKER_BUILDKIT=1 docker build -f "tools/docker/Dockerfile" -t lingodb-dev --target baseimg "."

build-docker-py-dev:
	DOCKER_BUILDKIT=1 docker build -f "tools/python/bridge/Dockerfile" -t lingodb-py-dev --target devimg "."
build-py-bridge:
	DOCKER_BUILDKIT=1 docker build -f "tools/python/bridge/Dockerfile" -t lingodb-py-dev-build --target build "."
	docker run --rm  -v "${ROOT_DIR}:/built-packages" lingodb-py-dev-build create_package.sh cp$(PY_VERSION)-cp$(PY_VERSION)

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
	cmake --build build/lingodb-debug --target build_includes
	sed -i 's/-fno-lifetime-dse//g' build/lingodb-debug/compile_commands.json
	venv/bin/python3 tools/scripts/run-clang-tidy.py -p $(dir $<) -quiet -header-filter="$(shell pwd)/include/.*" -exclude="arrow|vendored" -clang-tidy-binary=clang-tidy-20
