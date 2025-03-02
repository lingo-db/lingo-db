ROOT_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
NPROCS := $(shell echo $$(nproc))
LLVM_LIT_BINARY := lit

DATA_BUILD_TYPE ?= debug
TEST_BUILD_TYPE ?= debug
SQLITE_TEST_BUILD_TYPE ?= release


build:
	mkdir -p $@


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



resources/data/%/.stamp: resources/data/%/.rawdata build/lingodb-$(DATA_BUILD_TYPE)/.buildstamp
	rm -f resources/data/$*/*.arrow
	rm -f resources/data/$*/*.arrow.sample
	rm -f resources/data/$*/*.json
	@dir_name=$(shell dirname $@) && \
	base_name=$$(basename $$dir_name) && \
	dataset_name=$$(echo $$base_name | sed -E 's/-[0-9]+$$//') && \
	cd $(dir $@)/.rawdata && $(ROOT_DIR)/build/lingodb-$(DATA_BUILD_TYPE)/sql ../ < $(ROOT_DIR)/resources/sql/$$dataset_name/initialize.sql
	touch $@
	rm -rf resources/data/$*/.rawdata



LDB_ARGS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  \
	   	 -DCMAKE_BUILD_TYPE=Debug

build/lingodb-debug/.stamp: build
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS)
	touch $@

build/lingodb-debug/.buildstamp: build/lingodb-debug/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@


build/lingodb-release/.buildstamp: build/lingodb-release/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@

build/lingodb-release/.stamp: build
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_BUILD_TYPE=Release
	touch $@

build/lingodb-asan/.buildstamp: build/lingodb-asan/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@

build/lingodb-asan/.stamp: build
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_BUILD_TYPE=ASAN
	touch $@

build/lingodb-relwithdebinfo/.buildstamp: build/lingodb-relwithdebinfo/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@

build/lingodb-relwithdebinfo/.stamp: build
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_BUILD_TYPE=RelWithDebInfo
	touch $@
build/lingodb-debug-coverage/.stamp: build
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_CXX_FLAGS="-O0 -fprofile-instr-generate -fcoverage-mapping" -DCMAKE_C_FLAGS="-O0 -fprofile-instr-generate -fcoverage-mapping" -DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_C_COMPILER=clang-20
	touch $@

.PHONY: run-test
run-test: build/lingodb-$(TEST_BUILD_TYPE)/.stamp
	cmake --build $(dir $<) --target mlir-db-opt run-mlir run-sql sql-to-mlir sqlite-tester -- -j${NPROCS}
	$(MAKE) test-no-rebuild

test-no-rebuild: build/lingodb-$(TEST_BUILD_TYPE)/.buildstamp resources/data/test/.stamp resources/data/uni/.stamp
	${LLVM_LIT_BINARY} -v build/lingodb-$(TEST_BUILD_TYPE)/test/lit -j 1
	find ./test/sqlite-small/ -maxdepth 1 -type f -name '*.test' | xargs -L 1 -P ${NPROCS} ./build/lingodb-$(TEST_BUILD_TYPE)/sqlite-tester

sqlite-test-no-rebuild: build/lingodb-$(SQLITE_TEST_BUILD_TYPE)/.buildstamp
	find ./test/sqlite/ -maxdepth 1 -type f -name '*.test' | xargs -L 1 -P ${NPROCS} ./build/lingodb-$(SQLITE_TEST_BUILD_TYPE)/sqlite-tester

.PHONY: test-coverage
test-coverage: build/lingodb-debug-coverage/.stamp resources/data/test/.stamp resources/data/uni/.stamp
	cmake --build $(dir $<) --target mlir-db-opt run-mlir run-sql sql-to-mlir -- -j${NPROCS}
	${LLVM_LIT_BINARY} -v --per-test-coverage  $(dir $<)/test/lit
	find $(dir $<) -type f -name "*.profraw" > $(dir $<)/profraw-files
	llvm-profdata-20 merge -o $(dir $<)/coverage.profdata --input-files=$(dir $<)/profraw-files

coverage: build/lingodb-debug-coverage/.stamp
	$(MAKE) test-coverage
	mkdir -p build/coverage-report
	llvmcov2html --exclude-dir=$(dir $<),vendored build/coverage-report $(dir $<)/run-mlir ./build/lingodb-debug-coverage/run-sql $(dir $<)/mlir-db-opt $(dir $<)/sql-to-mlir $(dir $<)/coverage.profdata


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


build-release: build/lingodb-release/.buildstamp
build-debug: build/lingodb-debug/.buildstamp
build-asan: build/lingodb-asan/.buildstamp

.PHONY: clean
clean:
	rm -rf build

lint: build/lingodb-debug/.stamp
	cmake --build build/lingodb-debug --target build_includes
	sed -i 's/-fno-lifetime-dse//g' build/lingodb-debug/compile_commands.json
	python3 tools/scripts/run-clang-tidy.py -p $(dir $<) -quiet -header-filter="$(shell pwd)/include/.*" -exclude="arrow|vendored" -clang-tidy-binary=clang-tidy-20
