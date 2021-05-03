build/llvm-build:
	mkdir -p build/llvm-build
	cmake -G Ninja llvm-project/llvm  -B build/llvm-build \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=OFF \
       -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON

build/arrow:
	mkdir -p build/arrow
	cmake arrow/cpp  -B build/arrow -DARROW_GANDIVA=1

build-arrow: build/arrow
	cmake --build build/arrow
	cmake --install build/arrow --prefix build/arrow/install

build-llvm: build/llvm-build
	cmake --build build/llvm-build -j4

build/llvm-build-debug:
	mkdir -p build/llvm-build-debug
	cmake -G Ninja llvm-project/llvm  -B build/llvm-build-debug \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=OFF \
       -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DLLVM_ENABLE_ASSERTIONS=ON

build-llvm-debug: build/llvm-build-debug
	cmake --build build/llvm-build-debug -j1

test-coverage:
	cmake --build build/build-debug-llvm-release-coverage --target mlir-db-opt -- -j 6
	./build/llvm-build/bin/llvm-lit build/build-debug-llvm-release-coverage/test
	lcov --no-external --capture --directory build/build-debug-llvm-release-coverage -b . --output-file build/build-debug-llvm-release-coverage/coverage.info
		lcov --remove build/build-debug-llvm-release-coverage/coverage.info -o build/build-debug-llvm-release-coverage/filtered-coverage.info \
            '**/build/llvm-build/*' '**/llvm-project/*' '*.inc'
	genhtml  --ignore-errors source build/build-debug-llvm-release-coverage/filtered-coverage.info --legend --title "lcov-test" --output-directory=build/build-debug-llvm-release-coverage/coverage-report

run-test:
	cmake --build build/build-debug-llvm-release --target mlir-db-opt -- -j 6
	cmake --build build/build-debug-llvm-release --target db-run -- -j 6
	./build/llvm-build/bin/llvm-lit -v build/build-debug-llvm-release/test
coverage-clean:
	rm -rf build/build-debug-llvm-release-coverage/coverage

clean:
	rm -rf build