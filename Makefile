build/llvm-build:
	mkdir -p build/llvm-build
	cmake -G Ninja llvm-project/llvm  -B build/llvm-build \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=OFF \
       -DLLVM_TARGETS_TO_BUILD="" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON

build-llvm: build/llvm-build
	cmake --build build/llvm-build -j4

build/llvm-build-debug:
	mkdir -p build/llvm-build-debug
	cmake -G Ninja llvm-project/llvm  -B build/llvm-build-debug \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=OFF \
       -DLLVM_TARGETS_TO_BUILD="" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DLLVM_ENABLE_ASSERTIONS=ON

build-llvm-debug: build/llvm-build-debug
	cmake --build build/llvm-build-debug -j1

clean:
	rm -rf build