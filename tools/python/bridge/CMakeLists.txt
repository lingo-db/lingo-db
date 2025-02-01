cmake_minimum_required(VERSION 3.18)
project(lingodb_bridge LANGUAGES CXX)

# Find Python & nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Query PyArrow paths from Python
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pyarrow as pa; print(pa.get_include())"
    OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pyarrow as pa; print(' '.join(pa.get_libraries()))"
    OUTPUT_VARIABLE PYARROW_LIBRARIES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pyarrow as pa; print(' '.join(pa.get_library_dirs()))"
    OUTPUT_VARIABLE PYARROW_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.include_dir())"
    OUTPUT_VARIABLE NANOBIND_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pyarrow; print(pyarrow.create_library_symlinks())"
    OUTPUT_VARIABLE ARROW_INSTALLED
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("ARROW", ${ARROW_INSTALLED})


include_directories(${PYARROW_INCLUDE_DIR})
include_directories(/built-llvm/include)
link_directories(/built-llvm/lib)
link_directories(${PYARROW_LIBRARY_DIRS})


nanobind_add_module(ext src/extensions/pylingodb.cpp)
target_compile_definitions(ext PRIVATE MLIR_PYTHON_PACKAGE_PREFIX=lingodbbridge.mlir. )
target_link_libraries(ext PRIVATE libarrow_python.so libarrow.so)
install(TARGETS ext LIBRARY DESTINATION lingodbbridge)

nanobind_add_module(_mlir 
    src/extensions/mlir_vendored/MainModule.cpp
    src/extensions/mlir_vendored/IRAffine.cpp
    src/extensions/mlir_vendored/IRAttributes.cpp
    src/extensions/mlir_vendored/IRCore.cpp
    src/extensions/mlir_vendored/IRInterfaces.cpp
    src/extensions/mlir_vendored/IRModule.cpp
    src/extensions/mlir_vendored/IRTypes.cpp
    src/extensions/mlir_vendored/Pass.cpp
    src/extensions/mlir_vendored/Rewrite.cpp
    )

target_compile_definitions(_mlir PRIVATE MLIR_PYTHON_PACKAGE_PREFIX=lingodbbridge.mlir. )
target_link_libraries(_mlir PRIVATE LLVMSupport LLVMDemangle)
install(TARGETS _mlir LIBRARY DESTINATION lingodbbridge/mlir/_mlir_libs)

nanobind_add_module(mlir_init src/extensions/init_mlir_context.cpp)
target_compile_definitions(mlir_init PRIVATE MLIR_PYTHON_PACKAGE_PREFIX=lingodbbridge.mlir. )
target_link_libraries(mlir_init PRIVATE LLVMSupport LLVMDemangle)
install(TARGETS mlir_init LIBRARY DESTINATION lingodbbridge/mlir/_mlir_libs)

nanobind_add_module(mlir_lingodb src/extensions/custom_dialects_py)
target_compile_definitions(mlir_lingodb PRIVATE MLIR_PYTHON_PACKAGE_PREFIX=lingodbbridge.mlir. )
target_link_libraries(mlir_lingodb PRIVATE LLVMSupport LLVMDemangle)
install(TARGETS mlir_lingodb LIBRARY DESTINATION lingodbbridge/mlir/_mlir_libs)
