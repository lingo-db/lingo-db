cmake_minimum_required(VERSION 3.13.4)
if (POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif ()
project(lingodb LANGUAGES CXX C)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DTRACER")
include(FindZLIB)
message("Using ZLIB: ${ZLIB_INCLUDE_DIRS}")
message("Using Python3: ${Python3_EXECUTABLE}")
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import lingodbllvm; print(lingodbllvm.get_mlir_dir())"
        OUTPUT_VARIABLE MLIR_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pyarrow as pa; print(pa.get_include())"
        OUTPUT_VARIABLE ARROW_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pyarrow as pa; print(pa.get_library_dirs()[0])"
        OUTPUT_VARIABLE ARROW_LIB_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pkg_resources; print(pkg_resources.get_distribution('tbb').location+'/../../')"
        OUTPUT_VARIABLE TBB_LIB_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pkg_resources; print(pkg_resources.get_distribution('tbb-devel').location+'/../../../include')"
        OUTPUT_VARIABLE TBB_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(LLVM_BINARY_DIR "${MLIR_DIR}/../../../")
message("arrow: ${ARROW_INCLUDE_DIR} ${ARROW_LIB_PATH}")
message("tbb: ${TBB_LIB_PATH} ${TBB_INCLUDE_DIR} ")
include_directories(${ARROW_INCLUDE_DIR})
include_directories(${TBB_INCLUDE_DIR})

link_directories(${ARROW_LIB_PATH})
link_directories(${TBB_LIB_PATH})
option(COMPILE_FOR_PERF "compile for perf" OFF)
option(ENABLE_CRANELIFT_BACKEND "enable cranelift backend" OFF)
option(ENABLE_GPU_BACKEND "enable GPU backend" OFF)

if (COMPILE_FOR_PERF)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffixed-r15")
endif (COMPILE_FOR_PERF)
set (Clang_DIR "${MLIR_DIR}/../clang")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

find_package(Clang REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddClang)
include(HandleLLVMOptions)
set("CLANG_VERSION" ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/vendored)
add_subdirectory(${PROJECT_SOURCE_DIR}/vendored/libpg_query/ EXCLUDE_FROM_ALL)

include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
set(LLVM_LINK_COMPONENTS
        Support
        Core
        nativecodegen
        OrcJIT
        X86
        )


add_custom_target(build_includes)
set(runtime_funcs_ptr_libs "")
add_subdirectory(include)
add_subdirectory(tools/build-tools)
gen_rt_def(dump-rt-defs "DumpRuntime.h")
gen_rt_def(string-rt-defs "StringRuntime.h")
gen_rt_def(date-rt-defs "DateRuntime.h")
gen_rt_def(decimal-rt-defs "DecimalRuntime.h")
gen_rt_def(integer-rt-defs "IntegerRuntime.h")
gen_rt_def(float-rt-defs "FloatRuntime.h")
gen_rt_def(timing-rt-defs "Timing.h")
gen_rt_def(tb-arrow-column-defs "ArrowColumn.h")
gen_rt_def(tb-arrow-table-defs "ArrowTable.h")
gen_rt_def(ds-it-rt-defs "DataSourceIteration.h")
gen_rt_def(join-ht-rt-defs "LazyJoinHashtable.h")
gen_rt_def(ht-rt-defs "Hashtable.h")
gen_rt_def(ec-rt-defs "ExecutionContext.h")
gen_rt_def(db-rt-defs "RelationHelper.h")
gen_rt_def(grow-buf-rt-defs "GrowingBuffer.h")
gen_rt_def(buf-rt-defs "Buffer.h")
gen_rt_def(stv-rt-defs "SegmentTreeView.h")
gen_rt_def(heap-rt-defs "Heap.h")
gen_rt_def(idx-rt-defs "HashIndex.h")
gen_rt_def(hmm-rt-defs "HashMultiMap.h")
gen_rt_def(tls-rt-defs "ThreadLocal.h")
gen_rt_def(st-rt-defs "SimpleState.h")
gen_rt_def(paht-rt-defs "PreAggregationHashtable.h")
message("cpp files ${runtime_funcs_ptr_libs}")
string (REPLACE " " ";" runtime_funcs_ptr_libs "${runtime_funcs_ptr_libs}")

add_library(runtime_funcs_ptr ${runtime_funcs_ptr_libs})
add_dependencies(runtime_funcs_ptr build_includes)
add_subdirectory(lib/mlir-support)
if(ENABLE_CRANELIFT_BACKEND)
add_subdirectory(lib/execution/cranelift)
endif(ENABLE_CRANELIFT_BACKEND)
add_subdirectory(lib/RelAlg)
add_subdirectory(lib/TupleStream)
add_subdirectory(lib/SubOperator)

add_subdirectory(lib/DB)
add_subdirectory(lib/DSA)
add_subdirectory(lib/util)
add_subdirectory(lib/utility)

add_subdirectory(lib/Conversion/DBToStd)
add_subdirectory(lib/Conversion/DSAToStd)
add_subdirectory(lib/Conversion/RelAlgToSubOp)
add_subdirectory(lib/Conversion/SubOpToControlFlow)

add_subdirectory(lib/Conversion/UtilToLLVM)
if (ENABLE_CRANELIFT_BACKEND)
    add_subdirectory(lib/execution/CraneliftConversions)
endif(ENABLE_CRANELIFT_BACKEND)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${extension_libs}
        ${conversion_libs}
        ${translation_libs}
        MLIROptLib
        MLIRRelAlg
        MLIRTupleStream
        MLIRSubOperator
        MLIRUtil
        MLIRDB
        MLIRDSA
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSupport
        MLIRExecutionEngine
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        MLIRSupport
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRSideEffectInterfaces
        MLIRTransforms
        MLIRMathTransforms
        )
set(PRECOMPILED_CC_PATH "${CMAKE_CURRENT_BINARY_DIR}/precompiled_bitcode.cc")
add_subdirectory(lib/runtime)
add_subdirectory(lib/execution)
add_subdirectory(lib/frontend/SQL)
add_subdirectory(tools)
add_subdirectory(test)





