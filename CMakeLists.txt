cmake_minimum_required(VERSION 3.13.4)
if (POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif ()
project(lingodb LANGUAGES CXX C)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DTRACER")



# Define ASAN build type if not already set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Add custom ASAN build type
set(CMAKE_CXX_FLAGS_ASAN "-fsanitize=address -fno-omit-frame-pointer -g -O1 -DASAN_ACTIVE")
set(CMAKE_C_FLAGS_ASAN "-fsanitize=address -fno-omit-frame-pointer -g -O1 -DASAN_ACTIVE")
set(CMAKE_EXE_LINKER_FLAGS_ASAN "-fsanitize=address")
# Apply the flags if the build type is ASAN
if(CMAKE_BUILD_TYPE STREQUAL "ASAN")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_ASAN}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_ASAN}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_ASAN}")
endif()

include(FindZLIB)
message("Using ZLIB: ${ZLIB_INCLUDE_DIRS}")
message("Using Python3: ${Python3_EXECUTABLE}")

find_package(Arrow 19 REQUIRED)

option(ENABLE_GPU_BACKEND "enable GPU backend" OFF)


find_package(Clang 20.1 REQUIRED)
find_package(LLVM 20.1 REQUIRED)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
set (MLIR_DIR "${LLVM_DIR}/../mlir")
find_package(MLIR 20.1 REQUIRED)

set(LLVM_BUILD_STATIC ON)


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
include_directories(${ARROW_INCLUDE_DIR})

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
gen_rt_def(tracing-rt-defs "Tracing.h")
message("cpp files ${runtime_funcs_ptr_libs}")
string (REPLACE " " ";" runtime_funcs_ptr_libs "${runtime_funcs_ptr_libs}")

add_library(runtime_funcs_ptr ${runtime_funcs_ptr_libs})
add_dependencies(runtime_funcs_ptr build_includes)
add_subdirectory(src/compiler/mlir-support)

add_subdirectory(src/compiler/Dialect/RelAlg)
add_subdirectory(src/compiler/Dialect/TupleStream)
add_subdirectory(src/compiler/Dialect/SubOperator)

add_subdirectory(src/compiler/Dialect/DB)
add_subdirectory(src/compiler/Dialect/DSA)
add_subdirectory(src/compiler/Dialect/util)
add_subdirectory(src/utility)
add_subdirectory(src/scheduler)

add_subdirectory(src/compiler/Conversion/DBToStd)
add_subdirectory(src/compiler/Conversion/DSAToStd)
add_subdirectory(src/compiler/Conversion/RelAlgToSubOp)
add_subdirectory(src/compiler/Conversion/SubOpToControlFlow)

add_subdirectory(src/compiler/Conversion/UtilToLLVM)

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
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        MLIRSupport
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRSideEffectInterfaces
        MLIRTransforms
        MLIRMathTransforms
        MLIRExecutionEngine
)
set(PRECOMPILED_CC_PATH "${CMAKE_CURRENT_BINARY_DIR}/precompiled_bitcode.cc")
add_subdirectory(src/runtime)
add_subdirectory(src/execution)
add_subdirectory(src/compiler/frontend/SQL)
add_subdirectory(src/tools)
add_subdirectory(tools)
add_subdirectory(test)





