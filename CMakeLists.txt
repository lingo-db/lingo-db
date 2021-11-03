cmake_minimum_required(VERSION 3.13.4)
project(db-dialects LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
set(LLVM_LINK_COMPONENTS
        Support
        Core
        nativecodegen
        OrcJIT
        )

set(ArrowPython_DIR ${Arrow_DIR})
include(${Arrow_DIR}/FindArrow.cmake)
include(${Arrow_DIR}/FindArrowPython.cmake)
include_directories(${ARROW_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(lib/mlir-support)
add_subdirectory(lib/RelAlg)
add_subdirectory(lib/DB)
add_subdirectory(lib/util)
add_subdirectory(lib/Conversion/DBToArrowStd)
add_subdirectory(lib/Conversion/RelAlgToDB)
add_subdirectory(lib/Conversion/UtilToLLVM)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
        MLIROptLib
        MLIRRelAlg
        MLIRUtil
        MLIRDB
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSupport
        MLIRExecutionEngine
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        MLIRTranslation
        MLIRSupport
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRSideEffectInterfaces
        MLIRTransforms
        )
set(PRECOMPILED_CC_PATH "${CMAKE_CURRENT_BINARY_DIR}/precompiled_bitcode.cc")

add_subdirectory(lib/runtime)
add_subdirectory(lib/runner)


add_executable(mlir-db-opt mlir-opt.cpp)

add_executable(db-run db-run.cpp)
target_link_libraries(db-run runner runtime)
target_link_options(db-run PUBLIC -Wl,--export-dynamic)
add_executable(db-run-query db-run-query.cpp)
target_link_libraries(db-run-query runner runtime)
target_link_options(db-run-query PUBLIC -Wl,--export-dynamic)


llvm_update_compile_flags(mlir-db-opt)
target_link_libraries(mlir-db-opt PUBLIC ${LIBS} PRIVATE arrow_shared)
add_subdirectory(tools)
add_subdirectory(test)


find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(pybind11)

pybind11_add_module(pymlirdbext python/mlirdb.cpp)
target_link_libraries(pymlirdbext PUBLIC runner runtime PRIVATE arrow_shared arrow_python_shared)
target_link_options(pymlirdbext PUBLIC -Wl,--export-dynamic)
