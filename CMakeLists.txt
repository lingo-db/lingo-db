cmake_minimum_required(VERSION 3.13.4)
if (POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif ()
project(db-dialects LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")

option(COMPILE_FOR_PERF "compile for perf" OFF)
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
include_directories(${PROJECT_SOURCE_DIR}/torch-mlir/include)

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

set(ArrowPython_DIR ${Arrow_DIR})
include(${Arrow_DIR}/FindArrow.cmake)
include(${Arrow_DIR}/FindArrowPython.cmake)
include_directories(${ARROW_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(tools/runtime-header-tool)
add_subdirectory(lib/mlir-support)
add_subdirectory(lib/RelAlg)
add_subdirectory(lib/DB)
add_subdirectory(lib/DSA)
add_subdirectory(lib/util)
add_subdirectory(lib/Conversion/DBToArrowStd)
add_subdirectory(lib/Conversion/DSAToStd)
add_subdirectory(lib/Conversion/RelAlgToDB)
add_subdirectory(lib/Conversion/UtilToLLVM)
add_subdirectory(lib/Transforms)

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
        MLIRDSA
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSupport
        MLIRExecutionEngine
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
#        MLIRTranslation
        MLIRSupport
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRSideEffectInterfaces
        MLIRTransforms
        MLIRCustomTransforms
        MLIRMathTransforms
        TorchMLIRInitAll
        TorchMLIRConversionPasses
        TorchMLIRRefBackend
        TorchMLIRTMTensorDialect
        TorchMLIRTMTensorPasses
        TorchMLIRTorchConversionDialect
        TorchMLIRTorchConversionPasses
        TorchMLIRTorchDialect
        TorchMLIRTorchPasses
        TorchMLIRTorchToLinalg
        TorchMLIRTorchToSCF
        TorchMLIRTorchToStd
        TorchMLIRTorchToTosa
        TorchMLIRTorchUtils
        MLIRMathTransforms
        TorchMLIRTMTensorDialect
        TorchMLIRConversionUtils
        TorchMLIRTorchToTMTensor
        )
set(PRECOMPILED_CC_PATH "${CMAKE_CURRENT_BINARY_DIR}/precompiled_bitcode.cc")

add_subdirectory(lib/runtime)
add_subdirectory(lib/runner)
add_subdirectory(lib/frontend/SQL)

add_executable(mlir-db-opt mlir-opt.cpp)

add_executable(db-run-query db-run-query.cpp)
target_link_libraries(db-run-query runner runtime mlir-support)
target_link_options(db-run-query PUBLIC -Wl,--export-dynamic)


llvm_update_compile_flags(mlir-db-opt)
target_link_libraries(mlir-db-opt PUBLIC ${LIBS} runtime mlir-support PRIVATE arrow_shared)
if (COMPILE_FOR_PERF)
    target_compile_definitions(db-run-query PRIVATE RUN_QUERIES_WITH_PERF=1)
else ()
    target_compile_definitions(db-run-query PRIVATE RUN_QUERIES_WITH_PERF=0)
endif (COMPILE_FOR_PERF)

add_executable(mlir-ctxt-opt mlir-ctxt-opt.cpp)
llvm_update_compile_flags(mlir-ctxt-opt)
target_link_libraries(mlir-ctxt-opt PUBLIC ${LIBS} runtime PRIVATE arrow_shared)
add_subdirectory(tools)
add_subdirectory(test)





find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(pybind11)

pybind11_add_module(pymlirdbext python/mlirdb.cpp)
target_link_libraries(pymlirdbext PUBLIC runner runtime PRIVATE arrow_shared arrow_python_shared)
target_link_options(pymlirdbext PUBLIC -Wl,--export-dynamic)




