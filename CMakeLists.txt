cmake_minimum_required(VERSION 3.13.4)
project(db-dialects LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

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
        )

add_subdirectory(include)
add_subdirectory(lib)



get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
        MLIROptLib
        MLIRRelAlg
        MLIRDB
        MLIRIR
        MLIRParser
        MLIRPass

        MLIRTranslation
        MLIRSupport
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRSideEffectInterfaces
        MLIRTransforms
        )


add_llvm_executable(mlir-db-opt mlir-opt.cpp)
target_link_libraries(mlir-db-opt PUBLIC ${LIBS})