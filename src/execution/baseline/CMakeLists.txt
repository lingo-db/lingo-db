if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
    message(FATAL_ERROR "Baseline backend is only supported on x86_64 linux systems.")
endif()

## Add baseline backend to the runner target
target_sources(runner PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/BaselineBackend.cpp
)

## Add TPDE as a vendored dependency subdirectory
# properly handle asan builds
if (CMAKE_BUILD_TYPE MATCHES "ASAN")
    add_definitions(-DTPDE_ASAN_BUILD=ON)
endif ()
set(TPDE_INCLUDE_TESTS OFF CACHE BOOL "Include TPDE tests")
set(TPDE_ENABLE_LLVM OFF CACHE BOOL "Enable LLVM support in TPDE")
set(TPDE_ENABLE_COVERAGE OFF CACHE BOOL "Enable coverage support in TPDE")
set(TPDE_BUILD_DOCS OFF CACHE BOOL "Build documentation for TPDE")
set(TPDE_ENABLE_ENCODEGEN ON CACHE BOOL "Enable encodegen in TPDE")
add_subdirectory(${CMAKE_SOURCE_DIR}/vendored/tpde ${CMAKE_BINARY_DIR}/vendored/tpde)
target_link_libraries(runner PRIVATE tpde)
target_include_directories(runner PRIVATE tpde)

## Setup / run tpde encodegen to generate snippet encoders
set(SNIPPED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${SNIPPED_INCLUDE_DIR})
# Step 1: Create LLVM bitcode from encoder snippets
find_program(CLANG_EXECUTABLE NAMES clang-20 clang-19)
if (NOT CLANG_EXECUTABLE)
    message(FATAL_ERROR "Neither clang-20 nor clang-19 was found in PATH.")
endif ()
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc
        COMMAND ${CLANG_EXECUTABLE} -c -emit-llvm -ffreestanding -fcf-protection=none -O3 -fomit-frame-pointer -fno-math-errno --target=x86_64-linux -march=opteron -o ${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc ${CMAKE_CURRENT_SOURCE_DIR}/snippets_x64.c
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/snippets_x64.c
        DEPENDS_EXPLICIT_ONLY
        COMMENT "Generating LLVM bitcode from snippets.c"
)
# Step 2: Use tpde to generate C++ header file from LLVM bitcode
add_custom_command(
        OUTPUT ${SNIPPED_INCLUDE_DIR}/snippet_encoders_x64.hpp
        COMMAND $<TARGET_FILE:tpde::tpde_encodegen> -o ${SNIPPED_INCLUDE_DIR}/snippet_encoders_x64.hpp ${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc
        DEPENDS tpde::tpde_encodegen
        DEPENDS_EXPLICIT_ONLY
        COMMENT "Generating ${SNIPPED_INCLUDE_DIR}/snippet_encoders_x64.hpp from ${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc"
)
add_custom_target(snippet_encoders ALL
        DEPENDS ${SNIPPED_INCLUDE_DIR}/snippet_encoders_x64.hpp
)
# have snippet encoders built with the runner target which includes the baseline execution backend
add_dependencies(runner snippet_encoders)
add_dependencies(build_includes snippet_encoders tpde)
target_include_directories(runner PRIVATE ${SNIPPED_INCLUDE_DIR})
