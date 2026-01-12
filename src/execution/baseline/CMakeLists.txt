if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    message(FATAL_ERROR "The TPDE baseline backend can only be built for linux targets.")
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

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(TPDE_TARGET "x86_64-linux")
    set(TPDE_ARCH "opteron")
    set(BC_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc")
    set(HPP_OUTPUT "${SNIPPED_INCLUDE_DIR}/snippet_encoders_x64.hpp")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(TPDE_TARGET "aarch64-linux")
    set(TPDE_ARCH "armv8.5-a")
    set(BC_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/snippets_arm.bc")
    set(HPP_OUTPUT "${SNIPPED_INCLUDE_DIR}/snippet_encoders_arm.hpp")
else()
    message(FATAL_ERROR "Unsupported host architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()

add_custom_command(
        OUTPUT ${BC_OUTPUT}
        COMMAND ${CLANG_EXECUTABLE} -c -emit-llvm -ffreestanding -fcf-protection=none -O3 -fomit-frame-pointer -fno-math-errno --target=${TPDE_TARGET} -march=${TPDE_ARCH} -o ${BC_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/snippets.c
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/snippets.c
        DEPENDS_EXPLICIT_ONLY
        COMMENT "Generating LLVM bitcode for ${CMAKE_HOST_SYSTEM_PROCESSOR}"
)

add_custom_command(
        OUTPUT ${HPP_OUTPUT}
        COMMAND $<TARGET_FILE:tpde::tpde_encodegen> -o ${HPP_OUTPUT} ${BC_OUTPUT}
        DEPENDS ${BC_OUTPUT} tpde::tpde_encodegen
        DEPENDS_EXPLICIT_ONLY
        COMMENT "Generating snippet encoders header for ${CMAKE_HOST_SYSTEM_PROCESSOR}"
)

add_custom_target(snippet_encoders ALL DEPENDS ${HPP_OUTPUT})

# have snippet encoders built with the runner target which includes the baseline execution backend
add_dependencies(runner snippet_encoders)
add_dependencies(build_includes snippet_encoders tpde)
target_include_directories(runner PRIVATE ${SNIPPED_INCLUDE_DIR})

# properly handle asan builds
if (CMAKE_BUILD_TYPE MATCHES "ASAN")
    add_definitions(-DTPDE_ASAN_BUILD)
endif ()
