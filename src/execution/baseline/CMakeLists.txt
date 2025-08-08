if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
    message(FATAL_ERROR "The TPDE baseline backend can only be built for linux x86_64 targets.")
endif()

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

# properly handle asan builds
if (CMAKE_BUILD_TYPE MATCHES "ASAN")
    add_definitions(-DTPDE_ASAN_BUILD)
endif ()
