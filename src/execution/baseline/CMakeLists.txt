if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
    message(FATAL_ERROR "The TPDE baseline backend can only be built for linux x86_64 targets.")
endif()

set(SNIPPED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${SNIPPED_INCLUDE_DIR})

# Step 1: Create LLVM bitcode from encoder snippets
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc
        COMMAND clang -c -emit-llvm -ffreestanding -fcf-protection=none -O3 -fomit-frame-pointer -fno-math-errno --target=x86_64-linux -march=opteron -o ${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc ${CMAKE_CURRENT_SOURCE_DIR}/snippets_x64.c
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/snippets_x64.c
        COMMENT "Generating LLVM bitcode from snippets.c"
)
# Step 2: Use tpde to generate C++ header file from LLVM bitcode
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/snippet_encoders_x64.hpp
        COMMAND tpde_encodegen -o ${SNIPPED_INCLUDE_DIR}/snippet_encoders_x64.hpp ${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc
        COMMENT "Generating ${SNIPPED_INCLUDE_DIR}/snippet_encoders_x64.hpp from ${CMAKE_CURRENT_BINARY_DIR}/snippets_x64.bc"
)
add_custom_target(snippet_encoders ALL
        DEPENDS snippet_encoders_x64.hpp
)
# have snippet encoders built with the runner target which includes the baseline execution backend
add_dependencies(runner snippet_encoders)
target_include_directories(runner PRIVATE ${SNIPPED_INCLUDE_DIR})
