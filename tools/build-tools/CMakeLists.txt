add_clang_executable(runtime-header-tool
        runtime-header-tool.cpp
)
target_link_libraries(runtime-header-tool
        PRIVATE
        clangAST
        clangASTMatchers
        clangBasic
        clangFrontend
        clangSerialization
        clangTooling
)

## Query clang-tool flags once (equal for all calls)
# Base flags
set(CLANG_TOOL_FLAGS "-xc++" "--std=c++20" "-v" "-v" "-v")
set(CLANG_TOOL_FLAGS "${CLANG_TOOL_FLAGS}" "-I${CMAKE_SOURCE_DIR}/include")
set(CLANG_TOOL_FLAGS "${CLANG_TOOL_FLAGS}" "-I${ARROW_INCLUDE_DIR}")

# Get Clang resource directory
# Check if the compiler is ccache
if(CMAKE_CXX_COMPILER MATCHES "ccache")
    set(ACTUAL_CXX_COMPILER ${CMAKE_CXX_COMPILER_ARG1})
else()
    set(ACTUAL_CXX_COMPILER ${CMAKE_CXX_COMPILER})
endif()

# Remove whitespace from ACTUAL_CXX_COMPILER
string(STRIP "${ACTUAL_CXX_COMPILER}" ACTUAL_CXX_COMPILER)

# only use resource dir if actual compiler is clang
if (ACTUAL_CXX_COMPILER MATCHES "clang")
    execute_process(
            COMMAND ${ACTUAL_CXX_COMPILER} --print-resource-dir
            OUTPUT_VARIABLE CLANG_RESOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Fail if the result is empty
    if(CLANG_RESOURCE_DIR STREQUAL "")
        message(FATAL_ERROR "Failed to query clang resource directory. `${ACTUAL_CXX_COMPILER} --print-resource-dir` did not return a valid resource directory.")
    endif()
    message(STATUS "Queried clang resource dir: ${CLANG_RESOURCE_DIR}")
    set(CLANG_TOOL_FLAGS "${CLANG_TOOL_FLAGS}" "-resource-dir" "${CLANG_RESOURCE_DIR}")
elseif (CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
    foreach (dir ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
        message(STATUS "Adding implicit include directory: ${dir}")
        set(CLANG_TOOL_FLAGS "${CLANG_TOOL_FLAGS}" "-I${dir}")
    endforeach ()
else ()
    set(CLANG_TOOL_FLAGS "${CLANG_TOOL_FLAGS}" "-I${MLIR_DIR}/../../clang/20/include/")
endif ()

# Add xcode sysroot if building on macOS (required for finding libc++)
if(APPLE)
    execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE MACOS_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CLANG_TOOL_FLAGS "${CLANG_TOOL_FLAGS}" "-isysroot" "${MACOS_SDK_PATH}")
    message(STATUS "Queried macOS SDK path: ${MACOS_SDK_PATH}")
endif()
# cache clang-tool flags
set(CLANG_TOOL_FLAGS "${CLANG_TOOL_FLAGS}" CACHE STRING "Flags for Clang tools" FORCE)
##

target_compile_options(runtime-header-tool PRIVATE -fno-sanitize=address)
target_link_options(runtime-header-tool PRIVATE -fno-sanitize=address)
function(gen_rt_def_generic target_name header_file output_path input_path lib_prefix result_namespace)
    # Retrieve cached value
    set(CLANG_TOOL_FLAGS "${CLANG_TOOL_FLAGS}" CACHE STRING "Flags for Clang tools")
    add_custom_command(
            COMMAND ${CMAKE_BINARY_DIR}/bin/runtime-header-tool --oh=${CMAKE_BINARY_DIR}/${output_path}/${header_file} --ocpp=${CMAKE_BINARY_DIR}/${output_path}/${header_file}.cpp --lib-prefix=${lib_prefix} --result-namespace=${result_namespace} ${CMAKE_SOURCE_DIR}/${input_path}/${header_file} -- ${CLANG_TOOL_FLAGS}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            DEPENDS ${CMAKE_BINARY_DIR}/bin/runtime-header-tool
            DEPENDS ${CMAKE_SOURCE_DIR}/${input_path}/${header_file}
            COMMENT "Generate runtime definitions... ${header_file}"
            OUTPUT ${CMAKE_BINARY_DIR}/${output_path}/${header_file}.cpp ${CMAKE_BINARY_DIR}/${output_path}/${header_file}
    )
    message(STATUS "Adding custom target ${target_name}: ${CMAKE_BINARY_DIR}/${output_path}/${header_file}")
    add_custom_target(${target_name}
            ALL
            DEPENDS
            ${CMAKE_BINARY_DIR}/${output_path}/${header_file}
            ${CMAKE_BINARY_DIR}/${output_path}/${header_file}.cpp
    )

    add_dependencies(${target_name} runtime-header-tool)
    add_dependencies(build_includes ${target_name})
    set(runtime_funcs_ptr_lib "${CMAKE_BINARY_DIR}/${output_path}/${header_file}.cpp" PARENT_SCOPE)


endfunction()

function(gen_rt_def target_name header_file)
    gen_rt_def_generic(${target_name} ${header_file} "include/lingodb/compiler/runtime" "include/lingodb/runtime" "lingodb::runtime::" "lingodb::compiler::runtime")
    set(runtime_funcs_ptr_libs "${runtime_funcs_ptr_libs} ${runtime_funcs_ptr_lib}" PARENT_SCOPE)
endfunction()
