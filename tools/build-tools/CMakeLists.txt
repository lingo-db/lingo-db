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

target_compile_options(runtime-header-tool PRIVATE -fno-sanitize=address)
target_link_options(runtime-header-tool PRIVATE -fno-sanitize=address)
function(gen_rt_def_generic target_name header_file output_path input_path lib_prefix result_namespace)
    add_custom_command(
            COMMAND ${CMAKE_BINARY_DIR}/bin/runtime-header-tool --oh=${CMAKE_BINARY_DIR}/${output_path}/${header_file} --ocpp=${CMAKE_BINARY_DIR}/${output_path}/${header_file}.cpp --lib-prefix=${lib_prefix} --result-namespace=${result_namespace} ${CMAKE_SOURCE_DIR}/${input_path}/${header_file} -- -x c++ --std=c++20 -I ${MLIR_DIR}/../../clang/20/include/ -I ${CMAKE_SOURCE_DIR}/include -I ${ARROW_INCLUDE_DIR}
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