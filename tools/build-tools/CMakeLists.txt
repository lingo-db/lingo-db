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
function(gen_rt_def target_name header_file)
    add_custom_command(
            COMMAND ${CMAKE_BINARY_DIR}/bin/runtime-header-tool --oh=${CMAKE_BINARY_DIR}/include/runtime-defs/${header_file} --ocpp=${CMAKE_BINARY_DIR}/include/runtime-defs/${header_file}.cpp ${CMAKE_SOURCE_DIR}/include/runtime/${header_file} -- -x c++ --std=c++20 -I ${MLIR_DIR}/../../clang/18/include/ -I ${CMAKE_SOURCE_DIR}/include -I ${ARROW_INCLUDE_DIR} -I ${TBB_INCLUDE_DIR}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            DEPENDS ${CMAKE_BINARY_DIR}/bin/runtime-header-tool
            DEPENDS ${CMAKE_SOURCE_DIR}/include/runtime/${header_file}
            COMMENT "Generate runtime definitions... ${header_file}"
            OUTPUT ${CMAKE_BINARY_DIR}/include/runtime-defs/${header_file}.cpp ${CMAKE_BINARY_DIR}/include/runtime-defs/${header_file}
    )
    add_custom_target(${target_name}
            ALL
            DEPENDS
            ${CMAKE_BINARY_DIR}/include/runtime-defs/${header_file}
            )

    add_dependencies(${target_name} runtime-header-tool)
    add_dependencies(build_includes ${target_name})
    set(runtime_funcs_ptr_libs "${runtime_funcs_ptr_libs} ${CMAKE_BINARY_DIR}/include/runtime-defs/${header_file}.cpp" PARENT_SCOPE)


endfunction()

