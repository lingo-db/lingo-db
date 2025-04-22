if (APPLE)
    # Because the ld linker on macos does currently not offer to load all symbols from multiple archives (-force_load
    # only applies to a single archive), we need to combine the MLIR CAPI libraries into a single archive.

    # Define paths for combining MLIR CAPI libraries
    set(MLIR_CAPI_COMBINED_DIR ${CMAKE_CURRENT_BINARY_DIR}/mlir_capi_combined)
    set(MLIR_CAPI_COMBINED_LIB ${MLIR_CAPI_COMBINED_DIR}/libmlircapicombined.a)
    set(MLIR_CAPI_LIBS
            ${LLVM_BINARY_DIR}/lib/libMLIRCAPIIR.a
            ${LLVM_BINARY_DIR}/lib/libMLIRCAPIDebug.a
            ${LLVM_BINARY_DIR}/lib/libMLIRCAPIInterfaces.a
            ${LLVM_BINARY_DIR}/lib/libMLIRCAPITransforms.a
    )

    # Custom command to extract MLIR CAPI object files and create a combined archive
    add_custom_command(
            OUTPUT ${MLIR_CAPI_COMBINED_LIB}
            # 0. Ensure the target directory is clean before starting
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${MLIR_CAPI_COMBINED_DIR}
            # 1. Create the directory
            COMMAND ${CMAKE_COMMAND} -E make_directory ${MLIR_CAPI_COMBINED_DIR}
            # 2, 3 & 4. Navigate, extract, remove intermediate SYMDEF, and create combined archive
            # (in sh since the glob otherwise doesn't work)
            COMMAND ${CMAKE_COMMAND} -E env "AR=${CMAKE_AR}" sh -c "cd ${MLIR_CAPI_COMBINED_DIR} && \
                \"$AR\" x ${LLVM_BINARY_DIR}/lib/libMLIRCAPIIR.a && rm -f ./__.SYMDEF && \
                \"$AR\" x ${LLVM_BINARY_DIR}/lib/libMLIRCAPIDebug.a && rm -f ./__.SYMDEF && \
                \"$AR\" x ${LLVM_BINARY_DIR}/lib/libMLIRCAPIInterfaces.a && rm -f ./__.SYMDEF && \
                \"$AR\" x ${LLVM_BINARY_DIR}/lib/libMLIRCAPITransforms.a && rm -f ./__.SYMDEF && \
                \"$AR\" rcs libmlircapicombined.a *.o"
            DEPENDS ${MLIR_CAPI_LIBS}
            COMMENT "Combining MLIR CAPI static libraries into ${MLIR_CAPI_COMBINED_LIB}"
            VERBATIM # Ensure shell commands like 'cd', '&&', and '*' work correctly
    )
    # Custom target to ensure the combined library is built
    add_custom_target(mlir_capi_combined ALL DEPENDS ${MLIR_CAPI_COMBINED_LIB})
endif()


if (NOT CMAKE_BUILD_TYPE STREQUAL "ASAN")
    add_library(pybridge SHARED bridge.cpp custom_dialects.cpp)
    target_link_libraries(pybridge
            PRIVATE runner runtime mlir-support scheduler ${LIBS} Arrow::arrow_static
    )

    if (APPLE)
        # Ensure the combined library is created before pybridge is linked
        add_dependencies(pybridge mlir_capi_combined)
        target_link_options(pybridge
                PUBLIC
                -force_load ${MLIR_CAPI_COMBINED_LIB}
        )
        target_link_options(pybridge PUBLIC -exported_symbols_list ${CMAKE_CURRENT_SOURCE_DIR}/symbols.list.macos)
    else ()
        target_link_options(pybridge
                PUBLIC -Wl,--whole-archive
                ${LLVM_BINARY_DIR}/lib/libMLIRCAPIIR.a
                ${LLVM_BINARY_DIR}/lib/libMLIRCAPIDebug.a
                ${LLVM_BINARY_DIR}/lib/libMLIRCAPIInterfaces.a
                ${LLVM_BINARY_DIR}/lib/libMLIRCAPITransforms.a
                -Wl,--no-whole-archive
        )
        target_link_options(pybridge PUBLIC -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/symbols.list.gnu)
    endif ()
endif ()
