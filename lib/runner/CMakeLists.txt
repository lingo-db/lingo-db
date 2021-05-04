# turn the bitcode file into a C++ static data variable.
find_package(PythonInterp)

set(GANDIVA_PRECOMPILED_CC_PATH "${CMAKE_CURRENT_BINARY_DIR}/precompiled_gandiva_bitcode.cc")
set(GANDIVA_PRECOMPILED_BC_PATH "${CMAKE_SOURCE_DIR}/${ARROW_BUILD_DIR}/src/gandiva/irhelpers.bc")
set(GANDIVA_PRECOMPILED_CC_IN_PATH "${CMAKE_SOURCE_DIR}/tools/scripts/precompiled_bitcode.cc.in")
add_custom_command(OUTPUT ${GANDIVA_PRECOMPILED_CC_PATH}
        COMMAND ${PYTHON_EXECUTABLE}
        "${CMAKE_SOURCE_DIR}/tools/scripts/make_precompiled_bitcode.py"
        "${GANDIVA_PRECOMPILED_CC_IN_PATH}"
        ${GANDIVA_PRECOMPILED_BC_PATH}
        ${GANDIVA_PRECOMPILED_CC_PATH}
        DEPENDS ${GANDIVA_PRECOMPILED_CC_IN_PATH})
add_custom_target(precompiled ALL
        DEPENDS ${GANDIVA_PRECOMPILED_CC_PATH})
add_library(runner runner.cpp ${GANDIVA_PRECOMPILED_CC_PATH})
llvm_update_compile_flags(runner)
target_link_libraries(runner PRIVATE ${LIBS} PUBLIC arrow_shared)
add_dependencies(runner precompiled)
