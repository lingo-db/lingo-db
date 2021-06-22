set(PRECOMPILED_SRCS
        test.cpp string.cpp hash.cpp date.cpp)

find_program(LLVM_LINK_EXECUTABLE llvm-link HINTS ${LLVM_TOOLS_BINARY_DIR})

find_program(CLANG_EXECUTABLE
        NAMES clang-${LLVM_PACKAGE_VERSION}
        clang-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}
        clang-${LLVM_VERSION_MAJOR} clang
        HINTS ${LLVM_TOOLS_BINARY_DIR})

get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)


# Create bitcode for each of the source files.
foreach(SRC_FILE ${PRECOMPILED_SRCS})
    get_filename_component(SRC_BASE ${SRC_FILE} NAME_WE)
    get_filename_component(ABSOLUTE_SRC ${SRC_FILE} ABSOLUTE)
    set(BC_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SRC_BASE}.bc)
    set(PRECOMPILE_COMMAND)
    if(CMAKE_OSX_SYSROOT)
        list(APPEND PRECOMPILE_COMMAND
                ${CMAKE_COMMAND}
                -E
                env
                SDKROOT=${CMAKE_OSX_SYSROOT})
    endif()
    list(
            APPEND PRECOMPILE_COMMAND
            ${CLANG_EXECUTABLE}
                ${PLATFORM_CLANG_OPTIONS}
                -DNDEBUG # DCHECK macros not implemented in precompiled code
                -DARROW_STATIC # Do not set __declspec(dllimport) on MSVC on Arrow symbols
                -fno-use-cxa-atexit # Workaround for unresolved __dso_handle
                -emit-llvm
                -O3
                -c
                ${ABSOLUTE_SRC}
                -o
                ${BC_FILE}
                -I${CMAKE_SOURCE_DIR}/src
                -I${ARROW_BINARY_DIR}/src
    )
    foreach(dir ${dirs})
        list(
                APPEND PRECOMPILE_COMMAND
                -I ${dir}
        )
    endforeach()
    if(NOT ARROW_USE_NATIVE_INT128)
        list(APPEND PRECOMPILE_COMMAND -I${Boost_INCLUDE_DIR})
    endif()
    add_custom_command(OUTPUT ${BC_FILE} COMMAND ${PRECOMPILE_COMMAND} DEPENDS ${SRC_FILE})
    list(APPEND BC_FILES ${BC_FILE})
endforeach()
set(PRECOMPILED_BC_PATH "${CMAKE_CURRENT_BINARY_DIR}/irhelpers.bc")

set(GANDIVA_PRECOMPILED_BC_PATH "${CMAKE_SOURCE_DIR}/${ARROW_BUILD_DIR}/src/gandiva/irhelpers.bc")
set(PRECOMPILED_CC_IN_PATH "${CMAKE_SOURCE_DIR}/tools/scripts/precompiled_bitcode.cc.in")
# link all of the bitcode files into a single bitcode file.
add_custom_command(OUTPUT ${PRECOMPILED_BC_PATH}
        COMMAND ${LLVM_LINK_EXECUTABLE} -o ${PRECOMPILED_BC_PATH}
        ${BC_FILES} ${GANDIVA_PRECOMPILED_BC_PATH}
        DEPENDS ${BC_FILES})

find_package(PythonInterp)

add_custom_command(OUTPUT ${PRECOMPILED_CC_PATH}
        COMMAND ${PYTHON_EXECUTABLE}
        "${CMAKE_SOURCE_DIR}/tools/scripts/make_precompiled_bitcode.py"
        "${PRECOMPILED_CC_IN_PATH}"
        ${PRECOMPILED_BC_PATH}
        ${PRECOMPILED_CC_PATH}
        DEPENDS ${PRECOMPILED_CC_IN_PATH} ${PRECOMPILED_BC_PATH})
add_custom_target(precompiled ALL
        DEPENDS ${PRECOMPILED_CC_PATH})

