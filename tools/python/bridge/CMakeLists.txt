


add_library(pybridge SHARED bridge.cpp)

target_link_libraries(pybridge PRIVATE  runner runtime mlir-support ${LIBS} PRIVATE -static-libstdc++ arrow)
#target_link_options(pybridge PUBLIC -Wl,--export-dynamic)


#target_link_directories( pymlirdbext PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)
#
#
#find_package(Python COMPONENTS Interpreter Development REQUIRED)
#
#message("command: ${Python_EXECUTABLE} -c import pyarrow; print(pyarrow.get_include()) ")
#execute_process(COMMAND ${Python_EXECUTABLE} -c "import pyarrow; print(pyarrow.get_include())"
#        OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
#        RESULT_VARIABLE PYARROW_FOUND
#        )
#if(NOT PYARROW_FOUND EQUAL "0")
#    message(FATAL_ERROR "pyarrow headers not found")
#endif()
#message("pyarrow include: ${PYARROW_INCLUDE_DIR}")
#include_directories("${PYARROW_INCLUDE_DIR}")
