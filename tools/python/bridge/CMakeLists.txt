


add_library(pybridge SHARED bridge.cpp custom_dialects.cpp)

target_link_libraries(pybridge PRIVATE  runner runtime mlir-support ${LIBS} PRIVATE Arrow::arrow_static PUBLIC "-Wl,--whole-archive ${LLVM_BINARY_DIR}/lib/libMLIRCAPIIR.a  ${LLVM_BINARY_DIR}/lib/libMLIRCAPIDebug.a  ${LLVM_BINARY_DIR}/lib/libMLIRCAPIInterfaces.a -Wl,--no-whole-archive")
