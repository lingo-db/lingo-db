# Enable ExternalProject CMake module
include(ExternalProject)

# Set default ExternalProject root directory
set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/Rust)

ExternalProject_Add(
        rust_cranelift
        DOWNLOAD_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND cargo build --release --target-dir ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift
        INSTALL_COMMAND  cbindgen --config cbindgen.toml --crate ccranelift --output ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/ccranelift.h
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/lib/execution/cranelift/rust-cranelift"
        BINARY_DIR "${CMAKE_SOURCE_DIR}/lib/execution/cranelift/rust-cranelift"
        INSTALL_DIR "${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift"
        LOG_BUILD ON)
include_directories(${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift)
add_dependencies(build_includes rust_cranelift)
add_mlir_dialect_library(MLIRCranelift
        CraneliftDialect.cpp
        CraneliftOps.cpp
        Translator.cpp
        CraneliftTypes.cpp
        CraneliftEnums.cpp
        ADDITIONAL_HEADER_DIRS
        ../../include/mlir/Dialect/cranelift

        DEPENDS
        MLIRCraneliftOpsIncGen
        rust_cranelift

        LINK_LIBS PRIVATE ccranelift PUBLIC

        MLIRIR
        )
target_link_directories( MLIRCranelift PUBLIC ${CMAKE_BINARY_DIR}/lib/execution/cranelift/rust-cranelift/release)

