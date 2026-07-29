set(COMMON_LIBS runner utility runtime mlir-support scheduler)

# Find Catch2 (CMake >= 3.11 recommended)
find_package(Catch2 REQUIRED)
include_directories(CATCH2_INCLUDE_DIR)

set(TESTER_SOURCES
        catalog/TestCatalog.cpp
        catalog/TestTypes.cpp
        catalog/TestMetaData.cpp
        catalog/TestCatalogEntries.cpp
        runtime/TestUTF8.cpp
        runtime/StringRuntime.cpp
        runtime/DateRuntime.cpp
        storage/TestStorage.cpp
        utility/TestSerialization.cpp
        utility/TestHyperLogLog.cpp
        execution/TestBaselineUtils.cpp
)

if(ENABLE_PARQUET_SCANNER)
    list(APPEND TESTER_SOURCES storage/TestScanParquetFileTask.cpp)
endif()

add_executable(tester ${TESTER_SOURCES})
target_link_libraries(tester PRIVATE Catch2::Catch2WithMain ${LIBS} ${COMMON_LIBS})
set_target_properties(tester PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

enable_testing()
add_test(NAME UnitTests COMMAND tester)
