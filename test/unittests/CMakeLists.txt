
# Find Catch2 (CMake >= 3.11 recommended)
find_package(Catch2 REQUIRED)
include_directories(CATCH2_INCLUDE_DIR)
add_executable(tester
        catalog/TestCatalog.cpp
        catalog/TestTypes.cpp
        catalog/TestMetaData.cpp
        catalog/TestCatalogEntries.cpp
        runtime/TestUTF8.cpp
        runtime/StringRuntime.cpp
        runtime/DateRuntime.cpp
        storage/TestStorage.cpp
        utility/TestSerialization.cpp
        execution/TestBaselineUtils.cpp
)
target_link_libraries(tester PRIVATE Catch2::Catch2WithMain ${LIBS} utility catalog MLIRSQLFrontend runtime runner sql-frontend-new)

set_target_properties(tester PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

enable_testing()
add_test(NAME UnitTests COMMAND tester)
