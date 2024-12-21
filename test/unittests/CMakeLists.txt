
# Find Catch2 (CMake >= 3.11 recommended)
find_package(Catch2 REQUIRED)
include_directories(CATCH2_INCLUDE_DIR)
add_executable(tester
        utility/TestSerialization.cpp
)
target_link_libraries(tester PRIVATE Catch2::Catch2WithMain)
set_target_properties(tester PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

enable_testing()
add_test(NAME UnitTests COMMAND tester)
