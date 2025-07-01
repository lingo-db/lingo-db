cmake_minimum_required(VERSION 3.13.4)

# Define the library target
add_library(linenoise-ng
        linenoise.cpp
        wcwidth.cpp
        ConvertUTF.cpp
)

# Specify include directories
target_include_directories(linenoise-ng
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)