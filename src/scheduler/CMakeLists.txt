
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.83.0 REQUIRED COMPONENTS context)
include_directories(${Boost_INCLUDE_DIRS})

add_library(scheduler Scheduler.cpp)
target_link_libraries(scheduler Boost::context)