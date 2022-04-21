add_subdirectory(precompiled)
add_library(runtime OBJECT
        dynamic_calls/arrow_table_builder.cpp
        dynamic_calls/dump_values.cpp
        dynamic_calls/DataSourceIteration.cpp
        dynamic_calls/Vector.cpp
        dynamic_calls/LazyJoinHashtable.cpp
        dynamic_calls/Hashtable.cpp
        dynamic_calls/string.cpp
        dynamic_calls/hash.cpp
        dynamic_calls/date.cpp
        database.cpp metadata.cpp ArrowDirDatabase.cpp ExternalArrowDatabase.cpp MetaDataOnlyDatabase.cpp)
target_link_libraries(runtime PUBLIC arrow_shared)
target_link_options(runtime PUBLIC -export-dynamic)

