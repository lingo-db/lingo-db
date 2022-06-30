add_library(runtime OBJECT
        TableBuilder.cpp
        DumpRuntime.cpp
        DataSourceIteration.cpp
        Vector.cpp
        LazyJoinHashtable.cpp
        Hashtable.cpp
        StringRuntime.cpp
        Hash.cpp
        DateRuntime.cpp
        ExecutionContext.cpp
        Database.cpp MetaData.cpp ArrowDirDatabase.cpp ExternalArrowDatabase.cpp MetaDataOnlyDatabase.cpp)
target_link_libraries(runtime PUBLIC arrow_shared)
target_link_options(runtime PUBLIC -export-dynamic)

