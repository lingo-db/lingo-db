add_library(runtime OBJECT
        Heap.cpp
        TableBuilder.cpp
        DumpRuntime.cpp
        DataSourceIteration.cpp
        GrowingBuffer.cpp
        Buffer.cpp
        LazyJoinHashtable.cpp
        SegmentTreeView.cpp
        Hashtable.cpp
        StringRuntime.cpp
        FloatRuntime.cpp
        DecimalRuntime.cpp
        IntegerRuntime.cpp
        Hash.cpp
        DateRuntime.cpp
        ExecutionContext.cpp
        Database.cpp MetaData.cpp ArrowDirDatabase.cpp ExternalArrowDatabase.cpp MetaDataOnlyDatabase.cpp)
target_link_libraries(runtime PUBLIC arrow_shared)
target_link_options(runtime PUBLIC -Wl,-export-dynamic)

