add_library(runtime OBJECT
        ThreadLocal.cpp
        Heap.cpp
        TableBuilder.cpp
        DumpRuntime.cpp
        DataSourceIteration.cpp
        GrowingBuffer.cpp
        Buffer.cpp
        LazyJoinHashtable.cpp
        SegmentTreeView.cpp
        Hashtable.cpp
        HashMultiMap.cpp
        StringRuntime.cpp
        FloatRuntime.cpp
        DecimalRuntime.cpp
        IntegerRuntime.cpp
        Hash.cpp
        DateRuntime.cpp
        ExecutionContext.cpp
        Database.cpp
        MetaData.cpp
        ArrowDirDatabase.cpp
        ExternalArrowDatabase.cpp
        MetaDataOnlyDatabase.cpp
        ExternalHashIndex.cpp)
target_link_libraries(runtime PUBLIC arrow_shared PRIVATE TBB::tbb)
target_link_options(runtime PUBLIC -Wl,-export-dynamic)

