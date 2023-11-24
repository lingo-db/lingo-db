add_library(runtime OBJECT
        RecordBatchInfo.cpp
        ThreadLocal.cpp
        Heap.cpp
        TableBuilder.cpp
        DumpRuntime.cpp
        DataSourceIteration.cpp
        GrowingBuffer.cpp
        Buffer.cpp
        SimpleState.cpp
        LazyJoinHashtable.cpp
        SegmentTreeView.cpp
        Hashtable.cpp
        PreAggregationHashtable.cpp
        HashMultiMap.cpp
        StringRuntime.cpp
        FloatRuntime.cpp
        DecimalRuntime.cpp
        IntegerRuntime.cpp
        Hash.cpp
        Timing.cpp
        DateRuntime.cpp
        ExecutionContext.cpp
        RelationHelper.cpp
        #Database.cpp
        MetaData.cpp
        #ArrowDirDatabase.cpp
        #ExternalArrowDatabase.cpp
        #MetaDataOnlyDatabase.cpp
        #ExternalHashIndex.cpp
        HashIndex.cpp
        Relation.cpp
        Session.cpp
        Catalog.cpp)
target_link_libraries(runtime PRIVATE tbb arrow)

