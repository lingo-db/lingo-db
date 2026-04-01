add_library(runtime OBJECT
        helpers.cpp
        Tracing.cpp
        ArrowColumn.cpp
        ArrowTable.cpp
        ArrowView.cpp
        ThreadLocal.cpp
        Heap.cpp
        DumpRuntime.cpp
        DataSourceIteration.cpp
        Sorting.cpp
        GrowingBuffer.cpp
        Buffer.cpp
        SimpleState.cpp
        LazyJoinHashtable.cpp
        SegmentTreeView.cpp
        Hashtable.cpp
        PreAggregationHashtable.cpp
        HashMultiMap.cpp
        StringRuntime.cpp
        ListRuntime.cpp
        FloatRuntime.cpp
        DecimalRuntime.cpp
        IntegerRuntime.cpp
        Hash.cpp
        Timing.cpp
        DateRuntime.cpp
        ExecutionContext.cpp
        RelationHelper.cpp
        EntryLock.cpp
        #Database.cpp
        #ArrowDirDatabase.cpp
        #ExternalArrowDatabase.cpp
        #MetaDataOnlyDatabase.cpp
        #ExternalHashIndex.cpp
        LingoDBHashIndex.cpp
        Session.cpp
        storage/LingoDBTable.cpp
        storage/Restrictions.cpp
        storage/ExternalTableScan.cpp
)
set(COMPILE_DEFS "")
if (ENABLE_GPU_BACKEND)
    list(APPEND COMPILE_DEFS "GPU_ENABLED=1")
else ()
    list(APPEND COMPILE_DEFS "GPU_ENABLED=0")
endif (ENABLE_GPU_BACKEND)

target_compile_definitions(runtime PUBLIC ${COMPILE_DEFS})

#todo: should not be necessary, but it is for Arrow 23
if("abslAlt" IN_LIST ARROW_SYSTEM_DEPENDENCIES)
    target_link_libraries(runtime PRIVATE Arrow::arrow_static parquet_static absl::strings absl::time absl::bad_optional_access)
else()
    target_link_libraries(runtime PRIVATE Arrow::arrow_static parquet_static)
endif()

add_subdirectory(GPU)
target_link_libraries(runtime PUBLIC GPU-rt scheduler)
