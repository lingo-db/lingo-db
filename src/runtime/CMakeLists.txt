# Replace inline source list with a variable so we can conditionally append files.
set(RUNTIME_SOURCES
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
        TryExcept.cpp
        #Database.cpp
        #ArrowDirDatabase.cpp
        #ExternalArrowDatabase.cpp
        #MetaDataOnlyDatabase.cpp
        #ExternalHashIndex.cpp
        LingoDBHashIndex.cpp
        Session.cpp
        storage/LingoDBTable.cpp
        storage/Restrictions.cpp
        PythonRuntime.cpp
)

if (ENABLE_PYTHON STREQUAL "WASM")
    list(APPEND RUNTIME_SOURCES WASM.cpp ${UNCOMMON_SHARED_SOURCE})
endif()

add_library(runtime OBJECT ${RUNTIME_SOURCES})

set(COMPILE_DEFS "")
if (ENABLE_GPU_BACKEND)
    list(APPEND COMPILE_DEFS "GPU_ENABLED=1")
else ()
    list(APPEND COMPILE_DEFS "GPU_ENABLED=0")
endif (ENABLE_GPU_BACKEND)
list(APPEND COMPILE_DEFS "SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_compile_definitions(runtime PUBLIC ${COMPILE_DEFS})
set(RUNTIME_LINK_LIBARIES Arrow::arrow_static)

#todo: should not be necessary, but it is for Arrow 23
if("abslAlt" IN_LIST ARROW_SYSTEM_DEPENDENCIES)
    list(APPEND RUNTIME_LINK_LIBARIES absl::strings absl::time absl::bad_optional_access)
endif()


add_subdirectory(GPU)
list(APPEND RUNTIME_LINK_LIBARIES GPU-rt scheduler)
if(ENABLE_PYTHON STREQUAL "WASM")
    list(APPEND RUNTIME_LINK_LIBARIES cpython-wasm vmlib -lm)
endif()
target_link_libraries(runtime PUBLIC ${RUNTIME_LINK_LIBARIES})