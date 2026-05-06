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
        PythonRuntime.cpp
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
    target_link_libraries(runtime PRIVATE Arrow::arrow_static absl::strings absl::time absl::bad_optional_access)
else()
    target_link_libraries(runtime PRIVATE Arrow::arrow_static)
endif()


add_subdirectory(GPU)
target_link_libraries(runtime PUBLIC GPU-rt scheduler)
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    # Only link libpython when Embed was found. When building the wheel, the
    # host Python process supplies all CPython symbols at load time, so we
    # only need headers (Development.Module).
    if(TARGET Python3::Python)
        target_link_libraries(runtime PUBLIC Python3::Python)
    endif()

    # Tabular Python UDFs hand `arrow::Table` to pyarrow via arrow::py::wrap_table /
    # unwrap_table, which live in the `arrow_python` library shipped inside the
    # pyarrow Python package. Query pyarrow for its include dir and library
    # search path, then link against `libarrow_python.so` (or .dylib on macOS).
    # When pyarrow isn't available at build time, the wheel build supplies
    # PYARROW_INCLUDE_DIR / PYARROW_LIBRARY_DIRS via -Dx=y CMake args.
    #
    # Use the configured Python3_EXECUTABLE if find_package(Python3) discovered
    # one (it's set when the Interpreter component is requested or implicit on
    # some platforms). Fall back to plain `python3` otherwise — top-level
    # CMakeLists.txt asks only for Development.Module/Embed, so on Linux CI
    # the variable is empty and we'd otherwise feed the empty string to
    # execute_process.
    set(_pyarrow_python "${Python3_EXECUTABLE}")
    if(NOT _pyarrow_python)
        set(_pyarrow_python "python3")
    endif()
    if(NOT DEFINED PYARROW_INCLUDE_DIR)
        execute_process(
                COMMAND ${_pyarrow_python} -c "import pyarrow as pa; print(pa.get_include())"
                OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _pyarrow_include_rc)
    endif()
    if(NOT DEFINED PYARROW_LIBRARY_DIRS)
        execute_process(
                COMMAND ${_pyarrow_python} -c "import pyarrow as pa; print(pa.get_library_dirs()[0])"
                OUTPUT_VARIABLE PYARROW_LIBRARY_DIRS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _pyarrow_libdirs_rc)
    endif()
    if(PYARROW_INCLUDE_DIR AND PYARROW_LIBRARY_DIRS)
        message(STATUS "PyArrow include: ${PYARROW_INCLUDE_DIR}")
        message(STATUS "PyArrow lib dir: ${PYARROW_LIBRARY_DIRS}")
        execute_process(
                COMMAND ${_pyarrow_python} -c "import pyarrow; pyarrow.create_library_symlinks()"
                OUTPUT_QUIET ERROR_QUIET)
        target_include_directories(runtime PUBLIC ${PYARROW_INCLUDE_DIR})
        target_link_directories(runtime PUBLIC ${PYARROW_LIBRARY_DIRS})
        target_link_libraries(runtime PUBLIC arrow_python)
    else()
        message(WARNING "pyarrow not found — tabular Python UDFs will fail at runtime")
    endif()
elseif(ENABLE_PYTHON STREQUAL "WASM")
    target_sources(runtime PRIVATE WASM.cpp)
    target_link_libraries(runtime PUBLIC cpython-wasm)
endif()
