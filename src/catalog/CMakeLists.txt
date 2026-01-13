add_library(mlirTypes MLIRTypes.cpp UDFImplementer.cpp)
target_link_libraries(mlirTypes
        PUBLIC ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
)
add_dependencies(mlirTypes py-rt-defs)

add_library(catalog Catalog.cpp Types.cpp Defs.cpp TableCatalogEntry.cpp IndexCatalogEntry.cpp MetaData.cpp Column.cpp FunctionCatalogEntry.cpp)
add_dependencies(catalog mlirTypes)
target_link_libraries(catalog PUBLIC mlirTypes)

set(COMPILE_DEFS "SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_compile_definitions(mlirTypes PRIVATE ${COMPILE_DEFS})
