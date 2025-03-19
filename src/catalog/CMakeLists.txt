add_library(mlirTypes MLIRTypes.cpp)
target_link_libraries(mlirTypes
        PUBLIC ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
)

add_library(catalog Catalog.cpp Types.cpp Defs.cpp)
target_link_libraries(catalog PUBLIC mlirTypes)