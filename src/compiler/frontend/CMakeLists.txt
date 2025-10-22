set(GEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/include/lingodb/compiler/frontend/generated )
set(BISON_INPUT_FILE ${CMAKE_SOURCE_DIR}/src/compiler/frontend/parser.y)
set(FLEX_INPUT_FILE ${CMAKE_SOURCE_DIR}/src/compiler/frontend/lexer.l)
BISON_TARGET(parser ${BISON_INPUT_FILE} ${GEN_OUTPUT_DIR}/parser.cpp)
FLEX_TARGET(lexer ${FLEX_INPUT_FILE} ${GEN_OUTPUT_DIR}/lexer.cpp)


set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.83.0 REQUIRED COMPONENTS context)
include_directories(${Boost_INCLUDE_DIRS})

ADD_FLEX_BISON_DEPENDENCY(lexer parser)

add_library(MLIRSQLFrontendGen
        ${BISON_parser_OUTPUTS}
        ${FLEX_lexer_OUTPUTS}
)
add_dependencies(MLIRSQLFrontendGen build_dialect_includes )
add_dependencies(build_includes MLIRSQLFrontendGen)
include_directories(lingo-db-sql-translator ${GEN_OUTPUT_DIR})
add_library(MLIRSQLFrontend
        #analyzer
        sql_analyzer.cpp
        sql_context.cpp

        #translator
        sql_mlir_translator.cpp

        translation_context.cpp
#parser
        driver.cpp

        ast/parsed_expression.cpp
        #common
        frontend_error.cpp
        frontend_type.cpp

        ${BISON_parser_OUTPUTS}
        ${FLEX_lexer_OUTPUTS}

)

set(COMPILE_DEFS "SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_compile_definitions(MLIRSQLFrontend PRIVATE ${COMPILE_DEFS})

add_dependencies(MLIRSQLFrontend ec-rt-defs MLIRSQLFrontendGen)
target_link_libraries(MLIRSQLFrontend PUBLIC ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
        runtime_funcs_ptr
        utility
        catalog
        Boost::context


)