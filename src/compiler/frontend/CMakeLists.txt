set(BISON_INPUT_FILE ${CMAKE_SOURCE_DIR}/src/compiler/frontend/parser.y)
set(FLEX_INPUT_FILE ${CMAKE_SOURCE_DIR}/src/compiler/frontend/lexer.l)
BISON_TARGET(parser ${BISON_INPUT_FILE} ${GEN_OUTPUT_DIR}/parser.cpp)
FLEX_TARGET(lexer ${FLEX_INPUT_FILE} ${GEN_OUTPUT_DIR}/lexer.cpp)


ADD_FLEX_BISON_DEPENDENCY(lexer parser)
include_directories(lingo-db-sql-translator ${GEN_OUTPUT_DIR})
add_library(sql-frontend-new
        #analyzer
        sql_analyzer.cpp
        sql_context.cpp

        ast/bound/bound_tableref.cpp
        ast/bound/bound_expression.cpp
        ast/bound/bound_resultmodifier.cpp
        ast/bound/bound_groupby.cpp
        ast/bound/bound_aggregation.cpp
        ast/bound/bound_extend_node.cpp
        ast/bound/bound_insert_node.cpp
        ast/bound/bound_pipe_operator.cpp
        ast/bound/bound_query_node.cpp
        #translator
        sql_mlir_translator.cpp

        translation_context.cpp
#parser
        driver.cpp

        ast/parsed_expression.cpp
        ast/tableref.cpp
        ast/group_by_node.cpp
        ast/pipe_operator.cpp
        ast/aggregation_node.cpp
        ast/result_modifier.cpp
        ast/extend_node.cpp
        ast/select_node.cpp
        ast/cte_node.cpp
        ast/create_node.cpp
        ast/constraint.cpp
        ast/insert_node.cpp
        ast/query_node.cpp

        ${BISON_parser_OUTPUTS}
        ${FLEX_lexer_OUTPUTS}

)