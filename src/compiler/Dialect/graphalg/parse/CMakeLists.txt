add_library(GraphAlgParse STATIC
    Lexer.cpp
    Parser.cpp
)
target_link_libraries(GraphAlgParse PRIVATE GraphAlgIR)
