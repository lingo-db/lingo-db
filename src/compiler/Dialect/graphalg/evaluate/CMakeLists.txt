add_library(GraphAlgEvaluate
    Evaluator.cpp
)
target_link_libraries(
    GraphAlgEvaluate
    PRIVATE
    GraphAlgIR
)
