#ifndef LINGODB_COMPILER_FRONTEND_AST_PIPE_OPERATOR_H
#define LINGODB_COMPILER_FRONTEND_AST_PIPE_OPERATOR_H

#include "parsed_expression.h"
#include "table_producer.h"
#include "tableref.h"

#include <memory>
namespace lingodb::ast {
enum class PipeOperatorType : uint8_t {
   WHERE = 1,
   SELECT = 2,
   JOIN = 3,
   GROUP_BY = 4,
   RESULT_MODIFIER = 5,
   //UNION, INTERSECT...
   SET_OPERATION = 7,
   INTERSECT = 9,
   EXCEPT = 10,
   FROM = 11,
   AGGREGATE = 12,
   EXTEND = 13,
   DROP = 14,
   SET = 15,

};
class PipeOperator : public TableProducer {
   public:
   PipeOperator(PipeOperatorType pipeOpType, std::shared_ptr<AstNode> node) : TableProducer(NodeType::PIPE_OP), pipeOpType(pipeOpType), node(node) {}
   PipeOperatorType pipeOpType;
   std::shared_ptr<AstNode> node;

   std::shared_ptr<TableProducer> input = nullptr;
};
} // namespace lingodb::ast
#endif
