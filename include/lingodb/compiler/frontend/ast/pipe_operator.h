#pragma once
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
   UNION = 7,
   UNION_ALL = 8,
   INTERSECT = 9,
   EXCEPT = 10,
   FROM = 11,
   AGGREGATE = 12,
   EXTEND = 13,
   MAP = 14,

};
class PipeOperator : public TableProducer {
   public:
   PipeOperator(PipeOperatorType pipeOpType, std::shared_ptr<AstNode> node);
   PipeOperatorType pipeOpType;
   std::shared_ptr<AstNode> node;

   std::shared_ptr<TableProducer> input = nullptr;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
} // namespace lingodb::ast