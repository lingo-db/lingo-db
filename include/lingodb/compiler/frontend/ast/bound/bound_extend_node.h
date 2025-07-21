#pragma once
#include "bound_expression.h"
#include "lingodb/compiler/frontend/ast/table_producer.h"
namespace lingodb::ast {

class BoundExtendNode : public AstNode {
   public:
   static constexpr auto TYPE = NodeType::BOUND_EXTEND_NODE;
   BoundExtendNode(std::string mapName, std::vector<std::shared_ptr<BoundExpression>> extensions);

   std::string mapName;
   std::vector<std::shared_ptr<BoundExpression>> extensions;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
}