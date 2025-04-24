#pragma once
#include "lingodb/compiler/frontend/ast/bound/bound_expression.h"
#include "lingodb/compiler/frontend/ast/ast_node.h"
#include <memory>
#include <vector>
namespace lingodb::ast {
class BoundGroupByNode : public AstNode {
   public:
   BoundGroupByNode(std::vector<std::shared_ptr<BoundExpression>> groupExpressions);
   //! The total set of all group expressions
   std::vector<std::shared_ptr<BoundExpression>> groupExpressions;

   //TODO having clause

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
}