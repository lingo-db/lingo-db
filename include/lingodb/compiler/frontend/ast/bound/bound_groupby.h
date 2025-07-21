#pragma once
#include "lingodb/compiler/frontend/ast/ast_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_expression.h"
#include <memory>
#include <vector>
namespace lingodb::ast {
class BoundGroupByNode : public AstNode {
   public:
   BoundGroupByNode(std::vector<std::shared_ptr<NamedResult>> groupNamedResults);
   //! The total set of all group expressions
   std::vector<std::shared_ptr<NamedResult>> groupNamedResults;

   //TODO having clause

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
}