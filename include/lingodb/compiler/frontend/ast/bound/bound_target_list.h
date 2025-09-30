#ifndef LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_TARGET_LIST_H
#define LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_TARGET_LIST_H
#include "../ast_node.h"
#include "../parsed_expression.h"
#include "bound_expression.h"

#include <memory>
#include <vector>
namespace lingodb::ast {
class BoundTargetList : public AstNode {
   public:
   BoundTargetList(bool distinct, std::vector<std::shared_ptr<ColumnReference>> targets) : AstNode(NodeType::TARGET_LIST), distinct(distinct), targets(std::move(targets)) {}
   bool distinct = false;
   std::vector<std::shared_ptr<ColumnReference>> targets;
};
} // namespace lingodb::ast

#endif // LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_TARGET_LIST_H
