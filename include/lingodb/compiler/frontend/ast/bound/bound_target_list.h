#ifndef LINGODB_BOUND_TARGET_LIST_H
#define LINGODB_BOUND_TARGET_LIST_H
#include "../ast_node.h"
#include "bound_expression.h"
#include "../parsed_expression.h"

#include <memory>
#include <vector>
namespace lingodb::ast {
class BoundTargetList : public AstNode {
   public:
   BoundTargetList(bool distinct, std::vector<std::shared_ptr<ColumnReference>> targets) : AstNode(NodeType::TARGET_LIST), distinct(distinct), targets(std::move(targets)) {}
   bool distinct = false;
   std::vector<std::shared_ptr<ColumnReference>> targets;

};
}

#endif //LINGODB_BOUND_TARGET_LIST_H
