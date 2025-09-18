   #ifndef LINGODB_TARGET_LIST_H
#define LINGODB_TARGET_LIST_H
#include "ast_node.h"
#include "parsed_expression.h"

#include <memory>
#include <vector>
namespace lingodb::ast {
class TargetList : public AstNode {
   public:
   TargetList() : AstNode(NodeType::TARGET_LIST) {}
   std::vector<std::shared_ptr<ParsedExpression>> targets;
   bool distinct = false;
};
} // namespace lingodb::ast

#endif //LINGODB_TARGET_LIST_H
