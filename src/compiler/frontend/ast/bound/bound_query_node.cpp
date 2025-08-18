#include "lingodb/compiler/frontend/ast/bound/bound_query_node.h"
namespace lingodb::ast {
BoundSetOperationNode::BoundSetOperationNode(std::string alias, SetOperationType setType, bool setOpAll, std::shared_ptr<TableProducer> boundLeft, std::shared_ptr<TableProducer> boundRight, std::shared_ptr<analyzer::SQLScope> leftScope, std::shared_ptr<analyzer::SQLScope> rightScope) : QueryNode(QueryNodeType::BOUND_SET_OPERATION_NODE), setType(setType), setOpAll(setOpAll), boundLeft(boundLeft), boundRight(boundRight), leftScope(leftScope), rightScope(rightScope) {
   this->alias = alias;
}
std::string BoundSetOperationNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

BoundValuesQueryNode::BoundValuesQueryNode(std::string alias, std::shared_ptr<BoundExpressionListRef> expressionListRef) : QueryNode(TYPE), expressionListRef(std::move(expressionListRef)) {
   this->alias = alias;
}
std::string BoundValuesQueryNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

std::string BoundCTENode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
} // namespace lingodb::ast