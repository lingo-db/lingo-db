#include "lingodb/compiler/frontend/ast/bound/bound_tableref.h"
namespace lingodb::ast {
/*
 * BoundBase Table
*/

BoundBaseTableRef::BoundBaseTableRef(std::vector<std::shared_ptr<NamedResult>> namedResultsEntries, std::string alias, std::string relationName, std::string mlirScope) : BoundTableRef(TYPE, std::move(alias)), namedResultsEntries(std::move(namedResultsEntries)), mlirScope(mlirScope), relationName(relationName) {
}
std::string BoundBaseTableRef::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot;

   // Create node identifier for the base table reference
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Create label with table information
   std::string label = "BoundBaseTable\\n table:  tableCatalogEntry->getName() \\n tableCategoryEntry";

   //TODO
   // Add alias if it's not empty

   // Create the node with all information
   dot += nodeId + " [label=\"" + label + "\"];\n";

   return dot;
}

BoundJoinRef::BoundJoinRef(JoinType type, JoinCondType refType, std::shared_ptr<TableProducer> left, std::shared_ptr<TableProducer> right, boundJoinCond condition) : BoundTableRef(TYPE), type(type), refType(refType), left(std::move(left)), right(std::move(right)), condition(std::move(condition)) {
}
std::string BoundJoinRef::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

BoundCrossProductRef::BoundCrossProductRef(std::vector<std::shared_ptr<TableProducer>> boundTables) : BoundTableRef(TYPE), boundTables(boundTables) {
}
std::string BoundCrossProductRef::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

BoundSubqueryRef::BoundSubqueryRef(std::shared_ptr<analyzer::SQLScope> sqlScope, std::shared_ptr<TableProducer> subSelect) : BoundTableRef(TYPE), sqlScope(sqlScope), subSelect(subSelect) {
}
std::string BoundSubqueryRef::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

BoundExpressionListRef::BoundExpressionListRef(std::vector<std::vector<std::shared_ptr<BoundConstantExpression>>> values, std::vector<std::shared_ptr<NamedResult>> namedResultsEntries) : BoundTableRef(TYPE), values(std::move(values)), namedResultsEntries(std::move(namedResultsEntries)) {
}
std::string BoundExpressionListRef::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
} // namespace lingodb::ast