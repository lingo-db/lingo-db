#ifndef LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_GROUPBY_H
#define LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_GROUPBY_H

#include "lingodb/compiler/frontend/ast/ast_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_expression.h"
#include <memory>
#include <vector>
namespace lingodb::ast {
/**
 * Used for AGGREGATION pipe operator
 */
class BoundGroupByNode : public AstNode {
   public:
   BoundGroupByNode(std::vector<std::shared_ptr<ColumnReference>> groupColumnReference) : AstNode(NodeType::BOUND_GROUP_BY), groupByColumnReferences(groupColumnReference) {}
   //! The total set of all group by expressions
   std::vector<std::shared_ptr<ColumnReference>> groupByColumnReferences;

   /**
    * Stores the ColumnReferences of the group by expression, which are then aggregated within the corresponding grouping set
    * Example: GROUPING SETS ( (e1,e2), (e1), () )
    * localGroupByColumnReferences = {{e1,e2}, {e1}}
    */
   std::vector<std::vector<std::shared_ptr<ColumnReference>>> localGroupByColumnReferences;

   /**
    * The 'group by' expression must be nullable for grouping sets. These are mapped to nullable types and the resulting references are stored here.
    * See naming old Parser
    */
   std::vector<std::vector<std::shared_ptr<ColumnReference>>> localMapToNullColumnReferences;

   /**
    * Stores the ColumnReferences of the group by expression that are not available in the specific grouping set
    * Example: GROUPING SETS ( (e1,e2), (e1), () )
    * has zero localNotAvailableColumnReferences in the first "round", one in the second "round" as e2 is no longer available/included, ...
    * {{}, {e2}, {e1,e2}}
    */
   std::vector<std::vector<std::shared_ptr<ColumnReference>>> localNotAvailableColumnReferences;
   /**
    * Stores the current grouping state. Used for GROUPING function
    */
   std::vector<std::pair<size_t, std::shared_ptr<ColumnReference>>> localPresentIntval;

   /**
    * Stores the reference/ColumnReference after the union of the different grouping sets
    */
   std::vector<std::vector<std::shared_ptr<ColumnReference>>> unionColumnReferences;
   /**
    * Stores the reference/ColumnReference to the GROUPING function for the group by expression at the specified index
    */
   std::vector<std::pair<size_t, std::shared_ptr<ColumnReference>>> groupingFunctions;
};
} // namespace lingodb::ast
#endif
