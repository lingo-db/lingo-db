#pragma once
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
   BoundGroupByNode(std::vector<std::shared_ptr<NamedResult>> groupNamedResults) : AstNode(NodeType::BOUND_GROUP_BY), groupByNamedResults(groupNamedResults) {}
   //! The total set of all group by expressions
   std::vector<std::shared_ptr<NamedResult>> groupByNamedResults;

   /**
    * Stores the NamedResults of the group by expression, which are then aggregated within the corresponding grouping set
    * Example: GROUPING SETS ( (e1,e2), (e1), () )
    * localGroupByNamedResults = {{e1,e2}, {e1}}
    */
   std::vector<std::vector<std::shared_ptr<NamedResult>>>   localGroupByNamedResults;

   /**
   * Stores the NamedResults of the aggregation functions for each grouping set
   * This is necessary because the scope of the aggregation functions changes between the different grouping sets
   */
   std::vector<std::vector<std::shared_ptr<NamedResult>>> localAggregationNamedResults;

   /**
    * The 'group by' expression must be nullable for grouping sets. These are mapped to nullable types and the resulting references are stored here.
    * See naming old Parser
    */
   std::vector<std::vector<std::shared_ptr<NamedResult>>> localMapToNullNamedResults;

   /**
    * Stores the NamedResults of the group by expression that are not available in the specific grouping set
    * Example: GROUPING SETS ( (e1,e2), (e1), () )
    * has zero notAvailableNamedResults in the first "round", one in the second "round" as e2 is no longer available/included, ...
    * {{}, {e2}, {e1,e2}}
    */
   std::vector<std::vector<std::shared_ptr<NamedResult>>> localNotAvailableNamedResults;
   /**
    * Stores the current grouping state. Used for GROUPING function
    */
   std::vector<std::pair<size_t, std::shared_ptr<NamedResult>>>  localPresentIntval;

   /**
    * Stores the reference/NamedResult after the union of the different grouping sets
    */
   std::vector<std::vector<std::shared_ptr<NamedResult>>> unionNamedResults;
   /**
    * Stores the reference/NamedResult to the GROUPING function for the group by expression at the specified index
    */
   std::vector<std::pair<size_t, std::shared_ptr<NamedResult>>>  groupingFunctions;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override {
      return "";
   };
};
}