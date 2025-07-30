#pragma once
#include "lingodb/compiler/frontend/ast/ast_node.h"
#include "lingodb/compiler/frontend/ast/bound/bound_expression.h"
#include <memory>
#include <vector>
namespace lingodb::ast {
class BoundGroupByNode : public AstNode {
   public:
   BoundGroupByNode(std::vector<std::shared_ptr<NamedResult>> groupNamedResults, std::vector<std::set<size_t>> groupingSet);
   //! The total set of all group expressions
   std::vector<std::shared_ptr<NamedResult>> groupNamedResults;

   std::vector<std::vector<std::shared_ptr<NamedResult>>>   localGroupByNamedResults, localMapToNullNamedResults,
                                                            localNotAvailableNamedResults, localAggregationNamedResults;

   std::vector<std::pair<size_t, std::shared_ptr<NamedResult>>>  localPresentIntval;


   std::vector<std::vector<std::shared_ptr<NamedResult>>> unionNamedResults;

   std::vector<std::pair<size_t, std::shared_ptr<NamedResult>>>  groupingFunctions;

   std::vector<std::set<size_t>> groupingSet;




   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
}