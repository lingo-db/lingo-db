#ifndef LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_GOO_H
#define LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_GOO_H
#include "QueryGraph.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <bitset>
#include <memory>
namespace lingodb::compiler::dialect::relalg {
class GOO {
   QueryGraph& queryGraph;

   public:
   GOO(QueryGraph& qg) : queryGraph(qg) {}

   std::shared_ptr<Plan> solve();
   std::shared_ptr<Plan> createInitialPlan(QueryGraph::Node& n);
};
} // namespace lingodb::compiler::dialect::relalg

#endif //LINGODB_COMPILER_DIALECT_RELALG_TRANSFORMS_QUERYOPT_GOO_H
