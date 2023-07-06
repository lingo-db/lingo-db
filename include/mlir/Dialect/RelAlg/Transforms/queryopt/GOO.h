#ifndef MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_GOO_H
#define MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_GOO_H
#include "QueryGraph.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include <llvm/ADT/SmallBitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <bitset>
#include <memory>
namespace mlir::relalg {
class GOO {
   QueryGraph& queryGraph;

   public:
   GOO(QueryGraph& qg) : queryGraph(qg) {}

   std::shared_ptr<Plan> solve();
   std::shared_ptr<Plan> createInitialPlan(QueryGraph::Node& n);
};
} // namespace mlir::relalg

#endif // MLIR_DIALECT_RELALG_TRANSFORMS_QUERYOPT_GOO_H
