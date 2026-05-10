#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"

#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"
namespace {
using namespace lingodb::compiler::dialect;

class PrepareLoweringPass : public mlir::PassWrapper<PrepareLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareLoweringPass)
   virtual llvm::StringRef getArgument() const override { return "subop-prepare-lowering"; }
   void runOnOperation() override {
      subop::ColumnCreationAnalysis createdColumns(getOperation());
      std::vector<mlir::Operation*> opsToErase;
      getOperation()->walk([&](subop::NestedMapOp nestedMapOp) { // for each nested map
         llvm::DenseMap<tuples::Column*, size_t> colArgs;
         std::vector<mlir::Attribute> newParams;
         size_t argId = 1;
         for (auto colRefAttr : nestedMapOp.getParameters()) {
            colArgs.insert({&mlir::cast<tuples::ColumnRefAttr>(colRefAttr).getColumn(), argId++});
            newParams.push_back(colRefAttr);
         }
         auto tuple = nestedMapOp.getBody()->getArgument(0);
         for (auto* user : tuple.getUsers()) {
            if (auto combineOp = mlir::dyn_cast<subop::CombineTupleOp>(user)) { // if this tuple gets combined
               if (combineOp->getBlock() != nestedMapOp.getBody()) {
                  nestedMapOp.emitError("NestedMapOp: tuple parameter must only be used in the body block");
                  return signalPassFailure();
               }
               std::unordered_set<tuples::Column*> availableColumns;
               std::function<void(mlir::Value)> computeAvailableColumns = [&](mlir::Value v) { // fills availableColumns
                  if (auto* defOp = v.getDefiningOp()) {
                     auto createdOps = createdColumns.getCreatedColumns(defOp); // gather columns that combineOp's defining op created

                     availableColumns.insert(createdOps.begin(), createdOps.end());
                     for (auto operand : defOp->getOperands()) {
                        if (mlir::isa<tuples::TupleStreamType>(operand.getType())) {
                           computeAvailableColumns(operand);
                        }
                     }
                  }
               };
               std::unordered_set<tuples::Column*> requiredColumns;
               std::function<void(mlir::Operation*, std::unordered_set<tuples::Column*>)> addRequiredColumns = [&](mlir::Operation* op, std::unordered_set<tuples::Column*> availableColumns) {
                  auto created = createdColumns.getCreatedColumns(op); // for the combine op, get the columns it creates
                  availableColumns.insert(created.begin(), created.end()); // aggregate them in availableColumns, it contains all columns that are in the tuple-stream up to the current  combineOp
                  for (auto* usedColumn : subop::ColumnUsageAnalysis::getUsedColumnsForOp(op)) {
                     if (!availableColumns.contains(usedColumn)) {
                        requiredColumns.insert(usedColumn); // if the combineOp uses a column that it doesn't create or that doesn't come from its definitions, the combineOp requires it
                     }
                  }
                  for (auto* user : op->getUsers()) {
                     addRequiredColumns(user, availableColumns); // update the required cols for all users of this combineOp
                  }
               };
               computeAvailableColumns(combineOp);
               addRequiredColumns(combineOp, availableColumns);
               if (requiredColumns.empty()) {
                  combineOp->replaceAllUsesWith(mlir::ValueRange{combineOp.getStream()}); //useless combineop -> just remove
                  opsToErase.push_back(combineOp);
                  continue;
               } else {
                  std::vector<mlir::Attribute> createdByMap;
                  mlir::OpBuilder b(combineOp);
                  mlir::Block* mapBlock = new mlir::Block;
                  {
                     mlir::OpBuilder::InsertionGuard guard(b);
                     b.setInsertionPointToStart(mapBlock);
                     auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
                     std::vector<mlir::Value> mapOpReturnVals;
                     // requiredColumns is an unordered_set of pointers; sort
                     // by (scope, name) so the emitted MapOp's input/computes
                     // attribute order — and the nested_map's parameter order
                     // — are deterministic across runs.
                     std::vector<tuples::Column*> sortedRequired(requiredColumns.begin(), requiredColumns.end());
                     std::sort(sortedRequired.begin(), sortedRequired.end(),
                               [&](tuples::Column* a, tuples::Column* b) {
                                  return colManager.getName(a) < colManager.getName(b);
                               });
                     for (auto* r : sortedRequired) { // we need to introduce every required column to the MapOp (MapOp will replace CombineTupleOp)
                        auto colRefAttr = colManager.createRef(r);
                        auto colDefAttr = colManager.createDef(r);

                        if (!colArgs.contains(r)) { // if the required column is not a param of nested map
                           nestedMapOp.getBody()->addArgument(r->type, b.getUnknownLoc()); // we need to add it to nested map op arguments
                           colArgs.insert({r, argId++});
                           newParams.push_back(colRefAttr); // and include in the parameter
                        }
                        createdByMap.push_back(colDefAttr); // every required column is created/returned by the MapOp
                        mapOpReturnVals.push_back(nestedMapOp.getBody()->getArgument(colArgs[r]));
                     }
                     b.create<tuples::ReturnOp>(b.getUnknownLoc(), mapOpReturnVals);
                  }
                  auto mapOp = b.create<subop::MapOp>(b.getUnknownLoc(), tuples::TupleStreamType::get(b.getContext()), combineOp.getStream(), b.getArrayAttr(createdByMap), b.getArrayAttr({}));
                  mapOp.getFn().push_back(mapBlock);
                  combineOp->replaceAllUsesWith(mlir::ValueRange{mapOp.getResult()}); // replace combineOp with mapOp
                  opsToErase.push_back(combineOp);
                  createdColumns.update(mapOp);
               }

            } else {
               nestedMapOp.emitError("NestedMapOp: tuple parameter must only be used with CombineTupleOp");
               return signalPassFailure();
            }
         }
         nestedMapOp.setParametersAttr(mlir::ArrayAttr::get(&getContext(), newParams));
      });
      for (auto* op : opsToErase) {
         op->erase();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createPrepareLoweringPass() { return std::make_unique<PrepareLoweringPass>(); }