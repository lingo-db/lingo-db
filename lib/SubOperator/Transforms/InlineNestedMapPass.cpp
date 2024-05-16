#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"

#include "mlir/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"

#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <queue>
namespace {

class InlineNestedMapPass : public mlir::PassWrapper<InlineNestedMapPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlineNestedMapPass)
   virtual llvm::StringRef getArgument() const override { return "subop-nested-map-inline"; }

   void runOnOperation() override {
      std::vector<mlir::subop::NestedMapOp> nestedMapOps;
      getOperation()->walk([&](mlir::subop::NestedMapOp nestedMapOp) {
         nestedMapOps.push_back(nestedMapOp);
      });
      for (auto nestedMap : nestedMapOps) {
         auto returnOp = mlir::dyn_cast<mlir::tuples::ReturnOp>(nestedMap.getRegion().front().getTerminator());
         if (!returnOp) {
            nestedMap.emitError("NestedMapOp must be terminated by a ReturnOp");
            return signalPassFailure();
         }
         if (returnOp->getNumOperands() != 1) {
            continue;
         }

         mlir::Value streamResult = nestedMap.getResult();
         auto builder = mlir::OpBuilder(returnOp);
         mlir::Value replacement = builder.create<mlir::subop::CombineTupleOp>(nestedMap.getLoc(), returnOp.getOperand(0), nestedMap.getRegion().front().getArgument(0));
         std::vector<mlir::Operation*> opsToMove;
         std::queue<std::tuple<mlir::OpOperand&, mlir::Value, bool>> opsToProcess;
         for (auto& use : streamResult.getUses()) {
            opsToProcess.push({use, streamResult, false});
         }
         std::vector<mlir::subop::UnionOp> unions;
         while (!opsToProcess.empty()) {
            auto [currentUse, v, encounteredUnion] = opsToProcess.front();
            auto* op = currentUse.getOwner();
            opsToProcess.pop();

            if (auto unionOp = mlir::dyn_cast<mlir::subop::UnionOp>(op)) {
               for (auto& use : unionOp.getResult().getUses()) {
                  opsToProcess.push({use, v, true});
               }
               std::vector<mlir::Value> args(unionOp.getOperands().begin(), unionOp.getOperands().end());
               args.erase(args.begin() + currentUse.getOperandNumber());
               unionOp->setOperands(args);
               unions.push_back(unionOp);
               continue;
            }
            if (std::find(opsToMove.begin(), opsToMove.end(), op) != opsToMove.end()) {
               continue;
            }
            if (encounteredUnion) {
               mlir::OpBuilder builder(op);
               mlir::IRMapping mapping;
               mapping.map(currentUse.get(), v);
               auto* cloned = builder.clone(*op, mapping);
               opsToMove.push_back(cloned);
               for (auto& use : op->getUses()) {
                  if (use.get().getType().isa_and_nonnull<mlir::tuples::TupleStreamType>()) {
                     opsToProcess.push({use, mapping.lookup(use.get()), encounteredUnion});
                  }
               }
            } else {
               opsToMove.push_back(op);
               for (auto& use : op->getUses()) {
                  if (use.get().getType().isa_and_nonnull<mlir::tuples::TupleStreamType>()) {
                     opsToProcess.push({use, use.get(), encounteredUnion});
                  }
               }
            }
         }
         for (auto* op : opsToMove) {
            op->moveBefore(returnOp);
         }
         streamResult.replaceAllUsesWith(replacement);
         returnOp->setOperands({});
         for (auto unionOp : unions) {
            if (unionOp.getNumOperands() == 1) {
               unionOp->replaceAllUsesWith(unionOp.getOperands());
               unionOp->erase();
            } else if (unionOp.getNumOperands() == 0) {
               std::function<void(mlir::OpOperand&)> removeFn = [&](mlir::OpOperand& use) {
                  if (auto unionOp = mlir::dyn_cast<mlir::subop::UnionOp>(use.getOwner())) {
                     std::vector<mlir::Value> args(unionOp.getOperands().begin(), unionOp.getOperands().end());
                     args.erase(args.begin() + use.getOperandNumber());
                     unionOp->setOperands(args);
                     unions.push_back(unionOp);
                  } else {
                     for (auto& use : use.getOwner()->getUses()) {
                        removeFn(use);
                     }
                     use.getOwner()->erase();
                  }
               };
               for (auto& use : unionOp->getUses()) {
                  removeFn(use);
               }
               unionOp->erase();
            }
         }
      }

      mlir::subop::ColumnUsageAnalysis usedColumns(getOperation());
      mlir::subop::ColumnCreationAnalysis createdColumns(getOperation());
      std::vector<mlir::Operation*> opsToErase;
      getOperation()->walk([&](mlir::subop::NestedMapOp nestedMapOp) {
         std::unordered_map<mlir::tuples::Column*, size_t> colArgs;
         std::vector<mlir::Attribute> newParams;
         size_t argId = 1;
         for (auto colRefAttr : nestedMapOp.getParameters()) {
            colArgs.insert({&colRefAttr.cast<mlir::tuples::ColumnRefAttr>().getColumn(), argId++});
            newParams.push_back(colRefAttr);
         }
         auto tuple = nestedMapOp.getBody()->getArgument(0);
         for (auto user : tuple.getUsers()) {
            if (auto combineOp = mlir::dyn_cast<mlir::subop::CombineTupleOp>(user)) {
               if (combineOp->getBlock() != nestedMapOp.getBody()) {
                  nestedMapOp.emitError("NestedMapOp: tuple parameter must only be used in the body block");
                  return signalPassFailure();
               }
               std::unordered_set<mlir::tuples::Column*> availableColumns;
               std::function<void(mlir::Value)> computeAvailableColumns = [&](mlir::Value v) {
                  if (auto defOp = v.getDefiningOp()) {
                     auto createdOps = createdColumns.getCreatedColumns(defOp);

                     availableColumns.insert(createdOps.begin(), createdOps.end());
                     for (auto operand : defOp->getOperands()) {
                        if (operand.getType().isa<mlir::tuples::TupleStreamType>()) {
                           computeAvailableColumns(operand);
                        }
                     }
                  }
               };
               std::unordered_set<mlir::tuples::Column*> requiredColumns;
               std::function<void(mlir::Operation*, std::unordered_set<mlir::tuples::Column*>)> addRequiredColumns = [&](mlir::Operation* op, std::unordered_set<mlir::tuples::Column*> availableColumns) {
                  auto created = createdColumns.getCreatedColumns(op);
                  availableColumns.insert(created.begin(), created.end());
                  for (auto usedColumn : usedColumns.getUsedColumns(op)) {
                     if (!availableColumns.contains(usedColumn)) {
                        requiredColumns.insert(usedColumn);
                     }
                  }
                  for (auto* user : op->getUsers()) {
                     addRequiredColumns(user, availableColumns);
                  }
               };
               computeAvailableColumns(combineOp);
               addRequiredColumns(combineOp,availableColumns);
               std::vector<mlir::Attribute> createdByMap;
               mlir::OpBuilder b(combineOp);
               mlir::Block* mapBlock = new mlir::Block;
               {
                  mlir::OpBuilder::InsertionGuard guard(b);
                  b.setInsertionPointToStart(mapBlock);
                  auto& colManager = getContext().getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
                  std::vector<mlir::Value> mapOpReturnVals;
                  for (auto r : requiredColumns) {
                     auto colRefAttr = colManager.createRef(r);
                     auto colDefAttr = colManager.createDef(r);

                     if (!colArgs.contains(r)) {
                        nestedMapOp.getBody()->addArgument(r->type, b.getUnknownLoc());
                        colArgs.insert({r, argId++});
                        newParams.push_back(colRefAttr);
                     }
                     createdByMap.push_back(colDefAttr);
                     mapOpReturnVals.push_back(nestedMapOp.getBody()->getArgument(colArgs[r]));
                  }
                  b.create<mlir::tuples::ReturnOp>(b.getUnknownLoc(), mapOpReturnVals);
               }
               auto mapOp = b.create<mlir::subop::MapOp>(b.getUnknownLoc(), mlir::tuples::TupleStreamType::get(b.getContext()), combineOp.getStream(), b.getArrayAttr(createdByMap));
               mapOp.getFn().push_back(mapBlock);
               combineOp->replaceAllUsesWith(mlir::ValueRange{mapOp.getResult()});
               opsToErase.push_back(combineOp);

            } else {
               nestedMapOp.emitError("NestedMapOp: tuple parameter must only be used with CombineTupleOp");
               return signalPassFailure();
            }
         }
         nestedMapOp.setParametersAttr(mlir::ArrayAttr::get(&getContext(), newParams));

      });
      for (auto op : opsToErase) {
         op->erase();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createInlineNestedMapPass() { return std::make_unique<InlineNestedMapPass>(); }