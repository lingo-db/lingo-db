#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SubOperator/SubOperatorInterfaces.h"
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
namespace {

class ParallelizePass : public mlir::PassWrapper<ParallelizePass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelizePass)
   virtual llvm::StringRef getArgument() const override { return "subop-parallelize"; }
   void collectPipelineOperations(std::vector<mlir::Operation*>& ops, mlir::Operation* op, bool start = false) {
      if (op->getDialect()->getNamespace() == "subop" && !start) {
         ops.push_back(op);
      }
      for (auto res : op->getResults()) {
         if (res.getType().isa<mlir::tuples::TupleStreamType>()) {
            for (auto* user : res.getUsers()) {
               collectPipelineOperations(ops, user);
            }
         }
      }
      op->walk([&](mlir::Operation* nested) {
         if (nested != op) {
            auto isStreamType = [](mlir::Type t) { return t.isa<mlir::tuples::TupleStreamType>(); };
            if (llvm::none_of(nested->getOperandTypes(), isStreamType) && llvm::any_of(nested->getResultTypes(), isStreamType)) {
               collectPipelineOperations(ops, nested);
            }
         }
      });
   }
   struct ToThreadLocalInfo {
      mlir::Region* combineRegion = nullptr;
      mlir::Region* compareRegion = nullptr;
      std::vector<mlir::Operation*> shouldUse = {};
   };
   void runOnOperation() override {
      auto columnCreationAnalysis = getAnalysis<mlir::subop::ColumnCreationAnalysis>();
      std::unordered_map<mlir::Operation*, ToThreadLocalInfo> toThreadLocalsGlobal;
      std::unordered_set<mlir::Operation*> markAsAtomic;
      std::unordered_set<std::string> nestedMembers;

      getOperation()->walk([&](mlir::Operation* op) {
         if (!mlir::isa<mlir::subop::GetLocal>(op) && !mlir::isa_and_nonnull<mlir::func::FuncOp>(op->getParentOp())) {
            if (op->getNumResults() == 1) {
               if (auto stateType = op->getResultTypes()[0].dyn_cast_or_null<mlir::subop::State>()) {
                  for (auto m : stateType.getMembers().getNames()) {
                     nestedMembers.insert(m.cast<mlir::StringAttr>().str());
                  }
               }
            }
         }
      });

      //getOperation().dump();
      getOperation()->walk([&](mlir::Operation* op) {
         if (auto scanRefsOp = mlir::dyn_cast_or_null<mlir::subop::ScanRefsOp>(op)) {
            if (auto funcOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(scanRefsOp->getParentOp())) {
               std::unordered_map<mlir::Operation*, ToThreadLocalInfo> toThreadLocals;
               std::vector<mlir::Operation*> pipelineOps;
               collectPipelineOperations(pipelineOps, op, true);
               bool canBeParallel = true;
               auto isNested = [&](mlir::Value v) {
                  if (auto* def = v.getDefiningOp()) {
                     if (def->getParentOp() != funcOp.getOperation() && funcOp->isAncestor(def)) {
                        return true;
                     }
                  }
                  return false;
               };
               auto isStreamTypeOrNested = [&](mlir::Value v) {
                  if (v.getType().isa<mlir::tuples::TupleStreamType>()) return true;
                  if (isNested(v)) return true;
                  return false;
               };

               std::unordered_map<std::string, std::vector<mlir::Operation*>> readMembers;
               std::unordered_map<std::string, std::vector<mlir::Operation*>> writtenMembers;
               for (auto* pipelineOp : pipelineOps) {
                  auto getCollisions = [&]() {
                     std::unordered_set<mlir::Operation*> collisions;

                     if (auto subOp = mlir::dyn_cast<mlir::subop::SubOperator>(pipelineOp)) {
                        auto currentReadMembers = subOp.getReadMembers();
                        auto currentWrittenMembers = subOp.getWrittenMembers();

                        for (auto r : currentReadMembers) {
                           if (writtenMembers.contains(r)) {
                              collisions.insert(writtenMembers[r].begin(), writtenMembers[r].end());
                           }
                        }
                        for (auto w : currentWrittenMembers) {
                           if (readMembers.contains(w)) {
                              collisions.insert(readMembers[w].begin(), readMembers[w].end());
                           }
                        }
                        for (auto r : currentReadMembers) {
                           if (!nestedMembers.contains(r)) {
                              readMembers[r].push_back(subOp);
                           }
                        }
                        for (auto r : currentWrittenMembers) {
                           if (!nestedMembers.contains(r)) {
                              writtenMembers[r].push_back(subOp);
                           }
                        }
                     }
                     return collisions;
                  };

                  if (auto materializeOp = mlir::dyn_cast_or_null<mlir::subop::MaterializeOp>(pipelineOp)) {
                     if (getCollisions().empty() && !isNested(materializeOp.getState())) {
                        if (auto* createOp = materializeOp.getState().getDefiningOp()) {
                           toThreadLocals[createOp].shouldUse.push_back(materializeOp);
                           continue;
                        }
                     }
                  } else if (mlir::isa<mlir::subop::LookupOp>(pipelineOp)) {
                     if (getCollisions().empty()) {
                        //ignore lookupOps
                        continue;
                     }
                  } else if (auto lookupOrInsertOp = mlir::dyn_cast_or_null<mlir::subop::LookupOrInsertOp>(pipelineOp)) {
                     if (getCollisions().empty()) {
                        if (auto* createOp = lookupOrInsertOp.getState().getDefiningOp()) {
                           toThreadLocals[createOp].compareRegion = &lookupOrInsertOp.getEqFn();
                           toThreadLocals[createOp].shouldUse.push_back(lookupOrInsertOp);
                           continue;
                        }
                     }
                  } else if (auto gatherOp = mlir::dyn_cast<mlir::subop::GatherOp>(pipelineOp)) {
                     if (getCollisions().empty()) {
                        //ignore gathers
                        continue;
                     }
                  } else if (mlir::isa<mlir::subop::ScanListOp, mlir::subop::CombineTupleOp>(pipelineOp)) {
                     continue;
                  } else if (auto reduceOp = mlir::dyn_cast<mlir::subop::ReduceOp>(pipelineOp)) {
                     //todo: order of members in reduce fn must match member order in state
                     auto collisions = getCollisions();
                     if (collisions.size() == 1 && !reduceOp.getCombine().empty()) {
                        if (auto lookupOp = mlir::dyn_cast<mlir::subop::LookupOp>(*collisions.begin())) {
                           if (!isNested(lookupOp.getState())) {
                              if (auto* createOp = lookupOp.getState().getDefiningOp()) {
                                 if (auto simpleStateType = createOp->getResultTypes()[0].dyn_cast_or_null<mlir::subop::SimpleStateType>()) {
                                    if (simpleStateType.getMembers().getTypes().size() == reduceOp.getMembers().size()) {
                                       toThreadLocals[createOp].shouldUse.push_back(lookupOp);
                                       toThreadLocals[createOp].combineRegion = &reduceOp.getCombine();
                                       continue;
                                    }
                                 }
                              }
                           }
                        }
                        if (auto lookupOrInsertOp = mlir::dyn_cast<mlir::subop::LookupOrInsertOp>(*collisions.begin())) {
                           if (!isNested(lookupOrInsertOp.getState())) {
                              if (auto* createOp = lookupOrInsertOp.getState().getDefiningOp()) {
                                 if (auto hashMapType = createOp->getResultTypes()[0].dyn_cast_or_null<mlir::subop::HashMapType>()) {
                                    if (hashMapType.getValueMembers().getTypes().size() == reduceOp.getMembers().size()) {
                                       toThreadLocals[createOp].combineRegion = &reduceOp.getCombine();
                                       continue;
                                    }
                                 }
                              }
                           }
                        }
                     }

                  } else if (auto scatterOp = mlir::dyn_cast_or_null<mlir::subop::ScatterOp>(pipelineOp)) {
                     auto collisions = getCollisions();
                     if (collisions.size() == 0) {
                        continue;
                     }
                     if (collisions.size() == 1) {
                        if (auto lookupOp = mlir::dyn_cast<mlir::subop::LookupOp>(*collisions.begin())) {
                           markAsAtomic.insert(pipelineOp);
                           continue;
                        }
                     }
                  } else if (llvm::all_of(pipelineOp->getOperands(), isStreamTypeOrNested)) {
                     //ignore: do not interact with states
                     continue;
                  }
                  //llvm::dbgs() << "problematic operation:";
                  //pipelineOp->dump();
                  canBeParallel = false;
               }
               //finally: mark as parallel
               if (canBeParallel) {
                  scanRefsOp->setAttr("parallel", mlir::UnitAttr::get(&getContext()));
                  //llvm::dbgs()<<"parallel: ";
                  //scanRefsOp.dump();
                  for (auto l : toThreadLocals) {
                     if (toThreadLocalsGlobal.contains(l.first)) {
                        //todo: make sure that everything fits together
                        if (l.second.combineRegion) {
                           toThreadLocalsGlobal[l.first].combineRegion = l.second.combineRegion;
                        }
                        if (l.second.compareRegion) {
                           toThreadLocalsGlobal[l.first].compareRegion = l.second.compareRegion;
                        }
                        toThreadLocalsGlobal[l.first].shouldUse.insert(toThreadLocalsGlobal[l.first].shouldUse.end(), l.second.shouldUse.begin(), l.second.shouldUse.end());
                     } else {
                        toThreadLocalsGlobal.insert(l);
                     }
                  }
                  for (auto* mA : markAsAtomic) {
                     mA->setAttr("atomic", mlir::UnitAttr::get(&getContext()));
                  }
               } else {
                  //llvm::dbgs()<<"not parallel: ";
                  //scanRefsOp.dump();
               }
            }
         }
      });
      for (auto toThreadLocal : toThreadLocalsGlobal) {
         auto* createOp = toThreadLocal.first;
         mlir::OpBuilder builder(&getContext());
         builder.setInsertionPoint(createOp);
         auto threadLocalType = mlir::subop::ThreadLocalType::get(builder.getContext(), createOp->getResultTypes()[0]);
         auto createThreadLocal = builder.create<mlir::subop::CreateThreadLocalOp>(createOp->getLoc(), threadLocalType);
         auto* block = new mlir::Block;
         createThreadLocal.getInitFn().push_back(block);
         builder.setInsertionPointToStart(block);
         auto* clonedCreate = builder.clone(*createOp);
         clonedCreate->setAttr("allocateOnHeap", builder.getUnitAttr());
         builder.create<mlir::tuples::ReturnOp>(createOp->getLoc(), clonedCreate->getResult(0));
         std::unordered_set<mlir::Operation*> localUsers(toThreadLocal.second.shouldUse.begin(), toThreadLocal.second.shouldUse.end());
         std::vector<mlir::Operation*> otherUsers;
         for (auto* user : createOp->getUsers()) {
            if (!localUsers.contains(user)) {
               otherUsers.push_back(user); //todo fix behavior for nested
            }
         }
         auto getTopLevelOp = [](mlir::Operation* op) {
            while (!mlir::isa<mlir::func::FuncOp>(op->getParentOp())) {
               op = op->getParentOp();
            }
            return op;
         };
         std::sort(otherUsers.begin(), otherUsers.end(), [&](mlir::Operation* left, mlir::Operation* right) {
            return getTopLevelOp(left)->isBeforeInBlock(getTopLevelOp(right));
         });
         assert(otherUsers.size() > 0);
         builder.setInsertionPoint(getTopLevelOp(otherUsers[0]));
         auto mergeOp = builder.create<mlir::subop::MergeOp>(createOp->getLoc(), createOp->getResultTypes()[0], createThreadLocal.getResult());
         if (toThreadLocal.second.combineRegion) {
            mlir::IRMapping mapping;
            toThreadLocal.second.combineRegion->cloneInto(&mergeOp.getCombineFn(), mapping);
         }
         if (toThreadLocal.second.compareRegion) {
            mlir::IRMapping mapping;
            toThreadLocal.second.compareRegion->cloneInto(&mergeOp.getEqFn(), mapping);
         }
         mlir::Value merged = mergeOp.getRes();
         for (auto* localUser : localUsers) {
            builder.setInsertionPoint(localUser);
            mlir::Value local = builder.create<mlir::subop::GetLocal>(createOp->getLoc(), createOp->getResultTypes()[0], createThreadLocal.getResult());
            localUser->replaceUsesOfWith(createOp->getResult(0), local);
         }
         createOp->replaceAllUsesWith(mlir::ValueRange{merged});
         createOp->erase();
      }
      //getOperation()->dump();
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createParallelizePass() { return std::make_unique<ParallelizePass>(); }