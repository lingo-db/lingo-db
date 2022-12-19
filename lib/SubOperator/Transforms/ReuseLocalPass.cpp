#include "llvm/Support/Debug.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_set>
namespace {
static std::vector<mlir::subop::SubOperator> findWritingOps(mlir::Block& block, std::vector<std::string>& members) {
   std::unordered_set<std::string> memberSet(members.begin(), members.end());
   std::vector<mlir::subop::SubOperator> res;
   block.walk([&memberSet, &res](mlir::subop::SubOperator subop) {
      auto writtenBySubOp = subop.getWrittenMembers();
      for (auto writtenMember : writtenBySubOp) {
         if (memberSet.contains(writtenMember)) {
            res.push_back(subop);
            break;
         }
      }
   });
   return res;
}

static std::optional<std::string> lookupByValue(mlir::DictionaryAttr mapping, mlir::Attribute value) {
   for (auto m : mapping) {
      if (m.getValue() == value) {
         return m.getName().str();
      }
   }
   return {};
}

class AvoidUnnecessaryMaterialization : public mlir::RewritePattern {
   public:
   AvoidUnnecessaryMaterialization(mlir::MLIRContext* context)
      : RewritePattern(mlir::subop::ScanOp::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto scanOp = mlir::cast<mlir::subop::ScanOp>(op);
      if (!scanOp.getState().getType().isa<mlir::subop::BufferType>()) {
         return mlir::failure();
      }
      auto readMembers = scanOp.getReadMembers();
      auto* stateCreateOp = scanOp.getState().getDefiningOp();
      if (!stateCreateOp) {
         return mlir::failure();
      }
      auto writingOps = findWritingOps(*stateCreateOp->getBlock(), readMembers);
      if (writingOps.size() != 1) {
         return mlir::failure();
      }
      if (auto materializeOp = mlir::dyn_cast_or_null<mlir::subop::MaterializeOp>(writingOps[0].getOperation())) {
         if (materializeOp->getBlock() == scanOp->getBlock() && materializeOp->isBeforeInBlock(scanOp)) {
            if (auto scanOp2 = mlir::dyn_cast_or_null<mlir::subop::ScanOp>(materializeOp.getStream().getDefiningOp())) {
               std::vector<mlir::NamedAttribute> newMapping;
               for (auto curr : scanOp.getMapping()) {
                  auto currentMember = curr.getName();
                  auto otherColumnDef = colManager.createDef(&materializeOp.getMapping().get(currentMember).cast<mlir::tuples::ColumnRefAttr>().getColumn());
                  auto otherMember = lookupByValue(scanOp2.getMapping(), otherColumnDef);
                  newMapping.push_back(rewriter.getNamedAttr(otherMember.value(), curr.getValue()));
               }
               rewriter.updateRootInPlace(op, [&] {
                  scanOp.setOperand(scanOp2.getState());
                  scanOp.setMappingAttr(rewriter.getDictionaryAttr(newMapping));
               });
               return mlir::success();
            }
         }
      }
      return mlir::failure();
   }
};

class AvoidDeadMaterialization : public mlir::RewritePattern {
   public:
   AvoidDeadMaterialization(mlir::MLIRContext* context)
      : RewritePattern(mlir::subop::MaterializeOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto materializeOp = mlir::cast<mlir::subop::MaterializeOp>(op);
      for (auto* user : materializeOp.getState().getUsers()) {
         if (user != op) {
            //other user of state => "probably" still useful
            return mlir::failure();
         }
      }
      rewriter.eraseOp(op);
      return mlir::success();
   }
};
class ReuseHashtable : public mlir::RewritePattern {
   public:
   ReuseHashtable(mlir::MLIRContext* context)
      : RewritePattern(mlir::subop::CreateHashIndexedView::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto createHashIndexedView = mlir::cast<mlir::subop::CreateHashIndexedView>(op);
      auto loc = createHashIndexedView.getLoc();
      auto source = createHashIndexedView.getSource();
      std::vector<mlir::Operation*> users(source.getUsers().begin(), source.getUsers().end());
      if (users.size() != 2) {
         return mlir::failure();
      }
      auto* otherUser = users[0];
      auto* createView = users[1];
      if (!mlir::isa<mlir::subop::MaterializeOp>(otherUser)) {
         std::swap(otherUser, createView);
      }
      if (auto materializeOp = mlir::dyn_cast_or_null<mlir::subop::MaterializeOp>(otherUser)) {
         if (auto mapOp = mlir::dyn_cast_or_null<mlir::subop::MapOp>(materializeOp.getStream().getDefiningOp())) {
            if (auto scanOp = mlir::dyn_cast_or_null<mlir::subop::ScanOp>(mapOp.getStream().getDefiningOp())) {
               if (auto htType = scanOp.getState().getType().dyn_cast_or_null<mlir::subop::HashMapType>()) {
                  auto hashMember = createHashIndexedView.getHashMember();
                  auto linkMember = createHashIndexedView.getLinkMember();
                  auto mapReturnVals = mlir::cast<mlir::tuples::ReturnOp>(mapOp.getFn().front().getTerminator()).getResults();
                  auto* hashColumn = &materializeOp.getMapping().get(hashMember).cast<mlir::tuples::ColumnRefAttr>().getColumn();
                  auto* linkColumn = &materializeOp.getMapping().get(linkMember).cast<mlir::tuples::ColumnRefAttr>().getColumn();
                  std::vector<mlir::tuples::Column*> hashedColumns;
                  for (auto z : llvm::zip(mapOp.getComputedCols(), mapReturnVals)) {
                     auto* defCol = &std::get<0>(z).cast<mlir::tuples::ColumnDefAttr>().getColumn();
                     if (defCol == hashColumn) {
                        if (auto hashOp = mlir::dyn_cast_or_null<mlir::db::Hash>(std::get<1>(z).getDefiningOp())) {
                           if (auto packOp = mlir::dyn_cast_or_null<mlir::util::PackOp>(hashOp.getVal().getDefiningOp())) {
                              for (auto v : packOp.getVals()) {
                                 if (auto getCol = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(v.getDefiningOp())) {
                                    hashedColumns.push_back(&getCol.getAttr().getColumn());
                                 } else {
                                    return mlir::failure();
                                 }
                              }
                           }
                        }
                     } else if (defCol == linkColumn) {
                        if (!mlir::isa<mlir::util::InvalidRefOp>(std::get<1>(z).getDefiningOp())) {
                           return mlir::failure();
                        }
                     }
                  }
                  std::unordered_map<std::string, std::string> memberMapping;
                  for (auto m : materializeOp.getMapping()) {
                     auto colDef = colManager.createDef(&m.getValue().cast<mlir::tuples::ColumnRefAttr>().getColumn());
                     auto hmMember = lookupByValue(scanOp.getMapping(), colDef);
                     auto bufferMember = m.getName().str();
                     if (hmMember) {
                        memberMapping[bufferMember] = hmMember.value();
                     }
                  }
                  std::unordered_set<mlir::tuples::Column*> hashMapKey;
                  std::unordered_map<std::string, mlir::tuples::Column*> keyMemberToColumn;
                  for (auto keyMember : htType.getKeyMembers().getNames()) {
                     auto colDef = scanOp.getMapping().get(keyMember.cast<mlir::StringAttr>().str()).cast<mlir::tuples::ColumnDefAttr>();
                     hashMapKey.insert(&colDef.getColumn());
                     keyMemberToColumn[keyMember.cast<mlir::StringAttr>().str()] = &colDef.getColumn();
                  }
                  if (hashMapKey.size() != hashedColumns.size()) {
                     return mlir::failure();
                  }
                  for (auto* c : hashedColumns) {
                     if (!hashMapKey.contains(c)) {
                        return mlir::failure();
                     }
                  }
                  if (!createHashIndexedView->hasOneUse()) {
                     return mlir::failure();
                  }
                  auto* onlyUser = *createHashIndexedView->getUsers().begin();
                  if (auto lookupOp = mlir::dyn_cast_or_null<mlir::subop::LookupOp>(onlyUser)) {
                     if (auto map2Op = mlir::dyn_cast_or_null<mlir::subop::MapOp>(lookupOp.getStream().getDefiningOp())) {
                        if (!lookupOp->hasOneUse()) return mlir::failure();
                        if (auto nestedMapOp = mlir::dyn_cast_or_null<mlir::subop::NestedMapOp>(*lookupOp->getUsers().begin())) {
                           if (map2Op.getComputedCols().size() != 1) return mlir::failure();
                           if (nestedMapOp.getParameters().size() != 1) return mlir::failure();
                           if (&nestedMapOp.getParameters()[0].cast<mlir::tuples::ColumnRefAttr>().getColumn() != &lookupOp.getRef().getColumn()) return mlir::failure();
                           if (!nestedMapOp.getRegion().getArgument(1).hasOneUse()) return mlir::failure();
                           auto* listUser = *nestedMapOp.getRegion().getArgument(1).getUsers().begin();
                           if (auto scanListOp = mlir::dyn_cast_or_null<mlir::subop::ScanListOp>(listUser)) {
                              if (!scanListOp->hasOneUse()) return mlir::failure();
                              if (auto gatherOp = mlir::dyn_cast_or_null<mlir::subop::GatherOp>(*scanListOp->getUsers().begin())) {
                                 auto returnVals = mlir::cast<mlir::tuples::ReturnOp>(map2Op.getFn().front().getTerminator()).getResults();
                                 if (returnVals.size() != 1) return mlir::failure();
                                 std::vector<mlir::tuples::Column*> lookupHashedColumns;
                                 if (auto hashOp = mlir::dyn_cast_or_null<mlir::db::Hash>(returnVals[0].getDefiningOp())) {
                                    if (auto packOp = mlir::dyn_cast_or_null<mlir::util::PackOp>(hashOp.getVal().getDefiningOp())) {
                                       for (auto v : packOp.getVals()) {
                                          if (auto getCol = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(v.getDefiningOp())) {
                                             lookupHashedColumns.push_back(&getCol.getAttr().getColumn());
                                          } else {
                                             return mlir::failure();
                                          }
                                       }
                                    }
                                 }
                                 if (lookupHashedColumns.size() != hashedColumns.size()) return mlir::failure();
                                 std::unordered_map<mlir::tuples::Column*, mlir::tuples::Column*> columnMapping;
                                 for (auto z : llvm::zip(lookupHashedColumns, hashedColumns)) {
                                    columnMapping.insert({std::get<1>(z), std::get<0>(z)});
                                 }
                                 std::vector<mlir::Attribute> lookupColumns;
                                 std::unordered_map<std::string, mlir::tuples::Column*> keyColumnMapping;
                                 for (auto keyMember : htType.getKeyMembers().getNames()) {
                                    auto *col = columnMapping.at(keyMemberToColumn.at(keyMember.cast<mlir::StringAttr>().str()));
                                    lookupColumns.push_back(colManager.createRef(col));
                                    keyColumnMapping[keyMember.cast<mlir::StringAttr>().str()] = col;
                                 }
                                 mlir::Block* equalityBlock = new mlir::Block;
                                 std::vector<mlir::Value> leftArgs;
                                 std::vector<mlir::Value> rightArgs;
                                 for (auto t : htType.getKeyMembers().getTypes()) {
                                    leftArgs.push_back(equalityBlock->addArgument(t.cast<mlir::TypeAttr>().getValue(), loc));
                                 }
                                 for (auto t : htType.getKeyMembers().getTypes()) {
                                    rightArgs.push_back(equalityBlock->addArgument(t.cast<mlir::TypeAttr>().getValue(), loc));
                                 }
                                 {
                                    mlir::OpBuilder::InsertionGuard guard(rewriter);
                                    rewriter.setInsertionPointToStart(equalityBlock);
                                    std::vector<mlir::Value> comparisons;
                                    for (auto z : llvm::zip(leftArgs, rightArgs)) {
                                       comparisons.push_back(rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::isa, std::get<0>(z), std::get<1>(z)));
                                    }
                                    mlir::Value anded = rewriter.create<mlir::db::AndOp>(loc, comparisons);
                                    rewriter.create<mlir::tuples::ReturnOp>(loc, anded);
                                 }
                                 rewriter.setInsertionPointAfter(lookupOp);
                                 auto hmRefType = mlir::subop::LookupEntryRefType::get(getContext(), htType);
                                 auto hmRefListType = mlir::subop::ListType::get(getContext(), hmRefType);
                                 lookupOp.getRef().getColumn().type = hmRefListType;
                                 nestedMapOp.getRegion().getArgument(1).setType(hmRefListType);
                                 scanListOp.getElem().getColumn().type = hmRefType;
                                 auto newLookupOp = rewriter.replaceOpWithNewOp<mlir::subop::LookupOp>(lookupOp, lookupOp.getStream(), scanOp.getState(), rewriter.getArrayAttr(lookupColumns), lookupOp.getRef());
                                 newLookupOp.getEqFn().push_back(equalityBlock);
                                 rewriter.eraseOp(createHashIndexedView);
                                 std::vector<mlir::NamedAttribute> newGatherMapping;
                                 std::vector<mlir::Attribute> renamedColumns;
                                 for (auto m : gatherOp.getMapping()) {
                                    if (keyColumnMapping.contains(memberMapping.at(m.getName().str()))) {
                                       //renaming
                                       auto colDef = m.getValue().cast<mlir::tuples::ColumnDefAttr>();
                                       auto *c = keyColumnMapping.at(memberMapping.at(m.getName().str()));
                                       renamedColumns.push_back(mlir::tuples::ColumnDefAttr::get(getContext(), colDef.getName(), colDef.getColumnPtr(), rewriter.getArrayAttr({colManager.createRef(c)})));
                                    } else {
                                       newGatherMapping.push_back(rewriter.getNamedAttr(memberMapping.at(m.getName().str()), m.getValue()));
                                    }
                                 }
                                 gatherOp.setMappingAttr(rewriter.getDictionaryAttr(newGatherMapping));
                                 if (!renamedColumns.empty()) {
                                    rewriter.setInsertionPointAfter(gatherOp);
                                    mlir::Value combined=rewriter.create<mlir::subop::CombineTupleOp>(loc,gatherOp,nestedMapOp.getRegion().getArgument(0));
                                    mlir::Value renamed = rewriter.create<mlir::subop::RenamingOp>(loc, combined, rewriter.getArrayAttr(renamedColumns));
                                    gatherOp.getRes().replaceAllUsesExcept(renamed, combined.getDefiningOp());
                                 }
                                 return mlir::success();
                              }
                           }
                        }
                     }
                  }
               }
            }
         }
      }
      return mlir::failure();
   }
};
class ReuseLocalPass : public mlir::PassWrapper<ReuseLocalPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseLocalPass)
   virtual llvm::StringRef getArgument() const override { return "subop-reuse-local"; }

   void runOnOperation() override {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<AvoidUnnecessaryMaterialization>(&getContext());
      patterns.insert<AvoidDeadMaterialization>(&getContext());
      patterns.insert<ReuseHashtable>(&getContext());
      if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::subop::createReuseLocalPass() { return std::make_unique<ReuseLocalPass>(); }