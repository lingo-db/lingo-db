#include "llvm/ADT/TypeSwitch.h"
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
static std::pair<mlir::tuples::ColumnDefAttr, mlir::tuples::ColumnRefAttr> createColumn(mlir::Type type, std::string scope, std::string name) {
   auto& columnManager = type.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope(scope);
   std::string attributeName = name;
   mlir::tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = type;
   return {markAttrDef, columnManager.createRef(&ra)};
}

class AvoidUnnecessaryMaterialization : public mlir::RewritePattern {
   public:
   AvoidUnnecessaryMaterialization(mlir::MLIRContext* context)
      : RewritePattern(mlir::subop::ScanOp::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto scanOp = mlir::cast<mlir::subop::ScanOp>(op);
      if (!mlir::isa<mlir::subop::BufferType>(scanOp.getState().getType())) {
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
                  auto otherColumnDef = colManager.createDef(&mlir::cast<mlir::tuples::ColumnRefAttr>(materializeOp.getMapping().get(currentMember)).getColumn());
                  auto otherMember = lookupByValue(scanOp2.getMapping(), otherColumnDef);
                  newMapping.push_back(rewriter.getNamedAttr(otherMember.value(), curr.getValue()));
               }
               rewriter.modifyOpInPlace(op, [&] {
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

class AvoidArrayMaterialization : public mlir::RewritePattern {
   mlir::subop::ColumnUsageAnalysis& analysis;
   mlir::Operation* getRoot(mlir::Operation* op) const {
      if (op->getNumOperands() < 1) {
         return op;
      }
      bool isFirstTupleStream = mlir::isa<mlir::tuples::TupleStreamType>(op->getOperand(0).getType());
      bool noOtherTupleStream = llvm::none_of(op->getOperands().drop_front(), [](mlir::Value v) { return mlir::isa<mlir::tuples::TupleStreamType>(v.getType()); });
      if (isFirstTupleStream && noOtherTupleStream) {
         return getRoot(op->getOperand(0).getDefiningOp());
      } else if (isFirstTupleStream) {
         return nullptr;
      } else {
         return op;
      }
   }

   public:
   AvoidArrayMaterialization(mlir::MLIRContext* context, mlir::subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(mlir::subop::ScanRefsOp::getOperationName(), 1, context), analysis(analysis) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto scanRefsOp = mlir::cast<mlir::subop::ScanRefsOp>(op);
      if (!mlir::isa<mlir::subop::ArrayType>(scanRefsOp.getState().getType())) {
         return mlir::failure();
      }
      if (scanRefsOp->hasAttr("sequential")) {
         return mlir::failure();
      }
      auto state = scanRefsOp.getState();
      std::vector<mlir::subop::GetBeginReferenceOp> otherUsers;
      for (auto* user : state.getUsers()) {
         if (user == op) {
            continue;
         } else if (auto getBeginOp = mlir::dyn_cast_or_null<mlir::subop::GetBeginReferenceOp>(user)) {
            otherUsers.push_back(getBeginOp);
         } else {
            return mlir::failure();
         }
      }
      if (!otherUsers.size()) {
         return mlir::failure();
      }
      mlir::subop::ScatterOp scatterOp;
      mlir::subop::OffsetReferenceBy offsetByOp;
      std::vector<mlir::subop::EntriesBetweenOp> needsChanges;
      mlir::tuples::ColumnRefAttr replaceWith;
      mlir::tuples::ColumnRefAttr replaceIdxWith;
      for (auto user : otherUsers) {
         auto colUsers = analysis.findOperationsUsing(&user.getRef().getColumn());
         for (auto* colUser : colUsers) {
            if (auto* root = getRoot(colUser)) {
               if (root == op) {
                  if (auto entriesBetween = mlir::dyn_cast_or_null<mlir::subop::EntriesBetweenOp>(colUser)) {
                     needsChanges.push_back(entriesBetween);
                  } else {
                     return mlir::failure();
                  }
               } else {
                  if (auto otherScanRefsOp = mlir::dyn_cast_or_null<mlir::subop::ScanRefsOp>(root)) {
                     auto otherState = otherScanRefsOp.getState();
                     if (!mlir::isa<mlir::subop::ArrayType>(otherState.getType())) {
                        return mlir::failure();
                     }
                     if (auto offsetBy = mlir::dyn_cast_or_null<mlir::subop::OffsetReferenceBy>(colUser)) {
                        bool idxMatches = false;
                        for (auto* otherScanRefsUser : analysis.findOperationsUsing(&otherScanRefsOp.getRef().getColumn())) {
                           if (auto entriesBetweenOp = mlir::dyn_cast_or_null<mlir::subop::EntriesBetweenOp>(otherScanRefsUser)) {
                              if (&offsetBy.getIdx().getColumn() == &entriesBetweenOp.getBetween().getColumn()) {
                                 idxMatches = true;
                                 replaceIdxWith = colManager.createRef(&entriesBetweenOp.getBetween().getColumn());
                              }
                           }
                        }
                        if (!idxMatches) {
                           return mlir::failure();
                        }
                        auto offsetByUsers = analysis.findOperationsUsing(&offsetBy.getNewRef().getColumn());
                        if (offsetByUsers.size() != 1) {
                           return mlir::failure();
                        }
                        if (auto localScatterOp = mlir::dyn_cast_or_null<mlir::subop::ScatterOp>(*offsetByUsers.begin())) {
                           if (scatterOp) {
                              return mlir::failure();
                           }
                           scatterOp = localScatterOp;
                           offsetByOp = offsetBy;
                           replaceWith = colManager.createRef(&otherScanRefsOp.getRef().getColumn());
                           if (scatterOp.getWrittenMembers().size() != mlir::cast<mlir::subop::State>(scanRefsOp.getState().getType()).getMembers().getNames().size()) {
                              return mlir::failure();
                           }
                        } else {
                           return mlir::failure();
                        }
                     }
                  }
               }
            } else {
               return mlir::failure();
            }
         }
      }
      auto scanRefUsers = analysis.findOperationsUsing(&scanRefsOp.getRef().getColumn());
      std::vector<mlir::subop::GatherOp> gatherOps;
      for (auto* scanRefUser : scanRefUsers) {
         if (auto gatherOp = mlir::dyn_cast_or_null<mlir::subop::GatherOp>(scanRefUser)) {
            gatherOps.push_back(gatherOp);
         } else if (mlir::isa<mlir::subop::EntriesBetweenOp>(scanRefUser)) {
         } else {
            return mlir::failure();
         }
      }
      if (scatterOp) {
         std::unordered_map<std::string, mlir::tuples::ColumnRefAttr> scatterMapping;
         for (auto m : scatterOp.getMapping()) {
            scatterMapping.insert({m.getName().str(), mlir::cast<mlir::tuples::ColumnRefAttr>(m.getValue())});
         }
         std::vector<mlir::Attribute> renamed;
         for (auto gatherOp : gatherOps) {
            for (auto m : gatherOp.getMapping()) {
               renamed.push_back(colManager.createDef(&mlir::cast<mlir::tuples::ColumnDefAttr>(m.getValue()).getColumn(), rewriter.getArrayAttr(scatterMapping.at(m.getName().str()))));
            }
            rewriter.replaceOp(gatherOp, gatherOp.getStream());
         }
         for (auto betweenOp : needsChanges) {
            renamed.push_back(colManager.createDef(&betweenOp.getBetween().getColumn(), rewriter.getArrayAttr({replaceIdxWith})));
            rewriter.replaceOp(betweenOp, betweenOp.getStream());
         }
         for (auto getBeginOp : otherUsers) {
            rewriter.replaceOp(getBeginOp, getBeginOp.getStream());
         }
         rewriter.replaceOpWithNewOp<mlir::subop::RenamingOp>(op, scatterOp.getStream(), rewriter.getArrayAttr(renamed));
         rewriter.replaceOp(offsetByOp, offsetByOp.getStream());
         rewriter.eraseOp(scatterOp);
         //todo: replace scan_refs op with
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
      if (!materializeOp.getState().getDefiningOp()) {
         return mlir::failure();
      }
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
   const mlir::subop::ColumnUsageAnalysis& analysis;

   public:
   ReuseHashtable(mlir::MLIRContext* context, mlir::subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(mlir::subop::InsertOp::getOperationName(), 1, context), analysis(analysis) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto insertOp = mlir::cast<mlir::subop::InsertOp>(op);
      auto state = insertOp.getState();
      auto multimapType = mlir::dyn_cast_or_null<mlir::subop::MultiMapType>(state.getType());
      if (!multimapType) return mlir::failure();
      std::vector<mlir::subop::LookupOp> lookupOps;
      for (auto* user : state.getUsers()) {
         if (user == op) {
            //ignore use of materializeop
         } else if (auto lookupOp = mlir::dyn_cast_or_null<mlir::subop::LookupOp>(user)) {
            lookupOps.push_back(lookupOp);
         } else {
            return mlir::failure();
         }
      }

      if (auto scanOp = mlir::dyn_cast_or_null<mlir::subop::ScanOp>(insertOp.getStream().getDefiningOp())) {
         if (auto htType = mlir::dyn_cast_or_null<mlir::subop::MapType>(scanOp.getState().getType())) {
            std::vector<mlir::tuples::Column*> hashedColumns;
            for (auto m : multimapType.getKeyMembers().getNames()) {
               hashedColumns.push_back(&mlir::cast<mlir::tuples::ColumnRefAttr>(insertOp.getMapping().get(mlir::cast<mlir::StringAttr>(m).strref())).getColumn());
            }
            std::unordered_map<std::string, std::string> memberMapping;
            for (auto m : insertOp.getMapping()) {
               auto colDef = colManager.createDef(&mlir::cast<mlir::tuples::ColumnRefAttr>(m.getValue()).getColumn());
               auto hmMember = lookupByValue(scanOp.getMapping(), colDef);
               auto bufferMember = m.getName().str();
               if (hmMember) {
                  memberMapping[bufferMember] = hmMember.value();
               }
            }
            std::unordered_set<mlir::tuples::Column*> hashMapKey;
            std::unordered_map<std::string, mlir::tuples::Column*> keyMemberToColumn;
            for (auto keyMember : htType.getKeyMembers().getNames()) {
               auto colDef = mlir::cast<mlir::tuples::ColumnDefAttr>(scanOp.getMapping().get(mlir::cast<mlir::StringAttr>(keyMember).str()));
               hashMapKey.insert(&colDef.getColumn());
               keyMemberToColumn[mlir::cast<mlir::StringAttr>(keyMember).str()] = &colDef.getColumn();
            }
            if (hashMapKey.size() != hashedColumns.size()) {
               return mlir::failure();
            }
            for (auto* c : hashedColumns) {
               if (!hashMapKey.contains(c)) {
                  return mlir::failure();
               }
            }
            auto hmRefType = mlir::subop::MapEntryRefType::get(getContext(), htType);
            mlir::subop::SubOpStateUsageTransformer transformer(analysis, getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
               return llvm::TypeSwitch<mlir::Operation*, mlir::Type>(op)
                  .Case([&](mlir::subop::ScanListOp scanListOp) {
                     return hmRefType;
                  })
                  .Default([&](mlir::Operation* op) {
                     assert(false && "not supported yet");
                     return mlir::Type();
                  });
            });
            rewriter.eraseOp(insertOp);
            transformer.mapMembers(memberMapping);
            for (auto lookupOp : lookupOps) {
               std::vector<mlir::tuples::Column*> lookupHashedColumns;
               for (auto c : lookupOp.getKeys()) {
                  lookupHashedColumns.push_back(&mlir::cast<mlir::tuples::ColumnRefAttr>(c).getColumn());
               }
               if (lookupHashedColumns.size() != hashedColumns.size()) return mlir::failure();
               std::unordered_map<mlir::tuples::Column*, mlir::tuples::Column*> columnMapping;
               for (auto z : llvm::zip(lookupHashedColumns, hashedColumns)) {
                  columnMapping.insert({std::get<1>(z), std::get<0>(z)});
               }
               std::vector<mlir::Attribute> lookupColumns;
               for (auto keyMember : htType.getKeyMembers().getNames()) {
                  auto* col = columnMapping.at(keyMemberToColumn.at(mlir::cast<mlir::StringAttr>(keyMember).str()));
                  lookupColumns.push_back(colManager.createRef(col));
               }
               rewriter.setInsertionPointAfter(lookupOp);
               auto hmRefListType = mlir::subop::ListType::get(getContext(), hmRefType);
               auto lookupRef = lookupOp.getRef();
               auto [listDef, listRef] = createColumn(hmRefListType, "lookup", "list");
               auto* equalityBlock = &lookupOp.getEqFn().front();
               lookupOp.getEqFn().getBlocks().remove(equalityBlock);
               auto newLookupOp = rewriter.replaceOpWithNewOp<mlir::subop::LookupOp>(lookupOp, lookupOp.getStream(), scanOp.getState(), rewriter.getArrayAttr(lookupColumns), listDef);
               newLookupOp.getEqFn().push_back(equalityBlock);
               transformer.replaceColumn(&lookupRef.getColumn(), &listDef.getColumn());
            }
            transformer.updateValue(state, htType);
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
      auto columnUsageAnalysis = getAnalysis<mlir::subop::ColumnUsageAnalysis>();

      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<AvoidUnnecessaryMaterialization>(&getContext());
      patterns.insert<AvoidArrayMaterialization>(&getContext(), columnUsageAnalysis);
      patterns.insert<AvoidDeadMaterialization>(&getContext());
      patterns.insert<ReuseHashtable>(&getContext(), columnUsageAnalysis);
      if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::subop::createReuseLocalPass() { return std::make_unique<ReuseLocalPass>(); }