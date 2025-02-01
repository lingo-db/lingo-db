
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/TypeSwitch.h"

#include <unordered_set>

namespace {
using namespace lingodb::compiler::dialect;

static std::vector<subop::SubOperator> findWritingOps(mlir::Block& block, std::vector<std::string>& members) {
   std::unordered_set<std::string> memberSet(members.begin(), members.end());
   std::vector<subop::SubOperator> res;
   block.walk([&memberSet, &res](subop::SubOperator subop) {
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
static std::pair<tuples::ColumnDefAttr, tuples::ColumnRefAttr> createColumn(mlir::Type type, std::string scope, std::string name) {
   auto& columnManager = type.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope(scope);
   std::string attributeName = name;
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = type;
   return {markAttrDef, columnManager.createRef(&ra)};
}

class AvoidUnnecessaryMaterialization : public mlir::RewritePattern {
   public:
   AvoidUnnecessaryMaterialization(mlir::MLIRContext* context)
      : RewritePattern(subop::ScanOp::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      auto scanOp = mlir::cast<subop::ScanOp>(op);
      if (!mlir::isa<subop::BufferType>(scanOp.getState().getType())) {
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
      if (auto materializeOp = mlir::dyn_cast_or_null<subop::MaterializeOp>(writingOps[0].getOperation())) {
         if (materializeOp->getBlock() == scanOp->getBlock() && materializeOp->isBeforeInBlock(scanOp)) {
            if (auto scanOp2 = mlir::dyn_cast_or_null<subop::ScanOp>(materializeOp.getStream().getDefiningOp())) {
               std::vector<mlir::NamedAttribute> newMapping;
               for (auto curr : scanOp.getMapping()) {
                  auto currentMember = curr.getName();
                  auto otherColumnDef = colManager.createDef(&mlir::cast<tuples::ColumnRefAttr>(materializeOp.getMapping().get(currentMember)).getColumn());
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
   subop::ColumnUsageAnalysis& analysis;
   mlir::Operation* getRoot(mlir::Operation* op) const {
      if (op->getNumOperands() < 1) {
         return op;
      }
      bool isFirstTupleStream = mlir::isa<tuples::TupleStreamType>(op->getOperand(0).getType());
      bool noOtherTupleStream = llvm::none_of(op->getOperands().drop_front(), [](mlir::Value v) { return mlir::isa<tuples::TupleStreamType>(v.getType()); });
      if (isFirstTupleStream && noOtherTupleStream) {
         return getRoot(op->getOperand(0).getDefiningOp());
      } else if (isFirstTupleStream) {
         return nullptr;
      } else {
         return op;
      }
   }

   public:
   AvoidArrayMaterialization(mlir::MLIRContext* context, subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(subop::ScanRefsOp::getOperationName(), 1, context), analysis(analysis) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      auto scanRefsOp = mlir::cast<subop::ScanRefsOp>(op);
      if (!mlir::isa<subop::ArrayType>(scanRefsOp.getState().getType())) {
         return mlir::failure();
      }
      if (scanRefsOp->hasAttr("sequential")) {
         return mlir::failure();
      }
      auto state = scanRefsOp.getState();
      std::vector<subop::GetBeginReferenceOp> otherUsers;
      for (auto* user : state.getUsers()) {
         if (user == op) {
            continue;
         } else if (auto getBeginOp = mlir::dyn_cast_or_null<subop::GetBeginReferenceOp>(user)) {
            otherUsers.push_back(getBeginOp);
         } else {
            return mlir::failure();
         }
      }
      if (!otherUsers.size()) {
         return mlir::failure();
      }
      subop::ScatterOp scatterOp;
      subop::OffsetReferenceBy offsetByOp;
      std::vector<subop::EntriesBetweenOp> needsChanges;
      tuples::ColumnRefAttr replaceWith;
      tuples::ColumnRefAttr replaceIdxWith;
      for (auto user : otherUsers) {
         auto colUsers = analysis.findOperationsUsing(&user.getRef().getColumn());
         for (auto* colUser : colUsers) {
            if (auto* root = getRoot(colUser)) {
               if (root == op) {
                  if (auto entriesBetween = mlir::dyn_cast_or_null<subop::EntriesBetweenOp>(colUser)) {
                     needsChanges.push_back(entriesBetween);
                  } else {
                     return mlir::failure();
                  }
               } else {
                  if (auto otherScanRefsOp = mlir::dyn_cast_or_null<subop::ScanRefsOp>(root)) {
                     auto otherState = otherScanRefsOp.getState();
                     if (!mlir::isa<subop::ArrayType>(otherState.getType())) {
                        return mlir::failure();
                     }
                     if (auto offsetBy = mlir::dyn_cast_or_null<subop::OffsetReferenceBy>(colUser)) {
                        bool idxMatches = false;
                        for (auto* otherScanRefsUser : analysis.findOperationsUsing(&otherScanRefsOp.getRef().getColumn())) {
                           if (auto entriesBetweenOp = mlir::dyn_cast_or_null<subop::EntriesBetweenOp>(otherScanRefsUser)) {
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
                        if (auto localScatterOp = mlir::dyn_cast_or_null<subop::ScatterOp>(*offsetByUsers.begin())) {
                           if (scatterOp) {
                              return mlir::failure();
                           }
                           scatterOp = localScatterOp;
                           offsetByOp = offsetBy;
                           replaceWith = colManager.createRef(&otherScanRefsOp.getRef().getColumn());
                           if (scatterOp.getWrittenMembers().size() != mlir::cast<subop::State>(scanRefsOp.getState().getType()).getMembers().getNames().size()) {
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
      std::vector<subop::GatherOp> gatherOps;
      for (auto* scanRefUser : scanRefUsers) {
         if (auto gatherOp = mlir::dyn_cast_or_null<subop::GatherOp>(scanRefUser)) {
            gatherOps.push_back(gatherOp);
         } else if (mlir::isa<subop::EntriesBetweenOp>(scanRefUser)) {
         } else {
            return mlir::failure();
         }
      }
      if (scatterOp) {
         std::unordered_map<std::string, tuples::ColumnRefAttr> scatterMapping;
         for (auto m : scatterOp.getMapping()) {
            scatterMapping.insert({m.getName().str(), mlir::cast<tuples::ColumnRefAttr>(m.getValue())});
         }
         std::vector<mlir::Attribute> renamed;
         for (auto gatherOp : gatherOps) {
            for (auto m : gatherOp.getMapping()) {
               renamed.push_back(colManager.createDef(&mlir::cast<tuples::ColumnDefAttr>(m.getValue()).getColumn(), rewriter.getArrayAttr(scatterMapping.at(m.getName().str()))));
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
         rewriter.replaceOpWithNewOp<subop::RenamingOp>(op, scatterOp.getStream(), rewriter.getArrayAttr(renamed));
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
      : RewritePattern(subop::MaterializeOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto materializeOp = mlir::cast<subop::MaterializeOp>(op);
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
   const subop::ColumnUsageAnalysis& analysis;

   public:
   ReuseHashtable(mlir::MLIRContext* context, subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(subop::InsertOp::getOperationName(), 1, context), analysis(analysis) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& colManager = rewriter.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      auto insertOp = mlir::cast<subop::InsertOp>(op);
      auto state = insertOp.getState();
      auto multimapType = mlir::dyn_cast_or_null<subop::MultiMapType>(state.getType());
      if (!multimapType) return mlir::failure();
      std::vector<subop::LookupOp> lookupOps;
      for (auto* user : state.getUsers()) {
         if (user == op) {
            //ignore use of materializeop
         } else if (auto lookupOp = mlir::dyn_cast_or_null<subop::LookupOp>(user)) {
            lookupOps.push_back(lookupOp);
         } else {
            return mlir::failure();
         }
      }

      if (auto scanOp = mlir::dyn_cast_or_null<subop::ScanOp>(insertOp.getStream().getDefiningOp())) {
         if (auto htType = mlir::dyn_cast_or_null<subop::MapType>(scanOp.getState().getType())) {
            std::vector<tuples::Column*> hashedColumns;
            for (auto m : multimapType.getKeyMembers().getNames()) {
               hashedColumns.push_back(&mlir::cast<tuples::ColumnRefAttr>(insertOp.getMapping().get(mlir::cast<mlir::StringAttr>(m).strref())).getColumn());
            }
            std::unordered_map<std::string, std::string> memberMapping;
            for (auto m : insertOp.getMapping()) {
               auto colDef = colManager.createDef(&mlir::cast<tuples::ColumnRefAttr>(m.getValue()).getColumn());
               auto hmMember = lookupByValue(scanOp.getMapping(), colDef);
               auto bufferMember = m.getName().str();
               if (hmMember) {
                  memberMapping[bufferMember] = hmMember.value();
               }
            }
            std::unordered_set<tuples::Column*> hashMapKey;
            std::unordered_map<std::string, tuples::Column*> keyMemberToColumn;
            for (auto keyMember : htType.getKeyMembers().getNames()) {
               auto colDef = mlir::cast<tuples::ColumnDefAttr>(scanOp.getMapping().get(mlir::cast<mlir::StringAttr>(keyMember).str()));
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
            auto hmRefType = subop::MapEntryRefType::get(getContext(), htType);
            subop::SubOpStateUsageTransformer transformer(analysis, getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
               return llvm::TypeSwitch<mlir::Operation*, mlir::Type>(op)
                  .Case([&](subop::ScanListOp scanListOp) {
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
               std::vector<tuples::Column*> lookupHashedColumns;
               for (auto c : lookupOp.getKeys()) {
                  lookupHashedColumns.push_back(&mlir::cast<tuples::ColumnRefAttr>(c).getColumn());
               }
               if (lookupHashedColumns.size() != hashedColumns.size()) return mlir::failure();
               std::unordered_map<tuples::Column*, tuples::Column*> columnMapping;
               for (auto z : llvm::zip(lookupHashedColumns, hashedColumns)) {
                  columnMapping.insert({std::get<1>(z), std::get<0>(z)});
               }
               std::vector<mlir::Attribute> lookupColumns;
               for (auto keyMember : htType.getKeyMembers().getNames()) {
                  auto* col = columnMapping.at(keyMemberToColumn.at(mlir::cast<mlir::StringAttr>(keyMember).str()));
                  lookupColumns.push_back(colManager.createRef(col));
               }
               rewriter.setInsertionPointAfter(lookupOp);
               auto hmRefListType = subop::ListType::get(getContext(), hmRefType);
               auto lookupRef = lookupOp.getRef();
               auto [listDef, listRef] = createColumn(hmRefListType, "lookup", "list");
               auto* equalityBlock = &lookupOp.getEqFn().front();
               lookupOp.getEqFn().getBlocks().remove(equalityBlock);
               auto newLookupOp = rewriter.replaceOpWithNewOp<subop::LookupOp>(lookupOp, lookupOp.getStream(), scanOp.getState(), rewriter.getArrayAttr(lookupColumns), listDef);
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
      auto columnUsageAnalysis = getAnalysis<subop::ColumnUsageAnalysis>();

      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<AvoidUnnecessaryMaterialization>(&getContext());
      patterns.insert<AvoidArrayMaterialization>(&getContext(), columnUsageAnalysis);
      patterns.insert<AvoidDeadMaterialization>(&getContext());
      patterns.insert<ReuseHashtable>(&getContext(), columnUsageAnalysis);
      if (mlir::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> subop::createReuseLocalPass() { return std::make_unique<ReuseLocalPass>(); }