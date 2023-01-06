#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace {
static std::pair<mlir::tuples::ColumnDefAttr, mlir::tuples::ColumnRefAttr> createColumn(mlir::Type type, std::string scope, std::string name) {
   auto& columnManager = type.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope(scope);
   std::string attributeName = name;
   mlir::tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = type;
   return {markAttrDef, columnManager.createRef(&ra)};
}

class MultiMapAsHashIndexedView : public mlir::RewritePattern {
   const mlir::subop::ColumnUsageAnalysis& analysis;

   public:
   MultiMapAsHashIndexedView(mlir::MLIRContext* context, mlir::subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(mlir::subop::MaterializeOp::getOperationName(), 1, context), analysis(analysis) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& memberManager = getContext()->getLoadedDialect<mlir::subop::SubOperatorDialect>()->getMemberManager();
      auto materializeOp = mlir::cast<mlir::subop::MaterializeOp>(op);
      auto state = materializeOp.getState();
      auto multiMapType = state.getType().dyn_cast_or_null<mlir::subop::MultiMapType>();
      if (!multiMapType) {
         return mlir::failure();
      }
      std::vector<mlir::subop::LookupOp> lookupOps;
      std::vector<mlir::Operation*> otherUses;
      mlir::subop::GenericCreateOp createOp = mlir::dyn_cast_or_null<mlir::subop::GenericCreateOp>(state.getDefiningOp());
      for (auto* u : state.getUsers()) {
         if (u == op) {
            //ignore use;
         } else if (auto lookupOp = mlir::dyn_cast_or_null<mlir::subop::LookupOp>(u)) {
            lookupOps.push_back(lookupOp);
         } else {
            otherUses.push_back(u);
         }
      }

      auto hashMember = memberManager.getUniqueMember("hash");
      auto linkMember = memberManager.getUniqueMember("link");
      auto [hashDef, hashRef] = createColumn(rewriter.getIndexType(), "hj", "hash");
      auto linkType = mlir::util::RefType::get(rewriter.getContext(), rewriter.getI8Type());
      auto [linkDef, linkRef] = createColumn(linkType, "hj", "link");
      auto loc = op->getLoc();

      std::vector<mlir::Attribute> bufferMemberTypes{mlir::TypeAttr::get(rewriter.getIndexType()), mlir::TypeAttr::get(linkType)};
      std::vector<mlir::Attribute> bufferMemberNames{rewriter.getStringAttr(linkMember), rewriter.getStringAttr(hashMember)};
      bufferMemberNames.insert(bufferMemberNames.end(), multiMapType.getMembers().getNames().begin(), multiMapType.getMembers().getNames().end());
      bufferMemberTypes.insert(bufferMemberTypes.end(), multiMapType.getMembers().getTypes().begin(), multiMapType.getMembers().getTypes().end());
      std::vector<mlir::Attribute> hashIndexedViewTypes{multiMapType.getMembers().getTypes().begin(), multiMapType.getMembers().getTypes().end()};
      std::vector<mlir::Attribute> hashIndexedViewNames{multiMapType.getMembers().getNames().begin(), multiMapType.getMembers().getNames().end()};

      auto bufferType = mlir::subop::BufferType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(getContext(), rewriter.getArrayAttr(bufferMemberNames), rewriter.getArrayAttr(bufferMemberTypes)));
      mlir::Value buffer;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPoint(createOp);
         buffer = rewriter.create<mlir::subop::GenericCreateOp>(loc, bufferType);
      }
      mlir::Type hashIndexedViewType;
      mlir::Value hashIndexedView;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPoint(materializeOp);
         auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, materializeOp.getStream(), rewriter.getArrayAttr({hashDef, linkDef}));
         std::vector<mlir::NamedAttribute> newMapping(materializeOp.getMapping().begin(), materializeOp.getMapping().end());
         newMapping.push_back(rewriter.getNamedAttr(hashMember, hashRef));
         newMapping.push_back(rewriter.getNamedAttr(linkMember, linkRef));
         rewriter.create<mlir::subop::MaterializeOp>(loc, mapOp.getResult(), buffer, rewriter.getDictionaryAttr(newMapping));
         hashIndexedViewType = mlir::subop::HashIndexedViewType::get(rewriter.getContext(), mlir::subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr({rewriter.getStringAttr(hashMember)}), rewriter.getArrayAttr({mlir::TypeAttr::get(rewriter.getIndexType())})), mlir::subop::StateMembersAttr::get(getContext(), rewriter.getArrayAttr(hashIndexedViewNames), rewriter.getArrayAttr(hashIndexedViewTypes)));
         hashIndexedView = rewriter.create<mlir::subop::CreateHashIndexedView>(loc, hashIndexedViewType, buffer, hashMember, linkMember);
         auto* mapBlock = new mlir::Block;
         mlir::Value tuple = mapBlock->addArgument(mlir::tuples::TupleType::get(getContext()), loc);
         mapOp.getFn().push_back(mapBlock);
         rewriter.setInsertionPointToStart(mapBlock);
         std::vector<mlir::Value> values;
         for (auto keyMemberName : multiMapType.getKeyMembers().getNames()) {
            auto keyColumn = materializeOp.getMapping().get(keyMemberName.cast<mlir::StringAttr>().strref()).cast<mlir::tuples::ColumnRefAttr>();
            values.push_back(rewriter.create<mlir::tuples::GetColumnOp>(loc, keyColumn.getColumn().type, keyColumn, tuple));
         }
         mlir::Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.create<mlir::util::PackOp>(loc, values));
         mlir::Value inValidLink = rewriter.create<mlir::util::InvalidRefOp>(loc, linkType);
         rewriter.create<mlir::tuples::ReturnOp>(loc, mlir::ValueRange{hashed, inValidLink});
      }
      auto entryRefType = mlir::subop::LookupEntryRefType::get(rewriter.getContext(), hashIndexedViewType);
      auto entryRefListType = mlir::subop::ListType::get(rewriter.getContext(), entryRefType);
      mlir::subop::SubOpStateUsageTransformer transformer(analysis, getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
         return llvm::TypeSwitch<mlir::Operation*, mlir::Type>(op)
            .Case([&](mlir::subop::ScanListOp scanListOp) {
               return entryRefType;
            })
            .Default([&](mlir::Operation* op) {
               assert(false && "not supported yet");
               return mlir::Type();
            });

         //
      });
      for (auto lookupOp : lookupOps) {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPoint(lookupOp);
         auto [hashDefLookup, hashRefLookup] = createColumn(rewriter.getIndexType(), "hj", "hash");
         auto mapOp = rewriter.create<mlir::subop::MapOp>(loc, lookupOp.getStream(), rewriter.getArrayAttr({hashDefLookup}));
         auto [listDef, listRef] = createColumn(entryRefListType, "lookup", "list");
         auto lookupRef = lookupOp.getRef();
         auto lookupKeys = lookupOp.getKeys();
         rewriter.replaceOpWithNewOp<mlir::subop::LookupOp>(lookupOp, mlir::tuples::TupleStreamType::get(rewriter.getContext()), mapOp.getResult(), hashIndexedView, rewriter.getArrayAttr({hashRefLookup}), listDef);
         auto* mapBlock = new mlir::Block;
         mlir::Value tuple = mapBlock->addArgument(mlir::tuples::TupleType::get(getContext()), loc);
         mapOp.getFn().push_back(mapBlock);
         rewriter.setInsertionPointToStart(mapBlock);
         std::vector<mlir::Value> values;
         for (auto key : lookupKeys) {
            auto keyColumn = key.cast<mlir::tuples::ColumnRefAttr>();
            values.push_back(rewriter.create<mlir::tuples::GetColumnOp>(loc, keyColumn.getColumn().type, keyColumn, tuple));
         }
         mlir::Value hashed = rewriter.create<mlir::db::Hash>(loc, rewriter.create<mlir::util::PackOp>(loc, values));
         rewriter.create<mlir::tuples::ReturnOp>(loc, mlir::ValueRange{hashed});
         transformer.replaceColumn(&lookupRef.getColumn(), &listDef.getColumn());
      }
      rewriter.eraseOp(materializeOp);
      transformer.updateValue(state, buffer.getType());
      rewriter.replaceOp(createOp, buffer);
      return mlir::success();
   }
};
class SpecializeSubOpPass : public mlir::PassWrapper<SpecializeSubOpPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpecializeSubOpPass)
   virtual llvm::StringRef getArgument() const override { return "subop-specialize"; }

   SpecializeSubOpPass(bool withOptimizations){}
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::util::UtilDialect, mlir::db::DBDialect>();
   }
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      auto columnUsageAnalysis = getAnalysis<mlir::subop::ColumnUsageAnalysis>();

      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<MultiMapAsHashIndexedView>(&getContext(), columnUsageAnalysis);

      if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         assert(false && "should not happen");
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createSpecializeSubOpPass(bool withOptimizations) { return std::make_unique<SpecializeSubOpPass>(withOptimizations); }