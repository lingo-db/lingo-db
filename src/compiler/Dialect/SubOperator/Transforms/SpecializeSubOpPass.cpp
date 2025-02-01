#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Utils.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace {
using namespace lingodb::compiler::dialect;

static std::pair<tuples::ColumnDefAttr, tuples::ColumnRefAttr> createColumn(mlir::Type type, std::string scope, std::string name) {
   auto& columnManager = type.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope(scope);
   std::string attributeName = name;
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = type;
   return {markAttrDef, columnManager.createRef(&ra)};
}

class MultiMapAsHashIndexedView : public mlir::RewritePattern {
   const subop::ColumnUsageAnalysis& analysis;

   public:
   MultiMapAsHashIndexedView(mlir::MLIRContext* context, subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(subop::GenericCreateOp::getOperationName(), 1, context), analysis(analysis) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& memberManager = getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
      auto& columnManager = getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

      auto createOp = mlir::cast<subop::GenericCreateOp>(op);
      auto state = createOp.getRes();
      auto multiMapType = mlir::dyn_cast_or_null<subop::MultiMapType>(state.getType());
      if (!multiMapType) {
         return mlir::failure();
      }
      std::vector<subop::LookupOp> lookupOps;
      std::vector<mlir::Operation*> otherUses;
      subop::InsertOp insertOp;
      for (auto* u : state.getUsers()) {
         if (auto mOp = mlir::dyn_cast_or_null<subop::InsertOp>(u)) {
            if (insertOp) {
               return mlir::failure();
            } else {
               insertOp = mOp;
            }
         } else if (auto lookupOp = mlir::dyn_cast_or_null<subop::LookupOp>(u)) {
            lookupOps.push_back(lookupOp);
         } else {
            otherUses.push_back(u);
         }
      }

      auto hashMember = memberManager.getUniqueMember("hash");
      auto linkMember = memberManager.getUniqueMember("link");
      auto [hashDef, hashRef] = createColumn(rewriter.getIndexType(), "hj", "hash");
      auto linkType = util::RefType::get(rewriter.getContext(), rewriter.getI8Type());
      auto [linkDef, linkRef] = createColumn(linkType, "hj", "link");
      auto loc = op->getLoc();

      std::vector<mlir::Attribute> bufferMemberTypes{mlir::TypeAttr::get(linkType), mlir::TypeAttr::get(rewriter.getIndexType())};
      std::vector<mlir::Attribute> bufferMemberNames{rewriter.getStringAttr(linkMember), rewriter.getStringAttr(hashMember)};
      bufferMemberNames.insert(bufferMemberNames.end(), multiMapType.getMembers().getNames().begin(), multiMapType.getMembers().getNames().end());
      bufferMemberTypes.insert(bufferMemberTypes.end(), multiMapType.getMembers().getTypes().begin(), multiMapType.getMembers().getTypes().end());
      std::vector<mlir::Attribute> hashIndexedViewTypes{multiMapType.getMembers().getTypes().begin(), multiMapType.getMembers().getTypes().end()};
      std::vector<mlir::Attribute> hashIndexedViewNames{multiMapType.getMembers().getNames().begin(), multiMapType.getMembers().getNames().end()};

      auto bufferType = subop::BufferType::get(rewriter.getContext(), subop::StateMembersAttr::get(getContext(), rewriter.getArrayAttr(bufferMemberNames), rewriter.getArrayAttr(bufferMemberTypes)));
      mlir::Value buffer;
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPoint(createOp);
         buffer = rewriter.create<subop::GenericCreateOp>(loc, bufferType);
      }
      mlir::Type hashIndexedViewType;
      mlir::Value hashIndexedView;

      subop::MapCreationHelper buildHashHelper(rewriter.getContext());
      buildHashHelper.buildBlock(rewriter, [&](mlir::PatternRewriter& rewriter) {
         std::vector<mlir::Value> values;
         for (auto keyMemberName : multiMapType.getKeyMembers().getNames()) {
            auto keyColumnAttr = mlir::cast<tuples::ColumnRefAttr>(insertOp.getMapping().get(mlir::cast<mlir::StringAttr>(keyMemberName).strref()));
            values.push_back(buildHashHelper.access(keyColumnAttr, loc));
         }
         mlir::Value hashed = rewriter.create<db::Hash>(loc, rewriter.create<util::PackOp>(loc, values));
         mlir::Value inValidLink = rewriter.create<util::InvalidRefOp>(loc, linkType);
         rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange{hashed, inValidLink});
      });
      {
         mlir::OpBuilder::InsertionGuard guard(rewriter);
         rewriter.setInsertionPoint(insertOp);
         auto mapOp = rewriter.create<subop::MapOp>(loc, insertOp.getStream(), rewriter.getArrayAttr({hashDef, linkDef}), buildHashHelper.getColRefs());
         mapOp.getFn().push_back(buildHashHelper.getMapBlock());
         std::vector<mlir::NamedAttribute> newMapping(insertOp.getMapping().begin(), insertOp.getMapping().end());
         newMapping.push_back(rewriter.getNamedAttr(hashMember, hashRef));
         newMapping.push_back(rewriter.getNamedAttr(linkMember, linkRef));
         rewriter.create<subop::MaterializeOp>(loc, mapOp.getResult(), buffer, rewriter.getDictionaryAttr(newMapping));
         hashIndexedViewType = subop::HashIndexedViewType::get(rewriter.getContext(), subop::StateMembersAttr::get(rewriter.getContext(), rewriter.getArrayAttr({rewriter.getStringAttr(hashMember)}), rewriter.getArrayAttr({mlir::TypeAttr::get(rewriter.getIndexType())})), subop::StateMembersAttr::get(getContext(), rewriter.getArrayAttr(hashIndexedViewNames), rewriter.getArrayAttr(hashIndexedViewTypes)));
         hashIndexedView = rewriter.create<subop::CreateHashIndexedView>(loc, hashIndexedViewType, buffer, hashMember, linkMember);
      }
      auto entryRefType = subop::LookupEntryRefType::get(rewriter.getContext(), mlir::cast<subop::LookupAbleState>(hashIndexedViewType));
      auto entryRefListType = subop::ListType::get(rewriter.getContext(), entryRefType);
      subop::SubOpStateUsageTransformer transformer(analysis, getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
         return llvm::TypeSwitch<mlir::Operation*, mlir::Type>(op)
            .Case([&](subop::ScanListOp scanListOp) {
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
         auto [lookupPredDef, lookupPredRef] = createColumn(rewriter.getI1Type(), "lookup", "pred");
         auto lookupKeys = lookupOp.getKeys();
         subop::MapCreationHelper lookupHashHelper(rewriter.getContext());
         lookupHashHelper.buildBlock(rewriter, [&](mlir::PatternRewriter& rewriter) {
            std::vector<mlir::Value> values;
            for (auto key : lookupKeys) {
               auto keyColumnAttr = mlir::cast<tuples::ColumnRefAttr>(key);
               values.push_back(lookupHashHelper.access(keyColumnAttr, loc));
            }
            mlir::Value hashed = rewriter.create<db::Hash>(loc, rewriter.create<util::PackOp>(loc, values));
            rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange{hashed});
         });
         auto [listDef, listRef] = createColumn(entryRefListType, "lookup", "list");
         auto lookupRef = lookupOp.getRef();
         std::vector<mlir::NamedAttribute> gatheredForEqFn;
         std::vector<tuples::ColumnRefAttr> keyRefsForEqFn;
         for (auto keyMember : llvm::zip(multiMapType.getKeyMembers().getNames(), multiMapType.getKeyMembers().getTypes())) {
            auto name = mlir::cast<mlir::StringAttr>(std::get<0>(keyMember)).str();
            auto type = mlir::cast<mlir::TypeAttr>(std::get<1>(keyMember)).getValue();
            auto [lookupKeyMemberDef, lookupKeyMemberRef] = createColumn(type, "lookup", name);
            gatheredForEqFn.push_back(rewriter.getNamedAttr(name, lookupKeyMemberDef));
            keyRefsForEqFn.push_back(lookupKeyMemberRef);
         }
         auto mapOp = rewriter.create<subop::MapOp>(loc, lookupOp.getStream(), rewriter.getArrayAttr({hashDefLookup}), lookupHashHelper.getColRefs());
         mapOp.getFn().push_back(lookupHashHelper.getMapBlock());
         subop::MapCreationHelper predFnHelper(rewriter.getContext());
         predFnHelper.buildBlock(rewriter, [&](mlir::PatternRewriter& rewriter) {
            mlir::IRMapping mapping;
            size_t i = 0;
            for (auto key : keyRefsForEqFn) {
               mapping.map(lookupOp.getEqFn().getArgument(i++), predFnHelper.access(key, loc));
            }
            for (auto key : lookupKeys) {
               auto keyColumn = mlir::cast<tuples::ColumnRefAttr>(key);
               mapping.map(lookupOp.getEqFn().getArgument(i++), predFnHelper.access(keyColumn, loc));
            }
            for (auto& op : lookupOp.getEqFn().front()) {
               rewriter.clone(op, mapping);
            }
         });
         rewriter.replaceOpWithNewOp<subop::LookupOp>(lookupOp, tuples::TupleStreamType::get(rewriter.getContext()), mapOp.getResult(), hashIndexedView, rewriter.getArrayAttr({hashRefLookup}), listDef);

         mlir::Value currentTuple;
         transformer.setCallBeforeFn([&](mlir::Operation* op) {
            if (auto nestedMapOp = mlir::dyn_cast_or_null<subop::NestedMapOp>(op)) {
               currentTuple = nestedMapOp.getRegion().getArgument(0);
            }
         });
         transformer.setCallAfterFn([&](mlir::Operation* op) {
            if (auto scanListOp = mlir::dyn_cast_or_null<subop::ScanListOp>(op)) {
               assert(!!currentTuple);
               rewriter.setInsertionPointAfter(scanListOp);
               auto combined = rewriter.create<subop::CombineTupleOp>(loc, scanListOp.getRes(), currentTuple);
               auto gatherOp = rewriter.create<subop::GatherOp>(loc, combined.getRes(), columnManager.createRef(&scanListOp.getElem().getColumn()), rewriter.getDictionaryAttr(gatheredForEqFn));
               auto mapOp = rewriter.create<subop::MapOp>(loc, gatherOp.getRes(), rewriter.getArrayAttr({lookupPredDef}), predFnHelper.getColRefs());
               mapOp.getFn().push_back(predFnHelper.getMapBlock());
               auto filter = rewriter.create<subop::FilterOp>(loc, mapOp.getResult(), subop::FilterSemantic::all_true, rewriter.getArrayAttr({lookupPredRef}));
               scanListOp.getRes().replaceAllUsesExcept(filter.getResult(), combined);
            }
         });
         transformer.replaceColumn(&lookupRef.getColumn(), &listDef.getColumn());
         transformer.setCallBeforeFn({});
         transformer.setCallAfterFn({});
      }
      rewriter.eraseOp(insertOp);
      transformer.updateValue(state, buffer.getType());
      rewriter.replaceOp(createOp, buffer);
      return mlir::success();
   }
};
class MapAsHashMap : public mlir::RewritePattern {
   const subop::ColumnUsageAnalysis& analysis;

   public:
   MapAsHashMap(mlir::MLIRContext* context, subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(subop::GenericCreateOp::getOperationName(), 1, context), analysis(analysis) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto createOp = mlir::cast<subop::GenericCreateOp>(op);
      auto mapType = mlir::dyn_cast_or_null<subop::MapType>(createOp.getType());
      if (!mapType) {
         return mlir::failure();
      }
      auto hashMapType = subop::HashMapType::get(getContext(), mapType.getKeyMembers(), mapType.getValueMembers());

      mlir::TypeConverter typeConverter;
      typeConverter.addConversion([&](subop::ListType listType) {
         return subop::ListType::get(listType.getContext(), mlir::cast<subop::StateEntryReference>(typeConverter.convertType(listType.getT())));
      });
      typeConverter.addConversion([&](subop::OptionalType optionalType) {
         return subop::OptionalType::get(optionalType.getContext(), mlir::cast<subop::StateEntryReference>(typeConverter.convertType(optionalType.getT())));
      });
      typeConverter.addConversion([&](subop::MapEntryRefType refType) {
         return subop::HashMapEntryRefType::get(refType.getContext(), hashMapType);
      });
      typeConverter.addConversion([&](subop::LookupEntryRefType lookupRefType) {
         return subop::LookupEntryRefType::get(lookupRefType.getContext(), mlir::cast<subop::LookupAbleState>(typeConverter.convertType(lookupRefType.getState())));
      });
      typeConverter.addConversion([&](subop::MapType mapType) {
         return hashMapType;
      });
      subop::SubOpStateUsageTransformer transformer(analysis, getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
         return typeConverter.convertType(type);
      });
      transformer.updateValue(createOp.getRes(), hashMapType);
      rewriter.replaceOpWithNewOp<subop::GenericCreateOp>(op, hashMapType);

      return mlir::success();
   }
};
class MultiMapAsHashMultiMap : public mlir::RewritePattern {
   const subop::ColumnUsageAnalysis& analysis;

   public:
   MultiMapAsHashMultiMap(mlir::MLIRContext* context, subop::ColumnUsageAnalysis& analysis)
      : RewritePattern(subop::GenericCreateOp::getOperationName(), 1, context), analysis(analysis) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto createOp = mlir::cast<subop::GenericCreateOp>(op);
      auto multiMapType = mlir::dyn_cast_or_null<subop::MultiMapType>(createOp.getType());
      if (!multiMapType) {
         return mlir::failure();
      }
      auto hashMapType = subop::HashMultiMapType::get(getContext(), multiMapType.getKeyMembers(), multiMapType.getValueMembers());

      mlir::TypeConverter typeConverter;
      typeConverter.addConversion([&](subop::ListType listType) {
         return subop::ListType::get(listType.getContext(), mlir::cast<subop::StateEntryReference>(typeConverter.convertType(listType.getT())));
      });
      typeConverter.addConversion([&](subop::OptionalType optionalType) {
         return subop::OptionalType::get(optionalType.getContext(), mlir::cast<subop::StateEntryReference>(typeConverter.convertType(optionalType.getT())));
      });
      typeConverter.addConversion([&](subop::MultiMapEntryRefType refType) {
         return subop::HashMultiMapEntryRefType::get(refType.getContext(), hashMapType);
      });
      typeConverter.addConversion([&](subop::LookupEntryRefType lookupRefType) {
         return subop::LookupEntryRefType::get(lookupRefType.getContext(), mlir::cast<subop::LookupAbleState>(typeConverter.convertType(lookupRefType.getState())));
      });
      typeConverter.addConversion([&](subop::MultiMapType mapType) {
         return hashMapType;
      });
      subop::SubOpStateUsageTransformer transformer(analysis, getContext(), [&](mlir::Operation* op, mlir::Type type) -> mlir::Type {
         return typeConverter.convertType(type);
      });
      transformer.updateValue(createOp.getRes(), hashMapType);
      rewriter.replaceOpWithNewOp<subop::GenericCreateOp>(op, hashMapType);

      return mlir::success();
   }
};
class SpecializeSubOpPass : public mlir::PassWrapper<SpecializeSubOpPass, mlir::OperationPass<mlir::ModuleOp>> {
   bool withOptimizations;

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpecializeSubOpPass)
   virtual llvm::StringRef getArgument() const override { return "subop-specialize"; }

   SpecializeSubOpPass(bool withOptimizations) : withOptimizations(withOptimizations) {}
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<util::UtilDialect, db::DBDialect>();
   }
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      auto columnUsageAnalysis = getAnalysis<subop::ColumnUsageAnalysis>();

      mlir::RewritePatternSet patterns(&getContext());
      if (withOptimizations) {
         patterns.insert<MultiMapAsHashIndexedView>(&getContext(), columnUsageAnalysis);
      }
      patterns.insert<MapAsHashMap>(&getContext(), columnUsageAnalysis);
      patterns.insert<MultiMapAsHashMultiMap>(&getContext(), columnUsageAnalysis);

      if (mlir::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         assert(false && "should not happen");
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createSpecializeSubOpPass(bool withOptimizations) { return std::make_unique<SpecializeSubOpPass>(withOptimizations); }