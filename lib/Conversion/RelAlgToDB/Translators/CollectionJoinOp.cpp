#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/HashJoinTranslator.h>
#include <mlir/IR/BlockAndValueMapping.h>

class HashCollectionJoinTranslator : public mlir::relalg::JoinImpl {
   size_t vectorBuilderId;
   mlir::relalg::OrderedAttributes attrs;

   public:
   HashCollectionJoinTranslator(mlir::relalg::CollectionJoinOp collectionJoinOp) : mlir::relalg::JoinImpl(collectionJoinOp, collectionJoinOp.right(), collectionJoinOp.left()) {
      attrs = mlir::relalg::OrderedAttributes::fromRefArr(collectionJoinOp.attrs());
   }
   virtual void addAdditionalRequiredAttributes() override {
      for (const auto* attr : attrs.getAttrs()) {
         translator->requiredAttributes.insert(attr);
      }
   }
   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      mlir::Value vectorBuilder = context.builders[vectorBuilderId];
      auto ifOp = builder.create<mlir::db::IfOp>(
         loc, mlir::TypeRange{vectorBuilder.getType()}, matched, [&](mlir::OpBuilder& builder, mlir::Location loc) {
            mlir::Value packed = attrs.pack(context,builder,loc);
         mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(loc, vectorBuilder.getType(), vectorBuilder, packed);
         builder.create<mlir::db::YieldOp>(loc, mlir::ValueRange{mergedBuilder}); }, [&](mlir::OpBuilder& builder, mlir::Location loc) { builder.create<mlir::db::YieldOp>(loc, mlir::ValueRange{vectorBuilder}); });
      context.builders[vectorBuilderId] = ifOp.getResult(0);
   }

   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      mlir::Value vectorBuilder = builder.create<mlir::db::CreateVectorBuilder>(joinOp.getLoc(), mlir::db::VectorBuilderType::get(builder.getContext(), attrs.getTupleType(builder.getContext())));
      vectorBuilderId = context.getBuilderId();
      context.builders[vectorBuilderId] = vectorBuilder;
      translator->customLookupBuilders.push_back(vectorBuilderId);
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value vector = builder.create<mlir::db::BuilderBuild>(joinOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), attrs.getTupleType(builder.getContext())), context.builders[vectorBuilderId]);

      context.setValueForAttribute(scope, &cast<mlir::relalg::CollectionJoinOp>(joinOp).collAttr().getRelationalAttribute(), vector);
      translator->forwardConsume(builder, context);
      builder.create<mlir::db::FreeOp>(loc, vector);
   }
   virtual ~HashCollectionJoinTranslator() {}
};
std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createCollectionJoinTranslator(mlir::relalg::CollectionJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::Translator>) std::make_unique<mlir::relalg::HashJoinTranslator>(std::make_shared<HashCollectionJoinTranslator>(joinOp));
         }
      }
   }
   assert(false && "not implemented");
   return std::unique_ptr<mlir::relalg::Translator>();
}