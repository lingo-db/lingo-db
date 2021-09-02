#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/HashJoinUtils.h>
#include <mlir/IR/BlockAndValueMapping.h>

class HashCollectionJoinLowering : public mlir::relalg::HJNode<mlir::relalg::CollectionJoinOp> {
   size_t vectorBuilderId;
   std::vector<mlir::Type> tupleTypes;
   std::vector<mlir::relalg::RelationalAttribute*> tupleAttributes;
   mlir::TupleType tupleType;

   public:
   HashCollectionJoinLowering(mlir::relalg::CollectionJoinOp collectionJoinOp) : mlir::relalg::HJNode<mlir::relalg::CollectionJoinOp>(collectionJoinOp, collectionJoinOp.right(), collectionJoinOp.left()) {
      for (auto attr : collectionJoinOp.attrs()) {
         if (auto refAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            tupleTypes.push_back(refAttr.getRelationalAttribute().type);
            tupleAttributes.push_back(&refAttr.getRelationalAttribute());
         }
      }
      tupleType = mlir::TupleType::get(collectionJoinOp.getContext(), tupleTypes);
   }
   virtual void addAdditionalRequiredAttributes() override {
      for (auto *attr : tupleAttributes) {
         requiredAttributes.insert(attr);
      }
   }
   virtual void handleLookup(mlir::Value matched, mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      mlir::Value vectorBuilder = context.builders[vectorBuilderId];
      std::vector<mlir::Value> values;
      for (const auto* attr : tupleAttributes) {
         values.push_back(context.getValueForAttribute(attr));
      }

      auto ifOp = builder.create<mlir::db::IfOp>(joinOp->getLoc(), mlir::TypeRange{vectorBuilder.getType()}, matched);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      mlir::relalg::ProducerConsumerBuilder builder1(ifOp.thenRegion());
      mlir::Block* elseBlock = new mlir::Block;
      ifOp.elseRegion().push_back(elseBlock);
      mlir::relalg::ProducerConsumerBuilder builder3(ifOp.elseRegion());
      builder3.create<mlir::db::YieldOp>(joinOp->getLoc(), mlir::ValueRange{vectorBuilder});
      mlir::Value packed = builder1.create<mlir::util::PackOp>(joinOp->getLoc(), tupleType, values);
      mlir::Value mergedBuilder = builder1.create<mlir::db::BuilderMerge>(joinOp->getLoc(), vectorBuilder.getType(), vectorBuilder, packed);
      builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), mlir::ValueRange{mergedBuilder});

      context.builders[vectorBuilderId] = ifOp.getResult(0);
   }

   void beforeLookup(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      mlir::Value vectorBuilder = builder.create<mlir::db::CreateVectorBuilder>(joinOp.getLoc(), mlir::db::VectorBuilderType::get(builder.getContext(), tupleType));
      vectorBuilderId = context.getBuilderId();
      context.builders[vectorBuilderId] = vectorBuilder;
      this->customLookupBuilders.push_back(vectorBuilderId);
   }
   void afterLookup(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value vector = builder.create<mlir::db::BuilderBuild>(joinOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType), context.builders[vectorBuilderId]);

      context.setValueForAttribute(scope, &joinOp.collAttr().getRelationalAttribute(), vector);
      consumer->consume(this, builder, context);
      builder.create<mlir::db::FreeOp>(joinOp->getLoc(), vector);
   }
   virtual ~HashCollectionJoinLowering() {}
};
bool mlir::relalg::ProducerConsumerNodeRegistry::registeredCollectionJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::CollectionJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashCollectionJoinLowering>(joinOp);
         }
      }
   }
   assert(false && "not implemented");
   return std::unique_ptr<mlir::relalg::ProducerConsumerNode>();
});