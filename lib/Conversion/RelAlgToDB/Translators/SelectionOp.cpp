#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class SelectionTranslator : public mlir::relalg::Translator {
   mlir::relalg::SelectionOp selectionOp;

   public:
   SelectionTranslator(mlir::relalg::SelectionOp selectionOp) : mlir::relalg::Translator(selectionOp), selectionOp(selectionOp) {
   }

   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();

      mlir::Value matched = mergeRelationalBlock(
         builder.getInsertionBlock(), selectionOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope)[0];
      auto *parentOp = builder.getBlock()->getParentOp();
      if (mlir::isa_and_nonnull<mlir::db::ForOp>(parentOp)) {
         std::vector<std::pair<int,mlir::Value>> conditions;
         if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(matched.getDefiningOp())) {
            for (auto c : andOp.vals()) {
               int p=1000;
               if(auto *defOp=c.getDefiningOp()){
                  if(auto cmpOp=mlir::dyn_cast_or_null<mlir::db::CmpOp>(defOp)){
                     auto t=cmpOp.left().getType();
                     p= ::llvm::TypeSwitch<mlir::Type, int>(t)
                        .Case<::mlir::db::BoolType>([&](::mlir::db::BoolType t) {
                           return 1;
                        })
                        .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) {
                           return 2;
                        })
                        .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
                           return 3;
                        })
                        .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
                           return 2;
                        })
                        .Case<::mlir::db::UIntType>([&](::mlir::db::UIntType t) {
                           return 2;
                        })
                        .Case<::mlir::db::CharType>([&](::mlir::db::CharType t) {
                           return 2;
                        })
                        .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
                           return 10;
                        })
                        .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
                           return 2;
                        })
                        .Case<::mlir::db::IntervalType>([&](::mlir::db::IntervalType t) {
                           return 2;
                        })
                        .Case<::mlir::db::FloatType>([&](::mlir::db::FloatType t) {
                           return 2;
                        })
                        .Case<::mlir::db::DurationType>([&](::mlir::db::DurationType t) {
                           return 2;
                        })
                        .Case<::mlir::db::TimeType>([&](::mlir::db::TimeType t) {
                           return 2;
                        })
                        .Default([](::mlir::Type) { return 100; });
                  }
               }
               conditions.push_back({p,c});
            }
         } else {
            conditions.push_back({0,matched});
         }
         std::sort(conditions.begin(),conditions.end(),[](auto a,auto b){return a.first<b.first;});
         for (auto c : conditions) {
            auto negated = builder.create<mlir::db::NotOp>(selectionOp.getLoc(), c.second);
            builder.create<mlir::db::CondSkipOp>(selectionOp->getLoc(), negated, getRequiredBuilderValues(context));
         }
         consumer->consume(this, builder, context);
      } else {
         auto builderValuesBefore= getRequiredBuilderValues(context);
         auto ifOp = builder.create<mlir::db::IfOp>(
            selectionOp->getLoc(), getRequiredBuilderTypes(context), matched, [&](mlir::OpBuilder& builder1, mlir::Location) {
               consumer->consume(this, builder1, context);
               builder1.create<mlir::db::YieldOp>(selectionOp->getLoc(), getRequiredBuilderValues(context)); },
            requiredBuilders.empty() ? noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location loc) { builder2.create<mlir::db::YieldOp>(loc, builderValuesBefore); });
         setRequiredBuilderValues(context, ifOp.getResults());
      }
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~SelectionTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createSelectionTranslator(mlir::relalg::SelectionOp selectionOp) {
   return std::make_unique<SelectionTranslator>(selectionOp);
}