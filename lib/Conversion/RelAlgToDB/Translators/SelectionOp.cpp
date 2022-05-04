#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"

class SelectionTranslator : public mlir::relalg::Translator {
   mlir::relalg::SelectionOp selectionOp;

   public:
   SelectionTranslator(mlir::relalg::SelectionOp selectionOp) : mlir::relalg::Translator(selectionOp), selectionOp(selectionOp) {}

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();

      mlir::Value matched = mergeRelationalBlock(
         builder.getInsertionBlock(), selectionOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope)[0];
      auto* parentOp = builder.getBlock()->getParentOp();
      if (mlir::isa_and_nonnull<mlir::dsa::ForOp>(parentOp)) {
         std::vector<std::pair<int, mlir::Value>> conditions;
         if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(matched.getDefiningOp())) {
            for (auto c : andOp.vals()) {
               int p = 1000;
               if (auto* defOp = c.getDefiningOp()) {
                  if (auto betweenOp = mlir::dyn_cast_or_null<mlir::db::BetweenOp>(defOp)) {
                     auto t = betweenOp.val().getType();
                     p = ::llvm::TypeSwitch<mlir::Type, int>(t)
                            .Case<::mlir::IntegerType>([&](::mlir::IntegerType t) { return 1; })
                            .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) { return 2; })
                            .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) { return 3; })
                            .Case<::mlir::db::CharType, ::mlir::db::TimestampType, ::mlir::db::IntervalType, ::mlir::FloatType>([&](mlir::Type t) { return 2; })
                            .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) { return 10; })
                            .Default([](::mlir::Type) { return 100; });
                     p -= 1;
                  } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::relalg::CmpOpInterface>(defOp)) {
                     auto t = cmpOp.getLeft().getType();
                     p = ::llvm::TypeSwitch<mlir::Type, int>(t)
                            .Case<::mlir::IntegerType>([&](::mlir::IntegerType t) { return 1; })
                            .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) { return 2; })
                            .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) { return 3; })
                            .Case<::mlir::db::CharType, ::mlir::db::TimestampType, ::mlir::db::IntervalType, ::mlir::FloatType>([&](mlir::Type t) { return 2; })
                            .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) { return 10; })
                            .Default([](::mlir::Type) { return 100; });
                  }
                  conditions.push_back({p, c});
               }
            }
         } else {
            conditions.push_back({0, matched});
         }
         std::sort(conditions.begin(), conditions.end(), [](auto a, auto b) { return a.first < b.first; });
         for (auto c : conditions) {
            auto truth = builder.create<mlir::db::DeriveTruth>(selectionOp.getLoc(), c.second);
            auto negated = builder.create<mlir::db::NotOp>(selectionOp.getLoc(), truth);
            builder.create<mlir::dsa::CondSkipOp>(selectionOp->getLoc(), negated, mlir::ValueRange{});
         }
         consumer->consume(this, builder, context);
      } else {
         matched = builder.create<mlir::db::DeriveTruth>(selectionOp.getLoc(), matched);
         builder.create<mlir::scf::IfOp>(
            selectionOp->getLoc(), mlir::TypeRange{}, matched, [&](mlir::OpBuilder& builder1, mlir::Location) {
               consumer->consume(this, builder1, context);
               builder1.create<mlir::scf::YieldOp>(selectionOp->getLoc()); });
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