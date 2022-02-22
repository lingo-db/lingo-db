#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/HashJoinTranslator.h>
#include <mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>
#include <mlir/IR/BlockAndValueMapping.h>

class MarkJoinImpl : public mlir::relalg::JoinImpl {
   public:
   MarkJoinImpl(mlir::relalg::MarkJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::db::SetFlag>(loc, matchFoundFlag, matched);
   }

   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(loc, mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(loc, mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      context.setValueForAttribute(scope, &cast<mlir::relalg::MarkJoinOp>(joinOp).markattr().getRelationalAttribute(), matchFound);
      translator->forwardConsume(builder, context);
   }
   virtual ~MarkJoinImpl() {}
};

std::shared_ptr<mlir::relalg::JoinImpl> mlir::relalg::Translator::createMarkJoinImpl(mlir::relalg::MarkJoinOp joinOp) {
   return std::make_shared<MarkJoinImpl>(joinOp);
}