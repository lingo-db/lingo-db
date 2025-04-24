#pragma once
#include <stack>

#include <mlir/IR/Value.h>

namespace lingodb::translator {
class TranslationContext;
class TupleScope {
   public:
   TupleScope(TranslationContext* context);
   ~TupleScope();
   TranslationContext* context;
};
class TranslationContext {
   public:
   TranslationContext();
   mlir::Value getCurrentTuple();
   void setCurrentTuple(mlir::Value v);
   TupleScope createTupleScope() {
      return TupleScope(this);
   }
   std::stack<mlir::Value> currTuple;

   std::map<size_t, mlir::Type> translatedValuesType;
};
} // namespace lingodb::translator