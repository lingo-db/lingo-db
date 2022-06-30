#ifndef MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H
#define MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H
#include "llvm/ADT/ScopedHashTable.h"

#include "mlir/Dialect/RelAlg/IR/Column.h"
#include "mlir/IR/Value.h"
namespace mlir {
namespace relalg {
class TranslatorContext {
   llvm::ScopedHashTable<const mlir::relalg::Column*, mlir::Value> symbolTable;

   public:
   using AttributeResolverScope = llvm::ScopedHashTableScope<const mlir::relalg::Column*, mlir::Value>;

   mlir::Value getValueForAttribute(const mlir::relalg::Column* attribute) const {
      if (!symbolTable.lookup(attribute)) {
         assert(symbolTable.count(attribute));
      }

      return symbolTable.lookup(attribute);
   }
   mlir::Value getUnsafeValueForAttribute(const mlir::relalg::Column* attribute) const {
      return symbolTable.lookup(attribute);
   }
   void setValueForAttribute(AttributeResolverScope& scope, const mlir::relalg::Column* iu, mlir::Value v) {
      symbolTable.insertIntoScope(&scope, iu, v);
   }
   AttributeResolverScope createScope() {
      return AttributeResolverScope(symbolTable);
   }
   std::unordered_map<size_t, mlir::Value> builders;

   std::unordered_map<mlir::Operation*, std::pair<mlir::Value, std::vector<const mlir::relalg::Column*>>> materializedTmp;
};
} // end namespace relalg
} // end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H
