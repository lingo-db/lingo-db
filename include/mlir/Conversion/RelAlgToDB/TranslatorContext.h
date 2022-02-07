#ifndef MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H
#define MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H
#include "llvm/ADT/ScopedHashTable.h"

#include "mlir/Conversion/RelAlgToDB/Pipeline.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttribute.h"
#include "mlir/IR/Value.h"
namespace mlir {
namespace relalg {
class TranslatorContext {
   llvm::ScopedHashTable<const mlir::relalg::RelationalAttribute*, mlir::Value> symbolTable;

   public:
   using AttributeResolverScope = llvm::ScopedHashTableScope<const mlir::relalg::RelationalAttribute*, mlir::Value>;

   mlir::Value getValueForAttribute(const mlir::relalg::RelationalAttribute* attribute) const {
      if (!symbolTable.lookup(attribute)) {
         assert(symbolTable.count(attribute));
      }

      return symbolTable.lookup(attribute);
   }
   mlir::Value getUnsafeValueForAttribute(const mlir::relalg::RelationalAttribute* attribute) const {
      return symbolTable.lookup(attribute);
   }
   void setValueForAttribute(AttributeResolverScope& scope, const mlir::relalg::RelationalAttribute* iu, mlir::Value v) {
      symbolTable.insertIntoScope(&scope, iu, v);
   }
   AttributeResolverScope createScope() {
      return AttributeResolverScope(symbolTable);
   }
   std::unordered_map<size_t, mlir::Value> builders;
   PipelineManager pipelineManager;

   size_t getBuilderId() {
      static size_t id = 0;
      return id++;
   }
   std::unordered_map<mlir::Operation*, std::pair<PipelineDependency, std::vector<const mlir::relalg::RelationalAttribute*>>> materializedTmp;
};
} // end namespace relalg
} // end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_TRANSLATORCONTEXT_H
