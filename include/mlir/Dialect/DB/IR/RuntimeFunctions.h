#ifndef MLIR_DIALECT_DB_IR_RUNTIMEFUNCTIONS_H
#define MLIR_DIALECT_DB_IR_RUNTIMEFUNCTIONS_H
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/IR/Builders.h"
#include <memory>
#include <string>
#include <unordered_map>
namespace mlir::db {

struct RuntimeFunction {
   enum NullHandleType {
      HandlesNulls,
      HandlesInvalidVaues,
      NeedsWrapping
   };
   std::string name;
   NullHandleType nullHandleType;
   bool hasSideEffects;
   class LoweringImpl {
      public:
      virtual mlir::Value lower(mlir::OpBuilder& builder, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType) = 0;
      virtual ~LoweringImpl(){}
   };

   public:
   std::function<bool(mlir::TypeRange types, mlir::Type resType)> verifyFn;
   std::unique_ptr<LoweringImpl> lowering;
};
class RuntimeFunctionRegistry {
   std::unordered_map<std::string, std::unique_ptr<RuntimeFunction>> registeredFunctions;

   public:
   RuntimeFunctionRegistry(MLIRContext* context) {}

   public:
   static std::shared_ptr<RuntimeFunctionRegistry> getBuiltinRegistry(mlir::MLIRContext* context);
   mlir::Type unwrapNullable(RuntimeFunction::NullHandleType nullHandleType, mlir::Type t) {
      if (!t) return t;
      if (nullHandleType == RuntimeFunction::HandlesNulls) return t;
      return getBaseType(t);
   }
   std::vector<Type> unwrapNullables(RuntimeFunction::NullHandleType nullHandleType, mlir::TypeRange ts) {
      std::vector<Type> res;
      for (auto t : ts) {
         res.push_back(unwrapNullable(nullHandleType, t));
      }
      return res;
   }
   bool verify(std::string name, mlir::TypeRange types, mlir::Type resType) {
      auto *fn = lookup(name);
      if (!fn) return false;
      return fn->verifyFn(unwrapNullables(fn->nullHandleType, types), unwrapNullable(fn->nullHandleType, resType));
   }
   RuntimeFunction* lookup(std::string name);
};
} // end namespace mlir::db

#endif // MLIR_DIALECT_DB_IR_RUNTIMEFUNCTIONS_H
