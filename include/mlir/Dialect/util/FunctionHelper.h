#ifndef MLIR_DIALECT_UTIL_FUNCTIONHELPER_H
#define MLIR_DIALECT_UTIL_FUNCTIONHELPER_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
namespace mlir::util {
class FunctionSpec {
   std::string name;
   std::string mangledName;
   std::function<std::vector<mlir::Type>(mlir::MLIRContext*)> parameterTypes;
   std::function<std::vector<mlir::Type>(mlir::MLIRContext*)> resultTypes;
   bool noSideEffects;

   public:
   const std::string& getName() const {
      return name;
   }
   const std::string& getMangledName() const {
      return mangledName;
   }
   const std::function<std::vector<mlir::Type>(mlir::MLIRContext*)>& getParameterTypes() const {
      return parameterTypes;
   }
   const std::function<std::vector<mlir::Type>(mlir::MLIRContext*)>& getResultTypes() const {
      return resultTypes;
   }
   FunctionSpec(const std::string& name, const std::string& mangledName, const std::function<std::vector<mlir::Type>(mlir::MLIRContext*)>& parameterTypes, const std::function<std::vector<mlir::Type>(mlir::MLIRContext*)>& resultTypes, bool noSideEffects);

   std::function<mlir::ResultRange(mlir::ValueRange)> operator()(mlir::OpBuilder& builder, mlir::Location loc) const;
   bool isNoSideEffects() const {
      return noSideEffects;
   }
};

class FunctionHelper {
   ModuleOp parentModule;

   public:
   void setParentModule(const ModuleOp& parentModule);

   public:
   static mlir::ResultRange call(OpBuilder& builder, mlir::Location loc, const FunctionSpec& function, ValueRange values);
};
} // namespace mlir::util

#endif // MLIR_DIALECT_UTIL_FUNCTIONHELPER_H
