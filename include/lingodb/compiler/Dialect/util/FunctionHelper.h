#ifndef LINGODB_COMPILER_DIALECT_UTIL_FUNCTIONHELPER_H
#define LINGODB_COMPILER_DIALECT_UTIL_FUNCTIONHELPER_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
namespace lingodb::compiler::dialect::util {
template <class T>
static void* getPtr(T x) {
   return *reinterpret_cast<void**>(&x);
}
class FunctionSpec {
   std::string name;
   std::string mangledName;
   std::function<std::vector<mlir::Type>(mlir::MLIRContext*)> parameterTypes;
   std::function<std::vector<mlir::Type>(mlir::MLIRContext*)> resultTypes;

   public:
   void* (*getPointer)();

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

   FunctionSpec(const std::string& name, const std::string& mangledName, const std::function<std::vector<mlir::Type>(mlir::MLIRContext*)>& parameterTypes, const std::function<std::vector<mlir::Type>(mlir::MLIRContext*)>& resultTypes, void* (*getPointer)());

   std::function<mlir::ResultRange(mlir::ValueRange)> operator()(mlir::OpBuilder& builder, mlir::Location loc) const;
};

class FunctionHelper {
   mlir::ModuleOp parentModule;
   public:
   static void visitAllFunctions(const std::function<void(std::string, void*)>& fn);
   void setParentModule(const mlir::ModuleOp& parentModule);

   public:
   static mlir::func::CallOp call (mlir::OpBuilder& builder, mlir::Location loc, const FunctionSpec& function, mlir::ValueRange values);
   static mlir::Value convertValue(mlir::OpBuilder& builder, mlir::Value v, mlir::Type t, mlir::Location loc);
};
} // namespace lingodb::compiler::dialect::util

#endif //LINGODB_COMPILER_DIALECT_UTIL_FUNCTIONHELPER_H
