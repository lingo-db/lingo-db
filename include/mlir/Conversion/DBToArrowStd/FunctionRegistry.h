#ifndef MLIR_CONVERSION_DBTOARROWSTD_FUNCTIONREGISTRY_H
#define MLIR_CONVERSION_DBTOARROWSTD_FUNCTIONREGISTRY_H
#include "RegisteredFunctions.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"

#include <iostream>
namespace mlir::db::codegen {
class FunctionRegistry {
   public:
   enum class FunctionId : uint32_t {
#define OPERANDS_(...)
#define RETURNS_(...)
#define DECLARE_FUNCID(inst, ...) inst,
      FUNC_LIST(DECLARE_FUNCID, OPERANDS_, RETURNS_)
#undef DECLARE_FUNCID
#undef OPERANDS_
#undef RETURNS_
   };
   FunctionRegistry(ModuleOp parentModule) : context(parentModule->getContext()), parentModule(parentModule) {
   }
   FuncOp getFunction(OpBuilder builder, FunctionId function);
   ResultRange call(OpBuilder builder, mlir::Location loc, FunctionId function, ValueRange values);
   void registerFunctions();

   private:
   struct RegisteredFunction {
      std::string name;
      bool useWrapper;
      std::vector<Type> operands;
      std::vector<Type> results;
   };
   std::vector<RegisteredFunction> registeredFunctions;
   std::vector<FuncOp> insertedFunctions;
   MLIRContext* context;
   ModuleOp parentModule;

   public:
   RegisteredFunction& getRegisteredFunction(FunctionId functionId);

   private:
   FuncOp insertFunction(OpBuilder builder, RegisteredFunction& function);
   void registerFunction(FunctionId funcId, std::string name, std::vector<mlir::Type> ops, std::vector<mlir::Type> returns, bool useWrapper = true);
};
} // namespace mlir::db::codegen
#endif // MLIR_CONVERSION_DBTOARROWSTD_FUNCTIONREGISTRY_H
