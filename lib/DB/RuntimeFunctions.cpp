#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/Dialect/DB/Passes.h"

mlir::db::RuntimeFunction* mlir::db::RuntimeFunctionRegistry::lookup(std::string name) {
   return registeredFunctions[name].get();
}

std::shared_ptr<mlir::db::RuntimeFunctionRegistry> mlir::db::RuntimeFunctionRegistry::getBuiltinRegistry(mlir::MLIRContext* context) {
   auto builtinRegistry = std::make_shared<RuntimeFunctionRegistry>(context);
   auto testFn = std::make_unique<mlir::db::RuntimeFunction>();
   testFn->nullHandleType = RuntimeFunction::NeedsWrapping;
   testFn->verifyFn = [](mlir::TypeRange types, mlir::Type resType) -> bool {
      return types.size() == 1 && types[0].isIntOrIndex() && types[0] == resType;
   };
   testFn->name = "test";
   builtinRegistry->registeredFunctions.insert(std::make_pair("test", std::move(testFn)));

   auto dumpFn = std::make_unique<mlir::db::RuntimeFunction>();
   dumpFn->nullHandleType = RuntimeFunction::HandlesNulls;
   dumpFn->verifyFn = [](mlir::TypeRange types, mlir::Type resType) -> bool {
      return !resType && types.size() == 1 && (getBaseType(types[0]).isIntOrIndexOrFloat() || getBaseType(types[0]).isa<mlir::db::StringType, mlir::db::DecimalType, mlir::db::TimestampType, mlir::db::DateType, mlir::db::IntervalType, mlir::db::CharType>());
   };
   dumpFn->name = "DumpValue";
   builtinRegistry->registeredFunctions.insert(std::make_pair("DumpValue", std::move(dumpFn)));

   auto subStringFn = std::make_unique<mlir::db::RuntimeFunction>();
   subStringFn->nullHandleType = RuntimeFunction::NeedsWrapping;
   subStringFn->verifyFn = [](mlir::TypeRange types, mlir::Type resType) -> bool {
      return resType && resType.isa<mlir::db::StringType>() && types.size() == 3 &&
         getBaseType(types[0]).isa<mlir::db::StringType>() && getBaseType(types[1]).isIntOrIndex() && getBaseType(types[2]).isIntOrIndex();
   };
   subStringFn->name = "Substring";
   builtinRegistry->registeredFunctions.insert(std::make_pair("Substring", std::move(subStringFn)));
   return builtinRegistry;
}