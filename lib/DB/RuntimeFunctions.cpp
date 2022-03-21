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
   auto dateExtractFn = std::make_unique<mlir::db::RuntimeFunction>();
   dateExtractFn->nullHandleType = RuntimeFunction::NeedsWrapping;
   dateExtractFn->verifyFn = [](mlir::TypeRange types, mlir::Type resType) -> bool {
      return resType && resType.isInteger(64) && types.size() == 2 &&
         getBaseType(types[0]).isa<mlir::db::StringType, mlir::db::CharType>() && getBaseType(types[1]).isa<mlir::db::DateType>();
   };
   dateExtractFn->name = "ExtractFromDate";
   builtinRegistry->registeredFunctions.insert(std::make_pair("ExtractFromDate", std::move(dateExtractFn)));
   std::vector<std::string> extractable = {"Year", "Month", "Day"};
   for (auto e : extractable) {
      auto dateExtractFn = std::make_unique<mlir::db::RuntimeFunction>();
      dateExtractFn->nullHandleType = RuntimeFunction::HandlesInvalidVaues;
      dateExtractFn->verifyFn = [](mlir::TypeRange types, mlir::Type resType) -> bool {
         return resType && resType.isInteger(64) && types.size() == 1 && getBaseType(types[0]).isa<mlir::db::DateType>();
      };
      dateExtractFn->name = "Extract" + e + "FromDate";
      builtinRegistry->registeredFunctions.insert(std::make_pair(dateExtractFn->name, std::move(dateExtractFn)));
   }
   auto dateAddFn = std::make_unique<mlir::db::RuntimeFunction>();
   dateAddFn->nullHandleType = RuntimeFunction::HandlesInvalidVaues;
   dateAddFn->verifyFn = [](mlir::TypeRange types, mlir::Type resType) -> bool {
      return resType && resType.isa<mlir::db::DateType>() && types.size() == 2 &&
         getBaseType(types[0]).isa<mlir::db::DateType>() && getBaseType(types[1]).isa<mlir::db::IntervalType>();
   };
   dateAddFn->name = "DateAdd";
   builtinRegistry->registeredFunctions.insert(std::make_pair("DateAdd", std::move(dateAddFn)));
   auto dateSubFn = std::make_unique<mlir::db::RuntimeFunction>();
   dateSubFn->nullHandleType = RuntimeFunction::HandlesInvalidVaues;
   dateSubFn->verifyFn = [](mlir::TypeRange types, mlir::Type resType) -> bool {
      return resType && resType.isa<mlir::db::DateType>() && types.size() == 2 &&
         getBaseType(types[0]).isa<mlir::db::DateType>() && getBaseType(types[1]).isa<mlir::db::IntervalType>();
   };
   dateSubFn->name = "DateSubtract";
   builtinRegistry->registeredFunctions.insert(std::make_pair("DateSubtract", std::move(dateSubFn)));
   return builtinRegistry;
}