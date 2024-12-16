#ifndef LINGODB_COMPILER_DIALECT_DB_IR_RUNTIMEFUNCTIONS_H
#define LINGODB_COMPILER_DIALECT_DB_IR_RUNTIMEFUNCTIONS_H

#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/util/FunctionHelper.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Builders.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
namespace lingodb::compiler::dialect::db {

struct RuntimeFunction {
   enum NullHandleType {
      HandlesNulls,
      HandlesInvalidVaues,
      NeedsWrapping
   };
   std::string name;
   NullHandleType nullHandleType;
   bool hasSideEffects;

   public:
   using TypeMatcher = std::function<bool(mlir::Type)>;
   using ResTypeMatcher = std::function<bool(mlir::Type, mlir::TypeRange)>;
   static inline auto anyType = [](mlir::Type) { return true; };
   static inline auto intLike = [](mlir::Type t) { return getBaseType(t).isIntOrIndex(); };
   static inline auto onlyIndex = [](mlir::Type t) { return t.isIndex(); };
   static inline auto stringLike = [](mlir::Type t) { return mlir::isa<StringType, CharType>(getBaseType(t)); };
   static inline auto dateLike = [](mlir::Type t) { return mlir::isa<DateType, TimestampType>(getBaseType(t)); };
   static inline auto float64 = [](mlir::Type t) { return getBaseType(t).isF64(); };
   static inline auto dateInterval = [](mlir::Type t) { return mlir::isa<IntervalType>(getBaseType(t)); };
   static inline auto anyDecimal = [](mlir::Type t) { return mlir::isa<DecimalType>(getBaseType(t)); };
   static inline auto anyNumber = [](mlir::Type t) { return intLike(t) || anyDecimal(t) || getBaseType(t).isF64(); };
   static inline auto noReturnType = [](mlir::Type t, mlir::TypeRange) { return !t; };
   static ResTypeMatcher matchesArgument(size_t argIdx = 0) {
      return [](mlir::Type resType, mlir::TypeRange types) {
         return resType == types[0];
      };
   }
   std::function<bool(mlir::TypeRange types, mlir::Type resType)> verifyFn;
   std::optional<std::function<mlir::LogicalResult(mlir::TypeRange, ::llvm::ArrayRef<::mlir::Attribute>, ::llvm::SmallVectorImpl<::mlir::OpFoldResult>&)>> foldFn;
   using loweringFnT = std::function<mlir::Value(mlir::OpBuilder& builder, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter*, mlir::Location)>;
   std::variant<loweringFnT, lingodb::compiler::dialect::util::FunctionSpec> implementation;

   //builder functions
   RuntimeFunction& implementedAs(lingodb::compiler::dialect::util::FunctionSpec function) {
      implementation = function;
      nullHandleType = NeedsWrapping;
      return *this;
   }
   RuntimeFunction& implementedAs(loweringFnT function) {
      implementation = function;
      return *this;
   }
   RuntimeFunction& handlesNulls() {
      nullHandleType = HandlesNulls;
      return *this;
   }
   RuntimeFunction& handlesInvalid() {
      nullHandleType = HandlesInvalidVaues;
      return *this;
   }
   RuntimeFunction& needsWrapping() {
      nullHandleType = NeedsWrapping;
      return *this;
   }
   RuntimeFunction& folds(std::function<mlir::LogicalResult(mlir::TypeRange typeRange, ::llvm::ArrayRef<::mlir::Attribute>, ::llvm::SmallVectorImpl<::mlir::OpFoldResult>&)> foldFn) {
      this->foldFn = foldFn;
      return *this;
   }
   RuntimeFunction& matchesTypes(const std::vector<TypeMatcher>& matchers, ResTypeMatcher resMatcher) {
      verifyFn = [matchers = matchers, resMatcher = resMatcher](mlir::TypeRange types, mlir::Type resType) -> bool {
         if (types.size() != matchers.size()) return false;
         for (size_t i = 0; i < types.size(); i++) {
            if (!matchers[i](types[i])) {
               return false;
            }
         }
         return resMatcher(resType, types);
      };

      return *this;
   }
};

class RuntimeFunctionRegistry {
   std::unordered_map<std::string, std::unique_ptr<RuntimeFunction>> registeredFunctions;

   public:
   RuntimeFunctionRegistry(mlir::MLIRContext* context) {}

   public:
   static std::shared_ptr<RuntimeFunctionRegistry> getBuiltinRegistry(mlir::MLIRContext* context);
   mlir::Type unwrapNullable(RuntimeFunction::NullHandleType nullHandleType, mlir::Type t) {
      if (!t) return t;
      if (nullHandleType == RuntimeFunction::HandlesNulls) return t;
      return getBaseType(t);
   }
   std::vector<mlir::Type> unwrapNullables(RuntimeFunction::NullHandleType nullHandleType, mlir::TypeRange ts) {
      std::vector<mlir::Type> res;
      for (auto t : ts) {
         res.push_back(unwrapNullable(nullHandleType, t));
      }
      return res;
   }
   bool verify(std::string name, mlir::TypeRange types, mlir::Type resType) {
      auto* fn = lookup(name);
      if (!fn) return false;
      return fn->verifyFn(unwrapNullables(fn->nullHandleType, types), unwrapNullable(fn->nullHandleType, resType));
   }
   RuntimeFunction* lookup(std::string name);
   RuntimeFunction& add(std::string name) {
      registeredFunctions[name] = std::make_unique<RuntimeFunction>();
      registeredFunctions[name]->name = name;
      return *registeredFunctions[name];
   }
};
} // end namespace lingodb::compiler::dialect::db

#endif //LINGODB_COMPILER_DIALECT_DB_IR_RUNTIMEFUNCTIONS_H
