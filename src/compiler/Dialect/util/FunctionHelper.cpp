#include "lingodb/compiler/Dialect/util/FunctionHelper.h"

#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/Dialect/util/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace {
std::unordered_map<std::string, lingodb::compiler::dialect::util::FunctionSpec>& getFunctions() {
   static std::unordered_map<std::string, lingodb::compiler::dialect::util::FunctionSpec> functions;

   return functions;
}
} // end namespace
mlir::Value lingodb::compiler::dialect::util::FunctionHelper::convertValue(mlir::OpBuilder& builder, mlir::Value v, mlir::Type t, mlir::Location loc) {
   if (v.getType() == t) return v;
   mlir::Type currentType = v.getType();
   if (currentType.isIndex() || t.isIndex()) {
      return builder.create<mlir::arith::IndexCastOp>(loc, t, v);
   }
   if (mlir::isa<mlir::IntegerType>(currentType) && mlir::isa<mlir::IntegerType>(t)) {
      auto targetWidth = mlir::cast<mlir::IntegerType>(t).getWidth();
      auto sourceWidth = mlir::cast<mlir::IntegerType>(currentType).getWidth();
      if (targetWidth > sourceWidth) {
         return builder.create<mlir::arith::ExtSIOp>(loc, t, v);
      } else {
         return builder.create<mlir::arith::TruncIOp>(loc, t, v);
      }
   }
   if (mlir::isa<RefType>(t) && mlir::isa<RefType>(currentType)) {
      return builder.create<GenericMemrefCastOp>(loc, t, v);
   }
   if (mlir::isa<BufferType>(t) && mlir::isa<BufferType>(currentType)) {
      return builder.create<BufferCastOp>(loc, t, v);
   }
   return v; //todo
}
mlir::func::CallOp lingodb::compiler::dialect::util::FunctionHelper::call(mlir::OpBuilder& builder, mlir::Location loc, const FunctionSpec& function, mlir::ValueRange values) {
   auto fnHelper = builder.getContext()->getLoadedDialect<UtilDialect>()->getFunctionHelper();
   mlir::func::FuncOp funcOp = fnHelper.parentModule.lookupSymbol<mlir::func::FuncOp>(function.getMangledName());
   if (!funcOp) {
      mlir::OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointToStart(fnHelper.parentModule.getBody());
      funcOp = builder.create<mlir::func::FuncOp>(fnHelper.parentModule.getLoc(), function.getMangledName(), builder.getFunctionType(function.getParameterTypes()(builder.getContext()), function.getResultTypes()(builder.getContext())), builder.getStringAttr("private"), mlir::ArrayAttr{}, mlir::ArrayAttr{});
   }
   assert(values.size() == funcOp.getFunctionType().getNumInputs());
   std::vector<mlir::Value> convertedValues;
   for (size_t i = 0; i < funcOp.getFunctionType().getNumInputs(); i++) {
      mlir::Value converted = convertValue(builder, values[i], funcOp.getFunctionType().getInput(i), loc);
      convertedValues.push_back(converted);
      assert(converted.getType() == funcOp.getFunctionType().getInput(i));
   }
   auto funcCall = builder.create<mlir::func::CallOp>(loc, funcOp, convertedValues);
   return funcCall;}
void lingodb::compiler::dialect::util::FunctionHelper::setParentModule(const mlir::ModuleOp& parentModule) {
   FunctionHelper::parentModule = parentModule;
}

std::function<mlir::ResultRange(mlir::ValueRange)> lingodb::compiler::dialect::util::FunctionSpec::operator()(mlir::OpBuilder& builder, mlir::Location loc) const {
   std::function<mlir::ResultRange(mlir::ValueRange)> fn = [&builder, loc, this](mlir::ValueRange range) -> mlir::ResultRange { return lingodb::compiler::dialect::util::FunctionHelper::call(builder, loc, *this, range).getResults(); };
   return fn;
}

lingodb::compiler::dialect::util::FunctionSpec::FunctionSpec(const std::string& name, const std::string& mangledName, const std::function<std::vector<mlir::Type>(mlir::MLIRContext*)>& parameterTypes, const std::function<std::vector<mlir::Type>(mlir::MLIRContext*)>& resultTypes, void* (*getPointer)()) : name(name), mangledName(mangledName), parameterTypes(parameterTypes), resultTypes(resultTypes), getPointer(getPointer) {
   getFunctions().insert({mangledName, *this});
}

void lingodb::compiler::dialect::util::FunctionHelper::visitAllFunctions(const std::function<void(std::string, void*)>& fn) {
   for (auto f : getFunctions()) {
      fn(f.first, f.second.getPointer());
   }
}