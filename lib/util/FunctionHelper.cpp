#include "mlir/Dialect/util/FunctionHelper.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/Dialect/util/UtilOps.h"
static mlir::Value convertValue(mlir::OpBuilder& builder, mlir::Value v, mlir::Type t) {
   if (v.getType() == t) return v;
   mlir::Type currentType = v.getType();
   if (currentType.isIndex() || t.isIndex()) {
      return builder.create<mlir::arith::IndexCastOp>(builder.getUnknownLoc(), t, v);
   }
   if (currentType.isa<mlir::IntegerType>() && t.isa<mlir::IntegerType>()) {
      auto targetWidth = t.cast<mlir::IntegerType>().getWidth();
      auto sourceWidth = currentType.cast<mlir::IntegerType>().getWidth();
      if (targetWidth > sourceWidth) {
         return builder.create<mlir::arith::ExtSIOp>(builder.getUnknownLoc(), t, v);
      } else {
         return builder.create<mlir::arith::TruncIOp>(builder.getUnknownLoc(), t, v);
      }
   }
   if(t.isa<mlir::util::RefType>()&&currentType.isa<mlir::util::RefType>()){
      return builder.create<mlir::util::GenericMemrefCastOp>(builder.getUnknownLoc(),t,v);
   }
   return v; //todo
}
mlir::ResultRange mlir::util::FunctionHelper::call(OpBuilder& builder, mlir::Location loc, const FunctionSpec& function, ValueRange values) {
   auto fnHelper = builder.getContext()->getLoadedDialect<mlir::util::UtilDialect>()->getFunctionHelper();
   FuncOp funcOp = fnHelper.parentModule.lookupSymbol<mlir::FuncOp>(function.getMangledName());
   if (!funcOp) {
      OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointToStart(fnHelper.parentModule.getBody());
      funcOp = builder.create<FuncOp>(fnHelper.parentModule.getLoc(), function.getMangledName(), builder.getFunctionType(function.getParameterTypes()(builder.getContext()), function.getResultTypes()(builder.getContext())), builder.getStringAttr("private"));
      if (function.isNoSideEffects()) {
         funcOp->setAttr("const", builder.getUnitAttr());
      }
   }
   assert(values.size() == funcOp.getType().getNumInputs());
   std::vector<mlir::Value> convertedValues;
   for (size_t i = 0; i < funcOp.getType().getNumInputs(); i++) {
      mlir::Value converted = convertValue(builder, values[i], funcOp.getType().getInput(i));
      convertedValues.push_back(converted);
      assert(converted.getType() == funcOp.getType().getInput(i));
   }
   auto funcCall = builder.create<func::CallOp>(loc, funcOp, convertedValues);
   return funcCall.getResults();
}
void mlir::util::FunctionHelper::setParentModule(const mlir::ModuleOp& parentModule) {
   FunctionHelper::parentModule = parentModule;
}

std::function<mlir::ResultRange(mlir::ValueRange)> mlir::util::FunctionSpec::operator()(mlir::OpBuilder& builder, mlir::Location loc) const {
   std::function<mlir::ResultRange(mlir::ValueRange)> fn = [&builder, loc, this](mlir::ValueRange range) -> mlir::ResultRange { return mlir::util::FunctionHelper::call(builder, loc, *this, range); };
   return fn;
}
mlir::util::FunctionSpec::FunctionSpec(const std::string& name, const std::string& mangledName, const std::function<std::vector<mlir::Type>(mlir::MLIRContext*)>& parameterTypes, const std::function<std::vector<mlir::Type>(mlir::MLIRContext*)>& resultTypes, bool noSideEffects) : name(name), mangledName(mangledName), parameterTypes(parameterTypes), resultTypes(resultTypes), noSideEffects(noSideEffects) {}
