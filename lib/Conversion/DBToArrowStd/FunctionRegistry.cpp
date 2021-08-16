#include "mlir/Conversion/DBToArrowStd/FunctionRegistry.h"
#include "mlir/Dialect/util/UtilOps.h"

void mlir::db::codegen::FunctionRegistry::registerFunctions() {
#define INT_TYPE(W) IntegerType::get(context, W)
#define FLOAT_TYPE FloatType::getF32(context)
#define DOUBLE_TYPE FloatType::getF64(context)

#define BOOL_TYPE INT_TYPE(1)

#define INDEX_TYPE IndexType::get(context)

#define POINTER_TYPE mlir::util::GenericMemrefType::get(context,IntegerType::get(context, 8),llvm::Optional<int64_t>())
#define STRING_TYPE mlir::util::GenericMemrefType::get(context,IntegerType::get(context, 8),-1)
#define TUPLE_TYPE(...) TupleType::get(context, TypeRange({__VA_ARGS__}))
#define OPERANDS_(...)  { __VA_ARGS__ }
#define RETURNS_(...)  { __VA_ARGS__ }
#define FUNCTION_TYPE(operands,returns)  FunctionType::get(context,operands,returns)
#define REGISTER_FUNC(inst, name, operands, returns) registerFunction(FunctionId::inst, #name, operands, returns);
   FUNC_LIST(REGISTER_FUNC, OPERANDS_, RETURNS_)
#undef REGISTER_FUNC
#define REGISTER_FUNC(inst, name, operands, returns) registerFunction(FunctionId::inst, #name, operands, returns,false);
   PLAIN_FUNC_LIST(REGISTER_FUNC, OPERANDS_, RETURNS_)
#undef REGISTER_FUNC
#undef RETURNS_
#undef OPERANDS_
}
mlir::FuncOp mlir::db::codegen::FunctionRegistry::insertFunction(mlir::OpBuilder builder, mlir::db::codegen::FunctionRegistry::RegisteredFunction& function){
   ModuleOp parentModule = builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
   OpBuilder::InsertionGuard insertionGuard(builder);
   builder.setInsertionPointToStart(parentModule.getBody());
   FuncOp funcOp = builder.create<FuncOp>(parentModule.getLoc(), function.useWrapper?"_mlir_ciface_"+function.name:function.name, builder.getFunctionType(function.operands, function.results), builder.getStringAttr("private"));
   return funcOp;
}

mlir::FuncOp mlir::db::codegen::FunctionRegistry::getFunction(OpBuilder builder, FunctionId function) {
   size_t offset = static_cast<size_t>(function);
   if (insertedFunctions.size() > offset && insertedFunctions[offset]) {
      return insertedFunctions[offset];
   }
   if (registeredFunctions.size() > offset && !registeredFunctions[offset].name.empty()) {
      FuncOp inserted = insertFunction(builder, registeredFunctions[offset]);
      insertedFunctions[offset] = inserted;
      return inserted;
   }
   assert(false && "could not find function");
}
mlir::ResultRange mlir::db::codegen::FunctionRegistry::call(OpBuilder builder, FunctionId function, ValueRange values) {
   FuncOp func = getFunction(builder, function);
   auto funcCall = builder.create<CallOp>(builder.getUnknownLoc(), func, values);
   return funcCall.getResults();
}
void mlir::db::codegen::FunctionRegistry::registerFunction(FunctionId funcId, std::string name, std::vector<mlir::Type> ops, std::vector<mlir::Type> returns, bool useWrapper) {
   registeredFunctions.push_back({name, useWrapper, ops, returns});
   insertedFunctions.push_back(FuncOp());
}