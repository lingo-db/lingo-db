#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"

#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h"
#include "lingodb/compiler/mlir-support/tostring.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
using namespace mlir;
struct DBInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const override {
      return true;
   }
};
void lingodb::compiler::dialect::db::DBDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/DB/IR/DBOps.cpp.inc"

      >();
   addInterfaces<DBInlinerInterface>();
   registerTypes();
   runtimeFunctionRegistry = db::RuntimeFunctionRegistry::getBuiltinRegistry(getContext());
}

::mlir::Operation* lingodb::compiler::dialect::db::DBDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value, ::mlir::Type type, ::mlir::Location loc) {
   if (auto decimalType = mlir::dyn_cast_or_null<db::DecimalType>(type)) {
      if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(value)) {
         return builder.create<db::ConstantOp>(loc, type, builder.getStringAttr(support::decimalToString(intAttr.getValue().getLoBits(64).getLimitedValue(), intAttr.getValue().getHiBits(64).getLimitedValue(), decimalType.getS())));
      }
   }
   if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(type)) {
      if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(value)) {
         return builder.create<db::ConstantOp>(loc, type, builder.getStringAttr(support::dateToString(intAttr.getInt())));
      }
   }
   if (mlir::isa<db::StringType, mlir::IntegerType, mlir::FloatType>(type)) {
      return builder.create<db::ConstantOp>(loc, type, value);
   }
   return nullptr;
}
#include "lingodb/compiler/Dialect/DB/IR/DBOpsDialect.cpp.inc"
