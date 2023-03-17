#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir-support/tostring.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/RuntimeFunctions.h"
#include "mlir/IR/DialectImplementation.h"
#include <mlir/Transforms/InliningUtils.h>
using namespace mlir;
using namespace mlir::db;
struct DBInlinerInterface : public DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const override {
      return true;
   }
};
void DBDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"
      >();
   addInterfaces<DBInlinerInterface>();
   registerTypes();
   runtimeFunctionRegistry = mlir::db::RuntimeFunctionRegistry::getBuiltinRegistry(getContext());
}

::mlir::Operation* DBDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value, ::mlir::Type type, ::mlir::Location loc) {
   if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
      if (auto intAttr = value.dyn_cast_or_null<mlir::IntegerAttr>()) {
         return builder.create<mlir::db::ConstantOp>(loc, type, builder.getStringAttr(support::decimalToString(intAttr.getValue().getLoBits(64).getLimitedValue(), intAttr.getValue().getHiBits(64).getLimitedValue(), decimalType.getS())));
      }
   }
   if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
      if (auto intAttr = value.dyn_cast_or_null<mlir::IntegerAttr>()) {
         return builder.create<mlir::db::ConstantOp>(loc, type, builder.getStringAttr(support::dateToString(intAttr.getInt())));
      }
   }
   if (type.isa<mlir::db::StringType, mlir::IntegerType, mlir::FloatType>()) {
      return builder.create<mlir::db::ConstantOp>(loc, type, value);
   }
   return nullptr;
}
#include "mlir/Dialect/DB/IR/DBOpsDialect.cpp.inc"
