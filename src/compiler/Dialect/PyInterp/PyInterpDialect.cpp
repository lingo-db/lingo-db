#include "lingodb/compiler/Dialect/PyInterp/PyInterpDialect.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h"

#include "mlir/IR/Builders.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
namespace {
struct PyInterpInlinerInterface : public mlir::DialectInlinerInterface {
   using DialectInlinerInterface::DialectInlinerInterface;
   bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final override {
      return true;
   }
   virtual bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned, IRMapping& valueMapping) const override {
      return true;
   }
   bool isLegalToInline(mlir::Operation* call, mlir::Operation* callable, bool wouldBeCloned) const override {
      return true;
   }
};
}
namespace lingodb::compiler::dialect::py_interp {

void PyInterpDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.cpp.inc"
      >();
   registerTypes();
   addInterfaces<PyInterpInlinerInterface>();
}
::mlir::Operation* PyInterpDialect::materializeConstant(::mlir::OpBuilder& builder, ::mlir::Attribute value, ::mlir::Type type, ::mlir::Location loc) {
   if (mlir::isa<py_interp::PyObjectType>(type)) {
      if (auto strAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(value)) {
         return builder.create<py_interp::ConstStrPyObject>(loc, type, strAttr);
      }
   }
   return nullptr;
}

} // namespace lingodb::compiler::dialect::py_interp
#include "lingodb/compiler/Dialect/PyInterp/PyInterpDialect.cpp.inc"
