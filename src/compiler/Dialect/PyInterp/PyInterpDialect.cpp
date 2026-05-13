#include "lingodb/compiler/Dialect/PyInterp/PyInterpDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h"

#include "mlir/IR/Builders.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
namespace {
using namespace lingodb::compiler::dialect;
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

// Refcount semantics for `!py_interp.py_object`: each "fresh" reference
// returned by py_call / get_attr / import / cast_to_pyobject is owned by the
// SSA value; the memory-management pass cleans it up via py_interp.dec_ref at
// end of scope, and bumps it via py_interp.inc_ref when needed.
struct PyObjectTypeManagedModel
   : public db::ManagedType::ExternalModel<PyObjectTypeManagedModel,
                                           py_interp::PyObjectType> {
   bool needsManagement(mlir::Type) const { return true; }
   void emitAddUse(mlir::Type, mlir::OpBuilder& builder, mlir::Location loc,
                   mlir::Value value) const {
      builder.create<py_interp::IncRef>(loc, value);
   }
   void emitCleanupUse(mlir::Type, mlir::OpBuilder& builder, mlir::Location loc,
                       mlir::Value value, mlir::SymbolRefAttr /*elementFn*/) const {
      builder.create<py_interp::DecRef>(loc, value);
   }
   mlir::Value emitPromoteToGlobal(mlir::Type, mlir::OpBuilder& builder,
                                   mlir::Location loc,
                                   mlir::Value value) const {
      // PyObjects have no per-query arena; promoting past the map scope just
      // means bumping the refcount so a single owned reference still exists
      // after the cleanup_use the pass emits in the map block.
      builder.create<py_interp::IncRef>(loc, value);
      return value;
   }
};
} // namespace
namespace lingodb::compiler::dialect::py_interp {

void PyInterpDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.cpp.inc"
      >();
   registerTypes();
   addInterfaces<PyInterpInlinerInterface>();
   PyObjectType::attachInterface<PyObjectTypeManagedModel>(*getContext());
}

} // namespace lingodb::compiler::dialect::py_interp
#include "lingodb/compiler/Dialect/PyInterp/PyInterpDialect.cpp.inc"
