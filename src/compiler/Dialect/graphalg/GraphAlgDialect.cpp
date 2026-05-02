#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Transforms/InliningUtils.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>

#include "lingodb/compiler/Dialect/graphalg/GraphAlgOpsDialect.cpp.inc"

namespace graphalg {

namespace {

class GraphAlgOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
   public:
   using OpAsmDialectInterface::OpAsmDialectInterface;

   AliasResult getAlias(mlir::Attribute attr,
                        mlir::raw_ostream& os) const override {
      // Assign aliases to abstract dimension symbols.
      if (auto dimAttr = llvm::dyn_cast<DimAttr>(attr)) {
         if (dimAttr.isAbstract()) {
            os << "dim";
            return AliasResult::FinalAlias;
         }
      }

      return AliasResult::NoAlias;
   }
};

/**
 * Defines inlining support for ops in this dialect.
 *
 * Since all ops in the dialect are free of side effects, they are trivially
 * legal to inline.
 */
struct GraphAlgInlinerInterface : public mlir::DialectInlinerInterface {
   public:
   using DialectInlinerInterface::DialectInlinerInterface;

   // All operations can be inlined.
   bool isLegalToInline(mlir::Operation* op, mlir::Region* region,
                        bool wouldBeCloned,
                        mlir::IRMapping& mapping) const final {
      return true;
   }

   // All regions can be inlined.
   bool isLegalToInline(mlir::Region* dest, mlir::Region* src,
                        bool wouldBeCloned,
                        mlir::IRMapping& valueMapping) const final {
      return true;
   }
};

} // namespace

void GraphAlgDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/graphalg/GraphAlgOps.cpp.inc"
      >();

   registerAttributes();
   registerTypes();
   _cast.registerCasts(getContext());
   addInterface<GraphAlgOpAsmDialectInterface>();
   addInterface<GraphAlgInlinerInterface>();
}

mlir::Operation* GraphAlgDialect::materializeConstant(mlir::OpBuilder& builder,
                                                      mlir::Attribute value,
                                                      mlir::Type type,
                                                      mlir::Location loc) {
   if (auto typedValue = llvm::dyn_cast<mlir::TypedAttr>(value)) {
      if (auto matrix = llvm::dyn_cast<MatrixType>(type)) {
         if (typedValue.getType() == matrix.getSemiring()) {
            return builder.create<ConstantMatrixOp>(loc, matrix, typedValue);
         }
      } else if (auto sring = llvm::dyn_cast<SemiringTypeInterface>(type)) {
         if (typedValue.getType() == sring) {
            return builder.create<ConstantOp>(loc, typedValue);
         }
      }
   }

   mlir::emitError(loc) << "Unable to materialize constant value " << value
                        << " of type " << type;
   return nullptr;
}

bool GraphAlgDialect::isCastLegal(mlir::Type from, mlir::Type to) const {
   return _cast.isCastLegal(from, to);
}

mlir::TypedAttr GraphAlgDialect::castAttribute(mlir::TypedAttr attr,
                                               mlir::Type to) const {
   return _cast.castAttribute(attr, to);
}

} // namespace graphalg
