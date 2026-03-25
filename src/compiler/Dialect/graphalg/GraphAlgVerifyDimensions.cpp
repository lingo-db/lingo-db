#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGVERIFYDIMENSIONS
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

namespace {

/**
 * Verifies that ops only use dimension symbols that are specified in their
 * function parameters.
 *
 * With this property verified, it is possible to infer concrete dimensions for
 * all ops in a function by specifying concrete dimensions of the parameter
 * types.
 */
class GraphAlgVerifyDimensions
   : public impl::GraphAlgVerifyDimensionsBase<GraphAlgVerifyDimensions> {
   using impl::GraphAlgVerifyDimensionsBase<
      GraphAlgVerifyDimensions>::GraphAlgVerifyDimensionsBase;

   void runOnOperation() final;
};

/** Verifies that operations use a limited set of abstract dimension symbols. */
class AbstractDimVerifier {
   private:
   llvm::SmallDenseSet<DimAttr> _legalDims;

   bool isLegal(DimAttr attr) const;
   mlir::LogicalResult verify(mlir::Operation* op, mlir::Type type) const;
   mlir::LogicalResult verify(mlir::Operation* op,
                              mlir::NamedAttribute attr) const;
   mlir::LogicalResult verify(mlir::Operation* op) const;

   public:
   void addLegalDim(DimAttr attr);

   /**
   * Verifies that this op and all nested ops use only \c DimAttr that have
   * been explicitly marked legal.
   */
   mlir::LogicalResult verifyRecursively(mlir::Operation* op);
};

} // namespace

bool AbstractDimVerifier::isLegal(DimAttr attr) const {
   return attr.isConcrete() || _legalDims.contains(attr);
}

void AbstractDimVerifier::addLegalDim(DimAttr attr) { _legalDims.insert(attr); }

mlir::LogicalResult AbstractDimVerifier::verify(mlir::Operation* op,
                                                mlir::Type type) const {
   auto result = mlir::success();

   type.walk([&](DimAttr attr) {
      if (!isLegal(attr)) {
         op->emitOpError("defines type ") << type << " using dimension " << attr
                                          << " which has not been marked as legal";
         result = mlir::failure();
      }
   });

   return result;
}

mlir::LogicalResult
AbstractDimVerifier::verify(mlir::Operation* op,
                            mlir::NamedAttribute attr) const {
   auto dim = llvm::dyn_cast<DimAttr>(attr.getValue());
   if (dim && !isLegal(dim)) {
      return op->emitOpError("attribute ")
         << attr.getName() << " has value " << dim
         << " which has not been marked as legal";
   }

   return mlir::success();
}

mlir::LogicalResult AbstractDimVerifier::verify(mlir::Operation* op) const {
   auto result = mlir::success();

   // Verify result types.
   for (auto res : op->getOpResults()) {
      if (mlir::failed(verify(op, res.getType()))) {
         result = mlir::failure();
      }
   }

   // Verify attributes
   for (auto attr : op->getAttrs()) {
      if (mlir::failed(verify(op, attr))) {
         result = mlir::failure();
      }
   }

   // Verify block arguments.
   for (auto& region : op->getRegions()) {
      for (auto& block : region) {
         for (auto arg : block.getArgumentTypes()) {
            if (mlir::failed(verify(op, arg))) {
               result = mlir::failure();
            }
         }
      }
   }

   return result;
}

mlir::LogicalResult
AbstractDimVerifier::verifyRecursively(mlir::Operation* op) {
   auto result = verify(op);
   op->walk([&](mlir::Operation* op) {
      if (mlir::failed(verify(op))) {
         result = mlir::failure();
      }
   });

   return result;
}

void GraphAlgVerifyDimensions::runOnOperation() {
   AbstractDimVerifier dimVerifier;

   // All abstract dimensions specified in the function parameters are legal.
   for (auto arg : getOperation().getFunctionType().getInputs()) {
      arg.walk([&](DimAttr attr) { dimVerifier.addLegalDim(attr); });
   }

   if (mlir::failed(dimVerifier.verifyRecursively(getOperation()))) {
      return signalPassFailure();
   }
}

} // namespace graphalg
