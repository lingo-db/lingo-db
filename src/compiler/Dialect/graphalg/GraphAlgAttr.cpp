#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LogicalResult.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>
#include <lingodb/compiler/Dialect/graphalg/SemiringTypes.h>

#include "lingodb/compiler/Dialect/graphalg/GraphAlgEnumAttr.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/graphalg/GraphAlgAttr.cpp.inc"

namespace graphalg {

bool binaryOpIsCompare(BinaryOp op) {
   switch (op) {
      case BinaryOp::ADD:
      case BinaryOp::SUB:
      case BinaryOp::MUL:
      case BinaryOp::DIV:
         return false;
      case BinaryOp::EQ:
      case BinaryOp::NE:
      case BinaryOp::LT:
      case BinaryOp::GT:
      case BinaryOp::LE:
      case BinaryOp::GE:
         return true;
   }
}

mlir::LogicalResult
TropInfAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    mlir::Type type) {
   if (!llvm::isa<SemiringTypeInterface>(type)) {
      return emitError() << type << " is not a semiring";
   } else if (!llvm::isa<TropI64Type, TropF64Type, TropMaxI64Type>(type)) {
      return emitError() << TropInfAttr::name
                         << " is not an element of the semiring " << type;
   }

   return mlir::success();
}

mlir::LogicalResult
TropIntAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    mlir::Type type, mlir::IntegerAttr value) {
   if (!llvm::isa<TropI64Type, TropMaxI64Type>(type)) {
      return emitError() << type << " is not an integer semiring";
   }

   auto intType = SemiringTypes::forInt(type.getContext());
   if (value.getType() != intType) {
      return emitError() << value << " is not of type " << intType;
   }

   return mlir::success();
}

mlir::LogicalResult
TropFloatAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      mlir::Type type, mlir::FloatAttr value) {
   if (!llvm::isa<TropF64Type>(type)) {
      return emitError() << type << " is not a floating-point semiring";
   }

   auto realType = SemiringTypes::forReal(type.getContext());
   if (value.getType() != realType) {
      return emitError() << value << " is not of type " << realType;
   }

   return mlir::success();
}

DimAttr DimAttr::getOne(mlir::MLIRContext* context) {
   return getConcrete(context, 1);
}

DimAttr DimAttr::newAbstract(mlir::MLIRContext* context) {
   return Base::get(context,
                    mlir::DistinctAttr::create(mlir::UnitAttr::get(context)),
                    /*dim=*/0);
}

DimAttr DimAttr::getConcrete(mlir::MLIRContext* context, std::uint64_t dim) {
   return Base::get(context, /*id=*/nullptr, dim);
}

bool DimAttr::isAbstract() const { return getAbstractId() != nullptr; }

bool DimAttr::isConcrete() const { return getAbstractId() == nullptr; }

bool DimAttr::isOne() const { return isConcrete() && getConcreteDim() == 1; }

mlir::Attribute DimAttr::parse(mlir::AsmParser& parser, mlir::Type type) {
   if (parser.parseLess()) {
      return nullptr;
   }

   auto bare = parseBare(parser, type);
   if (!bare) {
      return nullptr;
   }

   if (parser.parseGreater()) {
      return nullptr;
   }

   return bare;
}

void DimAttr::print(mlir::AsmPrinter& printer) const {
   printer << "<";
   printBare(printer);
   printer << ">";
}

DimAttr DimAttr::parseBare(mlir::AsmParser& parser, mlir::Type type) {
   std::uint64_t dim;
   auto parsedDim = parser.parseOptionalInteger(dim);
   if (parsedDim.has_value()) {
      if (*parsedDim) {
         return {};
      }

      return getConcrete(parser.getContext(), dim);
   }

   auto id = mlir::FieldParser<mlir::DistinctAttr>::parse(parser);
   if (mlir::failed(id)) {
      return {};
   }

   return Base::get(parser.getContext(), *id, /*dim=*/0);
}

void DimAttr::printBare(mlir::AsmPrinter& printer) const {
   if (auto id = getAbstractId()) {
      printer << id;
   } else {
      printer << getConcreteDim();
   }
}

mlir::LogicalResult
MatrixAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   mlir::Type type, mlir::ArrayAttr elems) {
   auto matType = llvm::dyn_cast<MatrixType>(type);
   if (!matType.getRows().isConcrete()) {
      return emitError() << "row dimension is not concrete\n";
   } else if (!matType.getCols().isConcrete()) {
      return emitError() << "col dimension is not concrete\n";
   }

   auto nRows = matType.getRows().getConcreteDim();
   auto nCols = matType.getCols().getConcreteDim();
   auto nElems = nRows * nCols;
   if (nElems != elems.size()) {
      return emitError() << "expected " << nRows << " * " << nCols << " = "
                         << nElems << " elements, got " << elems.size();
   }

   for (auto elem : elems) {
      auto elemTyped = llvm::dyn_cast<mlir::TypedAttr>(elem);
      if (!elemTyped) {
         return emitError() << "untyped matrix element: " << elem;
      }

      if (elemTyped.getType() != matType.getSemiring()) {
         return emitError() << "element " << elem << " does not match matrix type "
                            << matType;
      }
   }

   return mlir::success();
}

// Need to define this here to avoid depending on GraphAlgAttr in
// GraphAlgDialect and creating a cycle.
void GraphAlgDialect::registerAttributes() {
   addAttributes<
#define GET_ATTRDEF_LIST
#include "lingodb/compiler/Dialect/graphalg/GraphAlgAttr.cpp.inc"
      >();
}

} // namespace graphalg
