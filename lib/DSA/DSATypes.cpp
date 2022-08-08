#include "mlir/Dialect/DSA/IR/DSATypes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/DSA/IR/DSAOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

mlir::Type mlir::dsa::RecordBatchType::getElementType() {
   return mlir::dsa::RecordType::get(getContext(), getRowType());
}
::mlir::Type mlir::dsa::GenericIterableType::parse(mlir::AsmParser& parser) {
   Type type;
   StringRef parserName;
   if (parser.parseLess() || parser.parseType(type) || parser.parseComma(), parser.parseKeyword(&parserName) || parser.parseGreater()) {
      return mlir::Type();
   }
   return mlir::dsa::GenericIterableType::get(parser.getBuilder().getContext(), type, parserName.str());
}
void mlir::dsa::GenericIterableType::print(mlir::AsmPrinter& p) const {
   p << "<" << getElementType() << "," << getIteratorName() << ">";
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
namespace mlir::dsa {
void DSADialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::dsa
