#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include <unordered_set>

using namespace mlir;


void mlir::util::RefType::print(::mlir::AsmPrinter& printer) const {
   printer << "<";
   printer << getElementType() << ">";
}
::mlir::Type mlir::util::RefType::parse(::mlir::AsmParser& parser) {
   Type elementType;
   if (parser.parseLess()) {
      return Type();
   }

   if (parser.parseType(elementType) || parser.parseGreater()) {
      return Type();
      return Type();
   }
   return mlir::util::RefType::get(parser.getContext(), elementType);
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/util/UtilOpsTypes.cpp.inc"

namespace mlir::util {
void UtilDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/util/UtilOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::util
