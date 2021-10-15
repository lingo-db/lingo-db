#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/Dialect/util/UtilTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include <unordered_set>

using namespace mlir;


void mlir::util::GenericMemrefType::print(::mlir::DialectAsmPrinter& printer) const {
   printer << getMnemonic() << "<";
   if (getSize() && getSize().getValue() == -1) {
      printer << "? x ";
   } else if (getSize()) {
      printer << getSize().getValue() << " x ";
   }
   printer << getElementType() << ">";
}
::mlir::Type mlir::util::GenericMemrefType::parse(::mlir::DialectAsmParser& parser) {
   Type elementType;
   llvm::Optional<int64_t> size;
   if (parser.parseLess()) {
      return Type();
   }
   if (parser.parseOptionalQuestion().succeeded()) {
      if (parser.parseKeyword("x")) {
         return Type();
      }
      size = -1;
   }
   if (parser.parseType(elementType) || parser.parseGreater()) {
      return Type();
   }
   return mlir::util::GenericMemrefType::get(parser.getContext(), elementType, size);
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

/// Parse a type registered to this dialect.
::mlir::Type UtilDialect::parseType(::mlir::DialectAsmParser& parser) const {
   StringRef memnonic;
   if (parser.parseKeyword(&memnonic)) {
      return Type();
   }
   auto loc = parser.getCurrentLocation();
   Type parsed;
   ::generatedTypeParser(parser, memnonic, parsed);
   if (!parsed) {
      parser.emitError(loc, "unknown type");
   }
   return parsed;
}
void UtilDialect::printType(::mlir::Type type,
                            ::mlir::DialectAsmPrinter& os) const {
   if (::generatedTypePrinter(type, os).failed()) {
      llvm::errs() << "could not print";
   }
}
} // namespace mlir::util
