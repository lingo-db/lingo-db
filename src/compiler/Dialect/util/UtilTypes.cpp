#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/Dialect/util/UtilTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include <unordered_set>

using namespace lingodb::compiler::dialect;

void util::RefType::print(::mlir::AsmPrinter& printer) const {
   printer << "<";
   printer << getElementType() << ">";
}
::mlir::Type util::RefType::parse(::mlir::AsmParser& parser) {
   Type elementType;
   if (parser.parseLess()) {
      return Type();
   }

   if (parser.parseType(elementType) || parser.parseGreater()) {
      return Type();
      return Type();
   }
   return util::RefType::get(parser.getContext(), elementType);
}
mlir::Type util::BufferType::getElementType() {
   return util::RefType::get(getContext(), getT());
}

#include "lingodb/compiler/Dialect/util/UtilOpsTypeInterfaces.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/util/UtilOpsTypes.cpp.inc"

namespace lingodb::compiler::dialect::util {
void UtilDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/util/UtilOpsTypes.cpp.inc"

      >();
}

} // namespace lingodb::compiler::dialect::util
