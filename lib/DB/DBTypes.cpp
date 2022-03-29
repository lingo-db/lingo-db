#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

namespace mlir {
template <>
struct FieldParser<mlir::db::DateUnitAttr> {
   static FailureOr<mlir::db::DateUnitAttr> parse(AsmParser& parser) {
      llvm::StringRef str;
      if (parser.parseKeyword(&str)) return failure();
      auto parsed = mlir::db::symbolizeDateUnitAttr(str);
      if (parsed.hasValue()) {
         return parsed.getValue();
      }
      return failure();
   }
};

template <>
struct FieldParser<mlir::db::IntervalUnitAttr> {
   static FailureOr<mlir::db::IntervalUnitAttr> parse(AsmParser& parser) {
      llvm::StringRef str;
      if (parser.parseKeyword(&str)) return failure();
      auto parsed = mlir::db::symbolizeIntervalUnitAttr(str);
      if (parsed.hasValue()) {
         return parsed.getValue();
      }
      return failure();
   }
};

template <>
struct FieldParser<mlir::db::TimeUnitAttr> {
   static FailureOr<mlir::db::TimeUnitAttr> parse(AsmParser& parser) {
      llvm::StringRef str;
      if (parser.parseKeyword(&str)) return failure();
      auto parsed = mlir::db::symbolizeTimeUnitAttr(str);
      if (parsed.hasValue()) {
         return parsed.getValue();
      }
      return failure();
   }
};

namespace db {
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const mlir::db::DateUnitAttr& dt) {
   os << mlir::db::stringifyDateUnitAttr(dt);
   return os;
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const mlir::db::IntervalUnitAttr& dt) {
   os << mlir::db::stringifyIntervalUnitAttr(dt);
   return os;
}
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const mlir::db::TimeUnitAttr& dt) {
   os << mlir::db::stringifyTimeUnitAttr(dt);
   return os;
}
} // end namespace db
} // end namespace mlir
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
namespace mlir::db {
void DBDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::db
