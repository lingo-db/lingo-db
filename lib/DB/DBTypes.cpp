#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

bool mlir::db::DBType::classof(Type t) {
   return ::llvm::TypeSwitch<Type, bool>(t)
      .Case<::mlir::db::BoolType>([&](::mlir::db::BoolType t) {
         return true;
      })
      .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) {
         return true;
      })
      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
         return true;
      })
      .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
         return true;
      })
      .Case<::mlir::db::UIntType>([&](::mlir::db::UIntType t) {
         return true;
      })
      .Case<::mlir::db::CharType>([&](::mlir::db::CharType t) {
         return true;
      })
      .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
         return true;
      })
      .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
         return true;
      })
      .Case<::mlir::db::IntervalType>([&](::mlir::db::IntervalType t) {
         return true;
      })
      .Case<::mlir::db::FloatType>([&](::mlir::db::FloatType t) {
         return true;
      })
      .Default([](::mlir::Type) { return false; });
}
bool mlir::db::DBType::isVarLen() const {
   return ::llvm::TypeSwitch<::mlir::db::DBType, bool>(*this)
      .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
         return true;
      })
      .Default([](::mlir::Type) { return false; });
}
mlir::Type mlir::db::CollectionType::getElementType() const {
   return ::llvm::TypeSwitch<::mlir::db::CollectionType, Type>(*this)
      .Case<::mlir::db::GenericIterableType>([&](::mlir::db::GenericIterableType t) {
         return t.getElementType();
      })
      .Case<::mlir::db::VectorType>([&](::mlir::db::VectorType t) {
         return t.getElementType();
      })
      .Case<::mlir::db::JoinHashtableType>([&](::mlir::db::JoinHashtableType t) {
         return TupleType::get(getContext(), {t.getKeyType(), t.getValType()});
      })
      .Case<::mlir::db::AggregationHashtableType>([&](::mlir::db::AggregationHashtableType t) {
         return TupleType::get(t.getContext(), {t.getKeyType(), t.getValType()});
      })
      .Default([](::mlir::Type) { return Type(); });
}
bool mlir::db::CollectionType::classof(Type t) {
   return ::llvm::TypeSwitch<Type, bool>(t)
      .Case<::mlir::db::GenericIterableType>([&](::mlir::db::GenericIterableType t) { return true; })
      .Case<::mlir::db::VectorType>([&](::mlir::db::VectorType t) {
         return true;
      })
      .Case<::mlir::db::JoinHashtableType>([&](::mlir::db::JoinHashtableType t) {
         return true;
      })
      .Case<::mlir::db::AggregationHashtableType>([&](::mlir::db::AggregationHashtableType t) {
         return true;
      })
      .Default([](::mlir::Type) { return false; });
}

::mlir::Type mlir::db::GenericIterableType::parse(mlir::AsmParser& parser) {
   Type type;
   StringRef parserName;
   if (parser.parseLess() || parser.parseType(type) || parser.parseComma(), parser.parseKeyword(&parserName) || parser.parseGreater()) {
      return mlir::Type();
   }
   return mlir::db::GenericIterableType::get(parser.getBuilder().getContext(), type, parserName.str());
}
void mlir::db::GenericIterableType::print(mlir::AsmPrinter& p) const {
   p << "<" << getElementType() << "," << getIteratorName() << ">";
}

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
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const mlir::db::DateUnitAttr& dt)
{
   os<<mlir::db::stringifyDateUnitAttr(dt);
   return os;
}
template<>
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
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const mlir::db::IntervalUnitAttr& dt)
{
   os<<mlir::db::stringifyIntervalUnitAttr(dt);
   return os;
}
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
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const mlir::db::TimeUnitAttr& dt)
{
   os<<mlir::db::stringifyTimeUnitAttr(dt);
   return os;
}
}
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
