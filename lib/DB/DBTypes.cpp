#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

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
      .Case<::mlir::db::DurationType>([&](::mlir::db::DurationType t) {
         return true;
      })
      .Case<::mlir::db::TimeType>([&](::mlir::db::TimeType t) {
         return true;
      })
      .Default([](::mlir::Type) { return false; });
}
bool mlir::db::DBType::isNullable() {
   return ::llvm::TypeSwitch<::mlir::db::DBType, bool>(*this)
      .Case<::mlir::db::BoolType>([&](::mlir::db::BoolType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::UIntType>([&](::mlir::db::UIntType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::IntervalType>([&](::mlir::db::IntervalType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::FloatType>([&](::mlir::db::FloatType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::DurationType>([&](::mlir::db::DurationType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::TimeType>([&](::mlir::db::TimeType t) {
         return t.getNullable();
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
mlir::db::DBType mlir::db::DBType::getBaseType() const {
   return ::llvm::TypeSwitch<::mlir::db::DBType, mlir::db::DBType>(*this)
      .Case<::mlir::db::BoolType>([&](::mlir::db::BoolType t) {
         return mlir::db::BoolType::get(t.getContext(), false);
      })
      .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) {
         return mlir::db::DateType::get(t.getContext(), false, t.getUnit());
      })
      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
         return mlir::db::DecimalType::get(t.getContext(), false, t.getP(), t.getS());
      })
      .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
         return mlir::db::IntType::get(t.getContext(), false, t.getWidth());
      })
      .Case<::mlir::db::UIntType>([&](::mlir::db::UIntType t) {
         return mlir::db::UIntType::get(t.getContext(), false, t.getWidth());
      })
      .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
         return mlir::db::StringType::get(t.getContext(), false);
      })
      .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
         return mlir::db::TimestampType::get(t.getContext(), false, t.getUnit(), t.getTz());
      })
      .Case<::mlir::db::IntervalType>([&](::mlir::db::IntervalType t) {
         return mlir::db::IntervalType::get(t.getContext(), false, t.getUnit());
      })
      .Case<::mlir::db::TimeType>([&](::mlir::db::TimeType t) {
         return mlir::db::TimeType::get(t.getContext(), false, t.getUnit());
      })
      .Case<::mlir::db::DurationType>([&](::mlir::db::DurationType t) {
         return mlir::db::DurationType::get(t.getContext(), false, t.getUnit());
      })
      .Case<::mlir::db::FloatType>([&](::mlir::db::FloatType t) {
         return mlir::db::FloatType::get(t.getContext(), false, t.getWidth());
      })
      .Default([](::mlir::Type) { return mlir::db::DBType(); });
}
mlir::db::DBType mlir::db::DBType::asNullable() const {
   return ::llvm::TypeSwitch<::mlir::db::DBType, mlir::db::DBType>(*this)
      .Case<::mlir::db::BoolType>([&](::mlir::db::BoolType t) {
         return mlir::db::BoolType::get(t.getContext(), true);
      })
      .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) {
         return mlir::db::DateType::get(t.getContext(), true, t.getUnit());
      })
      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
         return mlir::db::DecimalType::get(t.getContext(), true, t.getP(), t.getS());
      })
      .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
         return mlir::db::IntType::get(t.getContext(), true, t.getWidth());
      })
      .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
         return mlir::db::StringType::get(t.getContext(), true);
      })
      .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
         return mlir::db::TimestampType::get(t.getContext(), true, t.getUnit(), t.getTz());
      })
      .Case<::mlir::db::IntervalType>([&](::mlir::db::IntervalType t) {
         return mlir::db::IntervalType::get(t.getContext(), true, t.getUnit());
      })
      .Case<::mlir::db::TimeType>([&](::mlir::db::TimeType t) {
         return mlir::db::TimeType::get(t.getContext(), true, t.getUnit());
      })
      .Case<::mlir::db::DurationType>([&](::mlir::db::DurationType t) {
         return mlir::db::DurationType::get(t.getContext(), true, t.getUnit());
      })
      .Case<::mlir::db::FloatType>([&](::mlir::db::FloatType t) {
         return mlir::db::FloatType::get(t.getContext(), true, t.getWidth());
      })
      .Default([](::mlir::Type) { return mlir::db::DBType(); });
}
mlir::Type mlir::db::CollectionType::getElementType() const {
   return ::llvm::TypeSwitch<::mlir::db::CollectionType, Type>(*this)
      .Case<::mlir::db::GenericIterableType>([&](::mlir::db::GenericIterableType t) {
         return t.getElementType();
      })
      .Case<::mlir::db::RangeType>([&](::mlir::db::RangeType t) {
         return t.getElementType();
      })
      .Default([](::mlir::Type) { return Type(); });
}
template <class X>
struct ParseSingleImpl {};
template <>
struct ParseSingleImpl<unsigned> {
   static unsigned apply(bool& error, ::mlir::DialectAsmParser& parser) {
      unsigned res;
      if (parser.parseInteger(res)) {
         error = true;
      }
      return res;
   }
};
template <>
struct ParseSingleImpl<std::string> {
   static std::string apply(bool& error, ::mlir::DialectAsmParser& parser) {
      llvm::StringRef ref;
      if (parser.parseKeyword(&ref).failed()) {
         error = true;
      }
      return ref.str();
   }
};
template <>
struct ParseSingleImpl<mlir::db::TimeUnitAttr> {
   static mlir::db::TimeUnitAttr apply(bool& error, ::mlir::DialectAsmParser& parser) {
      mlir::db::TimeUnitAttr unit = mlir::db::TimeUnitAttr::second;

      ::llvm::StringRef attrStr;
      ::mlir::NamedAttrList attrStorage;
      auto loc = parser.getCurrentLocation();
      if (parser.parseKeyword(&attrStr)) {
         parser.emitError(loc, "expected keyword but none found");
         error = true;
         return unit;
      }
      if (!attrStr.empty()) {
         std::string str = attrStr.str();
         auto attrOptional = ::mlir::db::symbolizeTimeUnitAttr(attrStr);
         if (!attrOptional) {
            parser.emitError(loc, "invalid ")
               << "type attribute specification: \"" << attrStr << '"';
            return unit;
         }

         return attrOptional.getValue();
      }
      return unit;
   }
};
template <>
struct ParseSingleImpl<mlir::db::DateUnitAttr> {
   static mlir::db::DateUnitAttr apply(bool& error, ::mlir::DialectAsmParser& parser) {
      mlir::db::DateUnitAttr unit = mlir::db::DateUnitAttr::millisecond;

      ::llvm::StringRef attrStr;
      ::mlir::NamedAttrList attrStorage;
      auto loc = parser.getCurrentLocation();
      if (parser.parseKeyword(&attrStr)) {
         parser.emitError(loc, "expected keyword but none found");
         error = true;
         return unit;
      }
      if (!attrStr.empty()) {
         std::string str = attrStr.str();
         auto attrOptional = ::mlir::db::symbolizeDateUnitAttr(attrStr);
         if (!attrOptional) {
            parser.emitError(loc, "invalid ")
               << "type attribute specification: \"" << attrStr << '"';
            return unit;
         }

         return attrOptional.getValue();
      }
      return unit;
   }
};
template <>
struct ParseSingleImpl<mlir::db::IntervalUnitAttr> {
   static mlir::db::IntervalUnitAttr apply(bool& error, ::mlir::DialectAsmParser& parser) {
      mlir::db::IntervalUnitAttr unit = mlir::db::IntervalUnitAttr::months;

      ::llvm::StringRef attrStr;
      ::mlir::NamedAttrList attrStorage;
      auto loc = parser.getCurrentLocation();
      if (parser.parseKeyword(&attrStr)) {
         parser.emitError(loc, "expected keyword but none found");
         error = true;
         return unit;
      }
      if (!attrStr.empty()) {
         std::string str = attrStr.str();
         auto attrOptional = ::mlir::db::symbolizeIntervalUnitAttr(attrStr);
         if (!attrOptional) {
            parser.emitError(loc, "invalid ")
               << "type attribute specification: \"" << attrStr << '"';
            return unit;
         }

         return attrOptional.getValue();
      }
      return unit;
   }
};
template <class X>
X parseSingle(bool& first, bool& error, ::mlir::DialectAsmParser& parser) {
   if (first) {
      first = false;
   } else if (!first && parser.parseComma()) {
      error = true;
      return X();
   }
   return ParseSingleImpl<X>::apply(error, parser);
}
template <class T, class... ParamT>
mlir::Type parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   constexpr size_t additionalParams = sizeof...(ParamT);
   bool nullable = false;
   bool first = true;
   bool error = false;
   mlir::Type res;
   if constexpr (additionalParams == 0) {
      if (!parser.parseOptionalLess()) {
         if (!parser.parseOptionalKeyword("nullable")) {
            first = false;
            nullable = true;
         }
         if (parser.parseGreater()) {
            return mlir::Type();
         }
      }
      res = T::get(context, nullable, parseSingle<ParamT>(first, error, parser)...);
   } else {
      if (parser.parseLess()) {
         return mlir::Type();
      }
      std::tuple<ParamT...> a = {parseSingle<ParamT>(first, error, parser)...};
      if (parser.parseOptionalGreater().failed()) {
         if (!first && parser.parseOptionalComma().failed()) {
            return mlir::Type();
         }
         if (!parser.parseOptionalKeyword("nullable")) {
            first = false;
            nullable = true;
         }
         if (error || parser.parseGreater()) {
            return mlir::Type();
         }
      }
      std::apply([&](ParamT... params) { res = T::get(context, nullable, (params)...); }, a);
   }
   return res;
}
template <class X>
void printSingle(bool& first, ::mlir::DialectAsmPrinter& printer, X x) {
   if (first) {
      first = false;
   } else {
      printer << ",";
   }
   printer << x;
}
template <class T, class... ParamT>
void print(::mlir::DialectAsmPrinter& printer, bool nullable, ParamT... params) {
   using expander = int[];
   bool first = true;
   size_t args = sizeof...(ParamT) + nullable;
   printer << T::getMnemonic();
   if (!args) {
      return;
   }
   printer << "<";
   (void) expander{0, ((void) printSingle<ParamT>(first, printer, params), 0)...};
   if (nullable) {
      if (!first) {
         printer << ",";
      }
      first = false;
      printer << "nullable";
   }
   printer << ">";
}
::mlir::Type mlir::db::IntType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::IntType, unsigned>(context, parser);
}
void mlir::db::IntType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::IntType, unsigned>(printer, getNullable(), getWidth());
}
::mlir::Type mlir::db::UIntType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::UIntType, unsigned>(context, parser);
}
void mlir::db::UIntType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::UIntType, unsigned>(printer, getNullable(), getWidth());
}
::mlir::Type mlir::db::FloatType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::FloatType, unsigned>(context, parser);
}
void mlir::db::FloatType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::FloatType, unsigned>(printer, getNullable(), getWidth());
}
::mlir::Type mlir::db::BoolType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::BoolType>(context, parser);
}
void mlir::db::BoolType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::BoolType>(printer, getNullable());
}
::mlir::Type mlir::db::DateType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::DateType, DateUnitAttr>(context, parser);
}
void mlir::db::DateType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::DateType>(printer, getNullable(), mlir::db::stringifyDateUnitAttr(getUnit()).str());
}
::mlir::Type mlir::db::IntervalType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::IntervalType, IntervalUnitAttr>(context, parser);
}
void mlir::db::IntervalType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::IntervalType, std::string>(printer, getNullable(), mlir::db::stringifyIntervalUnitAttr(getUnit()).str());
}

::mlir::Type mlir::db::TimestampType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::TimestampType, TimeUnitAttr>(context, parser);
}
void mlir::db::TimestampType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::TimestampType>(printer, getNullable(), mlir::db::stringifyTimeUnitAttr(getUnit()).str());
}
::mlir::Type mlir::db::TimeType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::TimeType, TimeUnitAttr>(context, parser);
}
void mlir::db::TimeType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::TimeType>(printer, getNullable(), mlir::db::stringifyTimeUnitAttr(getUnit()).str());
}
::mlir::Type mlir::db::DurationType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::DurationType, TimeUnitAttr>(context, parser);
}
void mlir::db::DurationType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::DurationType>(printer, getNullable(), mlir::db::stringifyTimeUnitAttr(getUnit()).str());
}
::mlir::Type mlir::db::DecimalType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::DecimalType, unsigned, unsigned>(context, parser);
}
void mlir::db::DecimalType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::DecimalType, unsigned, unsigned>(printer, getNullable(), getP(), getS());
}
::mlir::Type mlir::db::StringType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::StringType>(context, parser);
}
void mlir::db::StringType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::StringType>(printer, getNullable());
}

::mlir::Type mlir::db::GenericIterableType::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser) {
   Type type;
   StringRef parserName;
   if (parser.parseLess() || parser.parseType(type) || parser.parseComma(), parser.parseKeyword(&parserName) || parser.parseGreater()) {
      return mlir::Type();
   }
   return mlir::db::GenericIterableType::get(parser.getBuilder().getContext(), type, parserName.str());
}
void mlir::db::GenericIterableType::print(mlir::DialectAsmPrinter& p) const {
   p << getMnemonic() << "<" << getElementType() << "," << getIteratorName() << ">";
}
::mlir::Type mlir::db::RangeType::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser) {
   Type type;
   if (parser.parseLess() || parser.parseType(type) || parser.parseGreater()) {
      return mlir::Type();
   }
   return mlir::db::RangeType::get(parser.getBuilder().getContext(), type);
}
void mlir::db::RangeType::print(mlir::DialectAsmPrinter& p) const {
   p << getMnemonic() << "<" << getElementType() << ">";
}
::mlir::Type mlir::db::TableBuilderType::parse(mlir::MLIRContext*, mlir::DialectAsmParser& parser) {
   Type type;
   if (parser.parseLess() || parser.parseType(type) || parser.parseGreater()) {
      return mlir::Type();
   }
   return mlir::db::TableBuilderType::get(parser.getBuilder().getContext(), type.dyn_cast<TupleType>());
}
void mlir::db::TableBuilderType::print(mlir::DialectAsmPrinter& p) const {
   p << getMnemonic() << "<" << getRowType() << ">";
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

/// Parse a type registered to this dialect.
::mlir::Type DBDialect::parseType(::mlir::DialectAsmParser& parser) const {
   StringRef memnonic;
   if (parser.parseKeyword(&memnonic)) {
      return Type();
   }
   auto loc = parser.getCurrentLocation();
   Type parsed;
   ::generatedTypeParser(parser.getBuilder().getContext(), parser, memnonic, parsed);
   if (!parsed) {
      parser.emitError(loc, "unknown type");
   }
   return parsed;
}
void DBDialect::printType(::mlir::Type type,
                          ::mlir::DialectAsmPrinter& os) const {
   if (::generatedTypePrinter(type, os).failed()) {
      llvm::errs() << "could not print";
   }
}
} // namespace mlir::db
