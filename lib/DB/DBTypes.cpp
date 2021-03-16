#include "mlir/Dialect/DB/IR/DBTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

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
      .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
         return t.getNullable();
      })
      .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
         return t.getNullable();
      })
      .Default([](::mlir::Type) { return false; });
}
mlir::db::DBType mlir::db::DBType::getBaseType() const{
   return ::llvm::TypeSwitch<::mlir::db::DBType, mlir::db::DBType>(*this)
      .Case<::mlir::db::BoolType>([&](::mlir::db::BoolType t) {
        return mlir::db::BoolType::get(t.getContext(),false);
      })
      .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) {
        return mlir::db::DateType::get(t.getContext(),false);
      })
      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
        return mlir::db::DecimalType::get(t.getContext(),false,t.getP(),t.getS());
      })
      .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
        return mlir::db::IntType::get(t.getContext(),false,t.getWidth());
      })
      .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
        return mlir::db::StringType::get(t.getContext(),false);
      })
      .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
        return mlir::db::TimestampType::get(t.getContext(),false);
      })
      .Default([](::mlir::Type) { return mlir::db::DBType(); });
}
mlir::db::DBType mlir::db::DBType::asNullable() const{
   return ::llvm::TypeSwitch<::mlir::db::DBType, mlir::db::DBType>(*this)
      .Case<::mlir::db::BoolType>([&](::mlir::db::BoolType t) {
        return mlir::db::BoolType::get(t.getContext(),true);
      })
      .Case<::mlir::db::DateType>([&](::mlir::db::DateType t) {
        return mlir::db::DateType::get(t.getContext(),true);
      })
      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
        return mlir::db::DecimalType::get(t.getContext(),true,t.getP(),t.getS());
      })
      .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
        return mlir::db::IntType::get(t.getContext(),true,t.getWidth());
      })
      .Case<::mlir::db::StringType>([&](::mlir::db::StringType t) {
        return mlir::db::StringType::get(t.getContext(),true);
      })
      .Case<::mlir::db::TimestampType>([&](::mlir::db::TimestampType t) {
        return mlir::db::TimestampType::get(t.getContext(),true);
      })
      .Default([](::mlir::Type) { return mlir::db::DBType(); });
}

template <class X>
struct parseSingleImpl {};
template <>
struct parseSingleImpl<unsigned> {
   static unsigned apply(bool& error, ::mlir::DialectAsmParser& parser) {
      unsigned res;
      if (parser.parseInteger(res)) {
         error = true;
      }
      return res;
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
   return parseSingleImpl<X>::apply(error, parser);
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
         if(!first&&parser.parseOptionalComma().failed()){
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
      if(!first){
         printer<<",";
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
::mlir::Type mlir::db::BoolType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::BoolType>(context, parser);
}
void mlir::db::BoolType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::BoolType>(printer, getNullable());
}
::mlir::Type mlir::db::DateType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::DateType>(context, parser);
}
void mlir::db::DateType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::DateType>(printer, getNullable());
}
::mlir::Type mlir::db::TimestampType::parse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser) {
   return ::parse<mlir::db::TimestampType>(context, parser);
}
void mlir::db::TimestampType::print(::mlir::DialectAsmPrinter& printer) const {
   ::print<mlir::db::TimestampType>(printer, getNullable());
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

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DB/IR/DBOpsTypes.cpp.inc"
namespace mlir::db {
::mlir::Type DBTypeParse(::mlir::MLIRContext* context, ::mlir::DialectAsmParser& parser, ::llvm::StringRef mnemonic) {
   return ::generatedTypeParser(context, parser, mnemonic);
}
::mlir::LogicalResult DBTypePrint(::mlir::Type type, ::mlir::DialectAsmPrinter& printer) {
   return ::generatedTypePrinter(type, printer);
}
}