#pragma once

#include <mlir/IR/DialectImplementation.h>

/**
 * Helper for parsing attributes using a 'parseBare' methods from a mlir
 * tablegen `custom<BareAttr>(...)` invocation. This is inteded for parsing
 * attributes as an alias, or without the typical `<` `>` wrapping.
 */
template <typename T>
inline mlir::ParseResult parseBareAttr(mlir::AsmParser& parser, T& value) {
   return parser.parseCustomAttributeWithFallback(
      value, nullptr,
      [&](mlir::Attribute& result, mlir::Type type) -> mlir::ParseResult {
         result = T::parseBare(parser, type);
         return mlir::success(!!result);
      });
}

template <typename T>
inline mlir::ParseResult parseBareAttr(mlir::AsmParser& parser,
                                       llvm::SmallVector<T>& value) {
   return parser.parseCommaSeparatedList([&]() -> mlir::ParseResult {
      return parseBareAttr(parser, value.emplace_back());
   });
}

template <typename T>
inline void printBareAttr(mlir::AsmPrinter& printer, const T& value) {
   if (mlir::failed(printer.printAlias(value))) {
      value.printBare(printer);
   }
}
template <typename T>
inline void printBareAttr(mlir::AsmPrinter& printer, llvm::ArrayRef<T> value) {
   llvm::interleaveComma(value, printer,
                         [&](const T& v) { printBareAttr(printer, v); });
}
template <typename T>
inline void printBareAttr(mlir::AsmPrinter& printer, mlir::Operation* op,
                          const T& value) {
   if (mlir::failed(printer.printAlias(value))) {
      value.printBare(printer);
   }
}
