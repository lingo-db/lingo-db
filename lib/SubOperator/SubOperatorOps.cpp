#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/TupleStream/ColumnManager.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOpsAttributes.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

#include <iostream>
#include <queue>

using namespace mlir;
static mlir::tuples::ColumnManager& getColumnManager(::mlir::OpAsmParser& parser) {
   return parser.getBuilder().getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
}
static ParseResult parseCustRef(OpAsmParser& parser, mlir::tuples::ColumnRefAttr& attr) {
   ::mlir::SymbolRefAttr parsedSymbolRefAttr;
   if (parser.parseAttribute(parsedSymbolRefAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
   return success();
}

static void printCustRef(OpAsmPrinter& p, mlir::Operation* op, mlir::tuples::ColumnRefAttr attr) {
   p << attr.getName();
}
static ParseResult parseCustRefArr(OpAsmParser& parser, ArrayAttr& attr) {
   ArrayAttr parsedAttr;
   std::vector<Attribute> attributes;
   if (parser.parseAttribute(parsedAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
      return failure();
   }
   for (auto a : parsedAttr) {
      SymbolRefAttr parsedSymbolRefAttr = a.dyn_cast<SymbolRefAttr>();
      mlir::tuples::ColumnRefAttr attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
      attributes.push_back(attr);
   }
   attr = ArrayAttr::get(parser.getBuilder().getContext(), attributes);
   return success();
}

static void printCustRefArr(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   std::vector<Attribute> attributes;
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      mlir::tuples::ColumnRefAttr parsedSymbolRefAttr = a.dyn_cast<mlir::tuples::ColumnRefAttr>();
      p << parsedSymbolRefAttr.getName();
   }
   p << "]";
}
static ParseResult parseCustDef(OpAsmParser& parser, mlir::tuples::ColumnDefAttr& attr) {
   SymbolRefAttr attrSymbolAttr;
   if (parser.parseAttribute(attrSymbolAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   std::string attrName(attrSymbolAttr.getLeafReference().getValue());
   if (parser.parseLParen()) { return failure(); }
   DictionaryAttr dictAttr;
   if (parser.parseAttribute(dictAttr)) { return failure(); }
   mlir::ArrayAttr fromExisting;
   if (parser.parseRParen()) { return failure(); }
   if (parser.parseOptionalEqual().succeeded()) {
      if (parseCustRefArr(parser, fromExisting)) {
         return failure();
      }
   }
   attr = getColumnManager(parser).createDef(attrSymbolAttr, fromExisting);
   auto propType = dictAttr.get("type").dyn_cast<TypeAttr>().getValue();
   attr.getColumn().type = propType;
   return success();
}
static void printCustDef(OpAsmPrinter& p, mlir::Operation* op, mlir::tuples::ColumnDefAttr attr) {
   p << attr.getName();
   std::vector<mlir::NamedAttribute> relAttrDefProps;
   MLIRContext* context = attr.getContext();
   const mlir::tuples::Column& relationalAttribute = attr.getColumn();
   relAttrDefProps.push_back({mlir::StringAttr::get(context, "type"), mlir::TypeAttr::get(relationalAttribute.type)});
   p << "(" << mlir::DictionaryAttr::get(context, relAttrDefProps) << ")";
   Attribute fromExisting = attr.getFromExisting();
   if (fromExisting) {
      ArrayAttr fromExistingArr = fromExisting.dyn_cast_or_null<ArrayAttr>();
      p << "=";
      printCustRefArr(p, op, fromExistingArr);
   }
}

static ParseResult parseStateColumnMapping(OpAsmParser& parser, DictionaryAttr& attr) {
   if (parser.parseLBrace()) return failure();
   std::vector<mlir::NamedAttribute> columns;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      StringRef colName;
      if (parser.parseKeyword(&colName)) { return failure(); }
      if (parser.parseEqual() || parser.parseGreater()) { return failure(); }
      mlir::tuples::ColumnDefAttr attrDefAttr;
      if (parseCustDef(parser, attrDefAttr)) {
         return failure();
      }
      columns.push_back({StringAttr::get(parser.getBuilder().getContext(), colName), attrDefAttr});
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   attr = mlir::DictionaryAttr::get(parser.getBuilder().getContext(), columns);
   return success();
}
static ParseResult parseColumnStateMapping(OpAsmParser& parser, DictionaryAttr& attr) {
   if (parser.parseLBrace()) return failure();
   std::vector<mlir::NamedAttribute> columns;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      mlir::tuples::ColumnRefAttr columnRefAttr;

      if (parseCustRef(parser, columnRefAttr)) {
         return failure();
      }
      if (parser.parseEqual() || parser.parseGreater()) { return failure(); }
      StringRef colName;
      if (parser.parseKeyword(&colName)) { return failure(); }

      columns.push_back({StringAttr::get(parser.getBuilder().getContext(), colName), columnRefAttr});
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   attr = mlir::DictionaryAttr::get(parser.getBuilder().getContext(), columns);
   return success();
}
static void printStateColumnMapping(OpAsmPrinter& p, mlir::Operation* op, DictionaryAttr attr) {
   p << "{";
   auto first = true;
   for (auto mapping : attr) {
      auto columnName = mapping.getName();
      auto attr = mapping.getValue();
      auto relationDefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>();
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      p << columnName.getValue() << " => ";
      printCustDef(p, op, relationDefAttr);
   }
   p << "}";
}
static void printColumnStateMapping(OpAsmPrinter& p, mlir::Operation* op, DictionaryAttr attr) {
   p << "{";
   auto first = true;
   for (auto mapping : attr) {
      auto columnName = mapping.getName();
      auto attr = mapping.getValue();
      auto relationRefAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnRefAttr>();
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      printCustRef(p, op, relationRefAttr);
      p << " => "<<columnName.getValue();
   }
   p << "}";
}
static ParseResult parseCustRegion(OpAsmParser& parser, Region& result) {
   OpAsmParser::Argument predArgument;
   SmallVector<OpAsmParser::Argument, 4> regionArgs;
   SmallVector<Type, 4> argTypes;
   if (parser.parseLParen()) {
      return failure();
   }
   while (true) {
      Type predArgType;
      if (!parser.parseOptionalRParen()) {
         break;
      }
      if (parser.parseArgument(predArgument) || parser.parseColonType(predArgType)) {
         return failure();
      }
      predArgument.type = predArgType;
      regionArgs.push_back(predArgument);
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRParen()) { return failure(); }
      break;
   }

   if (parser.parseRegion(result, regionArgs)) return failure();
   return success();
}
static void printCustRegion(OpAsmPrinter& p, Operation* op, Region& r) {
   p << "(";
   bool first = true;
   for (auto arg : r.front().getArguments()) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << arg << ": " << arg.getType();
   }
   p << ")";
   p.printRegion(r, false, true);
}
static ParseResult parseCustDefArr(OpAsmParser& parser, ArrayAttr& attr) {
   std::vector<Attribute> attributes;
   if (parser.parseLSquare()) return failure();
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      mlir::tuples::ColumnDefAttr attrDefAttr;
      if (parseCustDef(parser, attrDefAttr)) {
         return failure();
      }
      attributes.push_back(attrDefAttr);
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRSquare()) { return failure(); }
      break;
   }
   attr = parser.getBuilder().getArrayAttr(attributes);
   return success();
}
static void printCustDefArr(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      mlir::tuples::ColumnDefAttr parsedSymbolRefAttr = a.dyn_cast<mlir::tuples::ColumnDefAttr>();
      printCustDef(p, op, parsedSymbolRefAttr);
   }
   p << "]";
}

#define GET_OP_CLASSES
#include "mlir/Dialect/SubOperator/SubOperatorOps.cpp.inc"
