#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"

#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include <llvm/ADT/TypeSwitch.h>
#include <iostream>
#include <queue>

using namespace mlir;

///////////////////////////////////////////////////////////////////////////////////
// Utility Functions
///////////////////////////////////////////////////////////////////////////////////

mlir::relalg::ColumnManager& getColumnManager(::mlir::OpAsmParser& parser) {
   return parser.getBuilder().getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
}
::mlir::ParseResult parseSortSpec(::mlir::OpAsmParser& parser, mlir::relalg::SortSpec& spec) {
   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr, {"desc", "asc"})) {
      return parser.emitError(loc, "expected keyword containing one of the following enum values for attribute 'sortSpec' [desc,asc]");
   }
   if (!attrStr.empty()) {
      auto parsedSpec = ::mlir::relalg::symbolizeSortSpec(attrStr);
      if (!parsedSpec)
         return parser.emitError(loc, "invalid ")
            << "type attribute specification: \"" << attrStr << '"';
      spec = parsedSpec.getValue();
   }
   return success();
}
static ParseResult parseCustRef(OpAsmParser& parser, mlir::relalg::ColumnRefAttr& attr) {
   ::mlir::SymbolRefAttr parsedSymbolRefAttr;
   if (parser.parseAttribute(parsedSymbolRefAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
   return success();
}

void printCustRef(OpAsmPrinter& p, mlir::Operation* op, mlir::relalg::ColumnRefAttr attr) {
   p << attr.getName();
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
      predArgument.type=predArgType;
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
static ParseResult parseCustRefArr(OpAsmParser& parser, ArrayAttr& attr) {
   ArrayAttr parsedAttr;
   std::vector<Attribute> attributes;
   if (parser.parseAttribute(parsedAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
      return failure();
   }
   for (auto a : parsedAttr) {
      SymbolRefAttr parsedSymbolRefAttr = a.dyn_cast<SymbolRefAttr>();
      mlir::relalg::ColumnRefAttr attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
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
      mlir::relalg::ColumnRefAttr parsedSymbolRefAttr = a.dyn_cast<mlir::relalg::ColumnRefAttr>();
      p << parsedSymbolRefAttr.getName();
   }
   p << "]";
}
static ParseResult parseSortSpecs(OpAsmParser& parser, mlir::ArrayAttr& result) {
   if (parser.parseLSquare()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      mlir::relalg::ColumnRefAttr attrRefAttr;
      if (parser.parseLParen() || parseCustRef(parser, attrRefAttr) || parser.parseComma()) {
         return failure();
      }
      mlir::relalg::SortSpec spec;
      if (parseSortSpec(parser, spec) || parser.parseRParen()) {
         return failure();
      }
      mapping.push_back(mlir::relalg::SortSpecificationAttr::get(parser.getBuilder().getContext(), attrRefAttr, spec));
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRSquare()) { return failure(); }
      break;
   }
   result = mlir::ArrayAttr::get(parser.getBuilder().getContext(), mapping);
   return success();
}
static void printSortSpecs(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   std::vector<Attribute> attributes;
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      mlir::relalg::SortSpecificationAttr sortSpecificationAttr = a.dyn_cast<mlir::relalg::SortSpecificationAttr>();
      p << "(" << sortSpecificationAttr.getAttr().getName() << "," << mlir::relalg::stringifySortSpec(sortSpecificationAttr.getSortSpec()) << ")";
   }
   p << "]";
}

static ParseResult parseCustDef(OpAsmParser& parser, mlir::relalg::ColumnDefAttr& attr) {
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
static void printCustDef(OpAsmPrinter& p, mlir::Operation* op, mlir::relalg::ColumnDefAttr attr) {
   p<<attr.getName();
   std::vector<mlir::NamedAttribute> relAttrDefProps;
   MLIRContext* context = attr.getContext();
   const mlir::relalg::Column& relationalAttribute = attr.getColumn();
   relAttrDefProps.push_back({mlir::StringAttr::get(context, "type"), mlir::TypeAttr::get(relationalAttribute.type)});
   p << "(" << mlir::DictionaryAttr::get(context, relAttrDefProps) << ")";
   Attribute fromExisting = attr.getFromExisting();
   if (fromExisting) {
      ArrayAttr fromExistingArr = fromExisting.dyn_cast_or_null<ArrayAttr>();
      p << "=";
      printCustRefArr(p, op, fromExistingArr);
   }
}

static ParseResult parseCustDefArr(OpAsmParser& parser, ArrayAttr& attr) {
   std::vector<Attribute> attributes;
   if (parser.parseLSquare()) return failure();
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      mlir::relalg::ColumnDefAttr attrDefAttr;
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
      mlir::relalg::ColumnDefAttr parsedSymbolRefAttr = a.dyn_cast<mlir::relalg::ColumnDefAttr>();
      printCustDef(p, op, parsedSymbolRefAttr);
   }
   p << "]";
}

static ParseResult parseCustAttrMapping(OpAsmParser& parser, ArrayAttr& res) {
   if (parser.parseKeyword("mapping") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      mlir::relalg::ColumnDefAttr attrDefAttr;
      if (parseCustDef(parser, attrDefAttr)) {
         return failure();
      }
      mapping.push_back(attrDefAttr);
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   res = mlir::ArrayAttr::get(parser.getBuilder().getContext(), mapping);
   return success();
}
static void printCustAttrMapping(OpAsmPrinter& p, mlir::Operation* op, Attribute mapping) {
   p << " mapping: {";
   auto first = true;
   for (auto attr : mapping.dyn_cast_or_null<ArrayAttr>()) {
      auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      printCustDef(p, op, relationDefAttr);
   }
   p << "}";
}

///////////////////////////////////////////////////////////////////////////////////
// BaseTableOp
///////////////////////////////////////////////////////////////////////////////////
ParseResult mlir::relalg::BaseTableOp::parse(OpAsmParser& parser, OperationState& result) {
   if (parser.parseOptionalAttrDict(result.attributes)) return failure();
   if (parser.parseKeyword("columns") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::NamedAttribute> columns;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      StringRef colName;
      if (parser.parseKeyword(&colName)) { return failure(); }
      if (parser.parseEqual() || parser.parseGreater()) { return failure(); }
      mlir::relalg::ColumnDefAttr attrDefAttr;
      if (parseCustDef(parser, attrDefAttr)) {
         return failure();
      }
      columns.push_back({StringAttr::get(parser.getBuilder().getContext(), colName), attrDefAttr});
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   auto meta = result.attributes.get("meta");
   if (meta) {
      if (auto strAttr = meta.dyn_cast<mlir::StringAttr>()) {
         result.attributes.set("meta", mlir::relalg::TableMetaDataAttr::get(parser.getContext(), runtime::TableMetaData::deserialize(strAttr.str())));
      } else {
         return failure();
      }
   } else {
      result.addAttribute("meta", mlir::relalg::TableMetaDataAttr::get(parser.getContext(), std::make_shared<runtime::TableMetaData>()));
   }
   result.addAttribute("columns", mlir::DictionaryAttr::get(parser.getBuilder().getContext(), columns));
   return parser.addTypeToList(mlir::relalg::TupleStreamType::get(parser.getBuilder().getContext()), result.types);
}
void mlir::relalg::BaseTableOp::print(OpAsmPrinter& p) {
   p << " ";
   std::vector<mlir::NamedAttribute> colsToPrint;
   for (auto attr : this->getOperation()->getAttrs()) {
      if (attr.getName().str() == "meta") {
         if (auto metaAttr = attr.getValue().dyn_cast_or_null<mlir::relalg::TableMetaDataAttr>()) {
            if (metaAttr.getMeta()->isPresent()) {
               colsToPrint.push_back(mlir::NamedAttribute(mlir::StringAttr::get(getContext(), "meta"), mlir::StringAttr::get(getContext(), metaAttr.getMeta()->serialize())));
            }
         }
      } else {
         colsToPrint.push_back(attr);
      }
   }
   p.printOptionalAttrDict(colsToPrint, /*elidedAttrs=*/{"sym_name", "columns"});
   p << " columns: {";
   auto first = true;
   for (auto mapping : columns()) {
      auto columnName = mapping.getName();
      auto attr = mapping.getValue();
      auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      p << columnName.getValue() << " => ";
      printCustDef(p, *this, relationDefAttr);
   }
   p << "}";
}







#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"