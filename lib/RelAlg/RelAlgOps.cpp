#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
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

::mlir::LogicalResult mlir::relalg::AggrFuncOp::inferReturnTypes(::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
   mlir::relalg::AggrFuncOpAdaptor aggrFuncOpAdaptor(operands, attributes);
   if (aggrFuncOpAdaptor.fn() == AggrFunc::count) {
      inferredReturnTypes.push_back(mlir::IntegerType::get(context, 64));
   } else {
      inferredReturnTypes.push_back(mlir::db::NullableType::get(context, getBaseType(aggrFuncOpAdaptor.attr().getRelationalAttribute().type)));
   }
   return success();
}
mlir::relalg::RelationalAttributeManager& getRelationalAttributeManager(::mlir::OpAsmParser& parser) {
   return parser.getBuilder().getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
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
static ParseResult parseCustRef(OpAsmParser& parser, mlir::relalg::RelationalAttributeRefAttr& attr) {
   ::mlir::SymbolRefAttr parsedSymbolRefAttr;
   if (parser.parseAttribute(parsedSymbolRefAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   attr = getRelationalAttributeManager(parser).createRef(parsedSymbolRefAttr);
   return success();
}
void printCustRef(OpAsmPrinter& p, mlir::Operation* op, mlir::relalg::RelationalAttributeRefAttr attr) {
   p << attr.getName();
}
static ParseResult parseCustRegion(OpAsmParser& parser, Region& result) {
   OpAsmParser::OperandType predArgument;
   Type predArgType;
   SmallVector<OpAsmParser::OperandType, 4> regionArgs;
   SmallVector<Type, 4> argTypes;
   if (parser.parseLParen()) {
      return failure();
   }
   while (true) {
      if (!parser.parseOptionalRParen()) {
         break;
      }
      if (parser.parseRegionArgument(predArgument) || parser.parseColonType(predArgType)) {
         return failure();
      }
      regionArgs.push_back(predArgument);
      argTypes.push_back(predArgType);
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRParen()) { return failure(); }
      break;
   }

   if (parser.parseRegion(result, regionArgs, argTypes)) return failure();
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
      mlir::relalg::RelationalAttributeRefAttr attr = getRelationalAttributeManager(parser).createRef(parsedSymbolRefAttr);
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
      mlir::relalg::RelationalAttributeRefAttr parsedSymbolRefAttr = a.dyn_cast<mlir::relalg::RelationalAttributeRefAttr>();
      p << parsedSymbolRefAttr.getName();
   }
   p << "]";
}
static ParseResult parseSortSpecs(OpAsmParser& parser, mlir::ArrayAttr& result) {
   if (parser.parseLSquare()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      mlir::relalg::RelationalAttributeRefAttr attrRefAttr;
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

static ParseResult parseCustDef(OpAsmParser& parser, mlir::relalg::RelationalAttributeDefAttr& attr) {
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
   attr = getRelationalAttributeManager(parser).createDef(attrSymbolAttr, fromExisting);
   auto propType = dictAttr.get("type").dyn_cast<TypeAttr>().getValue();
   attr.getRelationalAttribute().type = propType;
   return success();
}
static void printCustDef(OpAsmPrinter& p, mlir::Operation* op, mlir::relalg::RelationalAttributeDefAttr attr) {
   p.printSymbolName(attr.getName());
   std::vector<mlir::NamedAttribute> relAttrDefProps;
   MLIRContext* context = attr.getContext();
   const mlir::relalg::RelationalAttribute& relationalAttribute = attr.getRelationalAttribute();
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
      mlir::relalg::RelationalAttributeDefAttr attrDefAttr;
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
      mlir::relalg::RelationalAttributeDefAttr parsedSymbolRefAttr = a.dyn_cast<mlir::relalg::RelationalAttributeDefAttr>();
      printCustDef(p, op, parsedSymbolRefAttr);
   }
   p << "]";
}

static ParseResult parseCustAttrMapping(OpAsmParser& parser, ArrayAttr& res) {
   if (parser.parseKeyword("mapping") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      mlir::relalg::RelationalAttributeDefAttr attrDefAttr;
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
      auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
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
static ParseResult parseBaseTableOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());
   if (parser.parseOptionalAttrDict(result.attributes)) return failure();
   if (parser.parseKeyword("columns") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::NamedAttribute> columns;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      StringRef colName;
      if (parser.parseKeyword(&colName)) { return failure(); }
      if (parser.parseEqual() || parser.parseGreater()) { return failure(); }
      mlir::relalg::RelationalAttributeDefAttr attrDefAttr;
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
static void print(OpAsmPrinter& p, relalg::BaseTableOp& op) {
   p << " ";
   p.printSymbolName(op.sym_name());
   if (op->getAttrs().size() > 1) p << ' ';
   std::vector<mlir::NamedAttribute> attrsToPrint;
   for (auto attr : op->getAttrs()) {
      if (attr.getName().str() == "meta") {
         if (auto metaAttr = attr.getValue().dyn_cast_or_null<mlir::relalg::TableMetaDataAttr>()) {
            if (metaAttr.getMeta()->isPresent()) {
               attrsToPrint.push_back(mlir::NamedAttribute(mlir::StringAttr::get(op->getContext(), "meta"), mlir::StringAttr::get(op->getContext(), metaAttr.getMeta()->serialize())));
            }
         }
      } else {
         attrsToPrint.push_back(attr);
      }
   }
   p.printOptionalAttrDict(attrsToPrint, /*elidedAttrs=*/{"sym_name", "columns"});
   p << " columns: {";
   auto first = true;
   for (auto mapping : op.columns()) {
      auto columnName = mapping.getName();
      auto attr = mapping.getValue();
      auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
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

///////////////////////////////////////////////////////////////////////////////////
// AddAttrOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseAddAttrOp(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::OperandType input, tuple, tupleOut;
   Type inputType;
   if (parser.parseOperand(tuple) || parser.parseComma()) {
      return failure();
   }
   mlir::relalg::RelationalAttributeDefAttr defAttr;
   if (parseCustDef(parser, defAttr)) {
      return failure();
   }
   result.addAttribute("attr", defAttr);
   if (parser.parseOperand(input)) {
      return failure();
   }
   inputType = defAttr.getRelationalAttribute().type;

   auto tupleType = mlir::relalg::TupleType::get(parser.getBuilder().getContext());
   if (parser.resolveOperand(tuple, tupleType, result.operands)) {
      return failure();
   }
   if (parser.resolveOperand(input, inputType, result.operands)) {
      return failure();
   }
   result.addTypes({tupleType});
   return success();
}
static void print(OpAsmPrinter& p, relalg::AddAttrOp& op) {
   p << " " << op.tuple();
   p << ", ";
   printCustDef(p, op, op.attr());
   p << " " << op.val();
}

static ParseResult parseAttrNS(OpAsmParser& parser, StringAttr& nameAttr) {
   NamedAttrList attrList;
   if (parser.parseSymbolName(nameAttr, "symbol", attrList)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());
   return success();
}
static void printAttrNS(OpAsmPrinter& p, mlir::Operation* op, StringAttr nameAttr) {
   p.printSymbolName(nameAttr);
}

#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"