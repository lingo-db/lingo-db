#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/IR/OpImplementation.h"
#include <iostream>
#include <queue>
#include <unordered_set>
using namespace mlir;

///////////////////////////////////////////////////////////////////////////////////
// Utility Functions
///////////////////////////////////////////////////////////////////////////////////

mlir::relalg::RelationalAttributeManager& getRelationalAttributeManager(::mlir::OpAsmParser& parser) {
   return parser.getBuilder().getContext()->getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
}
::mlir::ParseResult parseAggrFn(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   ::mlir::IntegerAttr typeAttr;

   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr, {"min", "max", "sum", "avg", "count"})) {
      return parser.emitError(loc, "expected keyword containing one of the following enum values for attribute 'type' [min, max, sum, avg, count]");
   }
   if (!attrStr.empty()) {
      auto attrOptional = ::mlir::relalg::symbolizeAggrFunc(attrStr);
      if (!attrOptional)
         return parser.emitError(loc, "invalid ")
            << "type attribute specification: \"" << attrStr << '"';
      ;

      typeAttr = parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue()));
      result.addAttribute("fn", typeAttr);
   }
   return success();
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
::mlir::ParseResult parseSetSemantic(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   ::mlir::IntegerAttr typeAttr;

   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr, {"distinct", "all"})) {
      return parser.emitError(loc, "expected keyword containing one of the following enum values for attribute 'type' [distinct, all]");
   }
   if (!attrStr.empty()) {
      auto attrOptional = ::mlir::relalg::symbolizeSetSemantic(attrStr);
      if (!attrOptional)
         return parser.emitError(loc, "invalid ")
            << "type attribute specification: \"" << attrStr << '"';
      ;

      typeAttr = parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue()));
      result.addAttribute("set_semantic", typeAttr);
   }
   return success();
}
void printSetSemantic(OpAsmPrinter& p, mlir::relalg::SetSemantic semantic) {
   std::string sem(mlir::relalg::stringifySetSemantic(semantic));
   p << sem;
}
static ParseResult parseAttributeRefAttr(OpAsmParser& parser, OperationState& result, mlir::relalg::RelationalAttributeRefAttr& attr) {
   ::mlir::SymbolRefAttr parsedSymbolRefAttr;
   if (parser.parseAttribute(parsedSymbolRefAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   attr = getRelationalAttributeManager(parser).createRef(parsedSymbolRefAttr);
   return success();
}
static ParseResult parseCustomRegion(OpAsmParser& parser, OperationState& result) {
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

   Region* body = result.addRegion();
   if (parser.parseRegion(*body, regionArgs, argTypes)) return failure();
   return success();
}
static void printCustomRegion(OpAsmPrinter& p, Region& r) {
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
static ParseResult parseAttributeRefArr(OpAsmParser& parser, OperationState& result, Attribute& attr) {
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

static void printAttributeRefArr(OpAsmPrinter& p, ArrayAttr arrayAttr) {
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
static ParseResult parseSortSpecs(OpAsmParser& parser, OperationState& result) {
   if (parser.parseLSquare()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      mlir::relalg::RelationalAttributeRefAttr attrRefAttr;
      if (parser.parseLParen() || parseAttributeRefAttr(parser, result, attrRefAttr) || parser.parseComma()) {
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
   result.addAttribute("sortspecs", mlir::ArrayAttr::get(parser.getBuilder().getContext(), mapping));
   return success();
}
static void printSortSpecs(OpAsmPrinter& p, ArrayAttr arrayAttr) {
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
static ParseResult addRelationOutput(OpAsmParser& parser, OperationState& result) {
   return parser.addTypeToList(mlir::relalg::TupleStreamType::get(parser.getBuilder().getContext()), result.types);
}
static ParseResult parseRelationalInputs(OpAsmParser& parser, OperationState& result, size_t inputs) {
   SmallVector<OpAsmParser::OperandType, 4> operands;
   if (parser.parseOperandList(operands)) {
      return failure();
   }
   if (parser.resolveOperands(operands, mlir::relalg::TupleStreamType::get(parser.getBuilder().getContext()), result.operands)) {
      return failure();
   }
   return success();
}

static ParseResult parseAttributeDefAttr(OpAsmParser& parser, OperationState& result, mlir::relalg::RelationalAttributeDefAttr& attr) {
   SymbolRefAttr attrSymbolAttr;
   if (parser.parseAttribute(attrSymbolAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   std::string attrName(attrSymbolAttr.getLeafReference().getValue());
   if (parser.parseLParen()) { return failure(); }
   DictionaryAttr dictAttr;
   if (parser.parseAttribute(dictAttr)) { return failure(); }
   Attribute fromExisting;
   if (parser.parseRParen()) { return failure(); }
   if (parser.parseOptionalEqual().succeeded()) {
      if (parseAttributeRefArr(parser, result, fromExisting)) {
         return failure();
      }
   }
   attr = getRelationalAttributeManager(parser).createDef(attrSymbolAttr, fromExisting);
   auto propType = dictAttr.get("type").dyn_cast<TypeAttr>().getValue();
   attr.getRelationalAttribute().type = propType;
   return success();
}
static void printAttributeDefAttr(OpAsmPrinter& p, mlir::relalg::RelationalAttributeDefAttr attr) {
   p.printSymbolName(attr.getName());
   std::vector<mlir::NamedAttribute> relAttrDefProps;
   MLIRContext* context = attr.getContext();
   const mlir::relalg::RelationalAttribute& relationalAttribute = attr.getRelationalAttribute();
   relAttrDefProps.push_back({mlir::Identifier::get("type", context), mlir::TypeAttr::get(relationalAttribute.type)});
   p << "(" << mlir::DictionaryAttr::get(context, relAttrDefProps) << ")";
   Attribute fromExisting = attr.getFromExisting();
   if (fromExisting) {
      ArrayAttr fromExistingArr = fromExisting.dyn_cast_or_null<ArrayAttr>();
      p << "=";
      printAttributeRefArr(p, fromExistingArr);
   }
}

static ParseResult parseAttributeDefArr(OpAsmParser& parser, OperationState& result, Attribute& attr) {
   std::vector<Attribute> attributes;
   if (parser.parseLSquare()) return failure();
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      mlir::relalg::RelationalAttributeDefAttr attrDefAttr;
      if (parseAttributeDefAttr(parser, result, attrDefAttr)) {
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
static void printAttributeDefArr(OpAsmPrinter& p, ArrayAttr arrayAttr) {
   p << "[";
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      mlir::relalg::RelationalAttributeDefAttr parsedSymbolRefAttr = a.dyn_cast<mlir::relalg::RelationalAttributeDefAttr>();
      printAttributeDefAttr(p, parsedSymbolRefAttr);
   }
   p << "]";
}

static ParseResult parseAttrMapping(OpAsmParser& parser, OperationState& result) {
   if (parser.parseKeyword("mapping") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      mlir::relalg::RelationalAttributeDefAttr attrDefAttr;
      if (parseAttributeDefAttr(parser, result, attrDefAttr)) {
         return failure();
      }
      mapping.push_back(attrDefAttr);
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   result.addAttribute("mapping", mlir::ArrayAttr::get(parser.getBuilder().getContext(), mapping));
   return success();
}
static void printMapping(OpAsmPrinter& p, Attribute mapping) {
   p << " mapping: {";
   auto first = true;
   for (auto attr : mapping.dyn_cast_or_null<ArrayAttr>()) {
      auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      printAttributeDefAttr(p, relationDefAttr);
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
      if (parseAttributeDefAttr(parser, result, attrDefAttr)) {
         return failure();
      }
      columns.push_back({Identifier::get(colName, parser.getBuilder().getContext()), attrDefAttr});
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   result.addAttribute("columns", mlir::DictionaryAttr::get(parser.getBuilder().getContext(), columns));
   return parser.addTypeToList(mlir::relalg::TupleStreamType::get(parser.getBuilder().getContext()), result.types);
}
static void print(OpAsmPrinter& p, relalg::BaseTableOp& op) {
   p << " ";
   p.printSymbolName(op.sym_name());
   if (op->getAttrs().size() > 1) p << ' ';
   p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"sym_name", "columns"});
   p << " columns: {";
   auto first = true;
   for (auto mapping : op.columns()) {
      auto [column_name, attr] = mapping;
      auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      p << column_name << " => ";
      printAttributeDefAttr(p, relationDefAttr);
   }
   p << "}";
}

///////////////////////////////////////////////////////////////////////////////////
// GetAttrOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseGetAttrOp(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::OperandType inputTuple;
   ::mlir::Type resultType;
   if (parser.parseOperand(inputTuple)) {
      return failure();
   }
   if (parser.resolveOperand(inputTuple, mlir::relalg::TupleType::get(parser.getBuilder().getContext()), result.operands)) {
      return failure();
   }
   mlir::relalg::RelationalAttributeRefAttr refAttr;
   if (parseAttributeRefAttr(parser, result, refAttr)) {
      return failure();
   }
   result.addAttribute("attr", refAttr);
   if (parser.parseColon()) {
      return ::failure();
   }
   if (parser.parseType(resultType)) {
      return ::failure();
   }
   return parser.addTypeToList(resultType, result.types);
}
static void print(OpAsmPrinter& p, relalg::GetAttrOp& op) {
   p << " " << op.tuple() << " ";
   p.printAttributeWithoutType(op.attr().getName());
   p << " : ";
   p << op.getType();
}

///////////////////////////////////////////////////////////////////////////////////
// GetScalarOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseGetScalarOp(OpAsmParser& parser, OperationState& result) {
   ::mlir::Type resultType;

   mlir::relalg::RelationalAttributeRefAttr refAttr;
   if (parseAttributeRefAttr(parser, result, refAttr)) {
      return failure();
   }
   result.addAttribute("attr", refAttr);
   parseRelationalInputs(parser, result, 1);
   if (parser.parseColon()) {
      return ::failure();
   }
   if (parser.parseType(resultType)) {
      return ::failure();
   }
   return parser.addTypeToList(resultType, result.types);
}
static void print(OpAsmPrinter& p, relalg::GetScalarOp& op) {
   p  << " ";
   p.printAttributeWithoutType(op.attr().getName());

   p << " " << op.rel();
   p << " : ";
   p << op.getType();
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
   if (parseAttributeDefAttr(parser, result, defAttr)) {
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
   printAttributeDefAttr(p, op.attr());
   p << " " << op.val();
}

///////////////////////////////////////////////////////////////////////////////////
// SelectionOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseSelectionOp(OpAsmParser& parser, OperationState& result) {
   parseRelationalInputs(parser, result, 1);
   parseCustomRegion(parser, result);
   parser.parseOptionalAttrDictWithKeyword(result.attributes);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::SelectionOp& op) {
   p  << " " << op.rel() << " ";
   printCustomRegion(p, op.getRegion());
   p.printOptionalAttrDictWithKeyword(op->getAttrs());
}

///////////////////////////////////////////////////////////////////////////////////
// MapOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseMapOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());
   parseRelationalInputs(parser, result, 1);
   parseCustomRegion(parser, result);
   parser.parseOptionalAttrDictWithKeyword(result.attributes);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::MapOp& op) {
   p  << " ";
   p.printSymbolName(op.sym_name());
   p << " " << op.rel() << " ";
   printCustomRegion(p, op.getRegion());
   p.printOptionalAttrDictWithKeyword(op->getAttrs(), {"sym_name"});
}

///////////////////////////////////////////////////////////////////////////////////
// AggregationOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseAggregationOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());
   parseRelationalInputs(parser, result, 1);
   Attribute groupByAttrs;
   parseAttributeRefArr(parser, result, groupByAttrs);
   result.addAttribute("group_by_attrs", groupByAttrs);
   parseCustomRegion(parser, result);
   parser.parseOptionalAttrDictWithKeyword(result.attributes);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::AggregationOp& op) {
   p  << " ";
   p.printSymbolName(op.sym_name());
   p << " " << op.rel() << " ";
   printAttributeRefArr(p, op.group_by_attrs());
   p << " ";
   printCustomRegion(p, op.getRegion());
   p.printOptionalAttrDictWithKeyword(op->getAttrs(), {"group_by_attrs", "sym_name"});
}

///////////////////////////////////////////////////////////////////////////////////
// AggrFuncOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseAggrFuncOp(OpAsmParser& parser, OperationState& result) {
   mlir::relalg::RelationalAttributeRefAttr refAttr;
   parseAggrFn(parser, result);
   if (parseAttributeRefAttr(parser, result, refAttr)) {
      return failure();
   }
   result.addAttribute("attr", refAttr);
   parseRelationalInputs(parser, result, 1);
   Type resultType;
   if (parser.parseColonType(resultType)) {
      return ::failure();
   }
   return parser.addTypeToList(resultType, result.types);
}
static void print(OpAsmPrinter& p, relalg::AggrFuncOp& op) {
   std::string fn(mlir::relalg::stringifyEnum(op.fn()));
   p << " " << fn << " ";
   p.printAttributeWithoutType(op.attr().getName());
   p << " " << op.rel();
   p << " : ";
   p << op.getType();
}

///////////////////////////////////////////////////////////////////////////////////
// MaterializeOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseMaterializeOp(OpAsmParser& parser, OperationState& result) {
   parseRelationalInputs(parser, result, 1);
   Attribute attrs;
   parseAttributeRefArr(parser, result, attrs);
   result.addAttribute("attrs", attrs);
   ArrayAttr columns;
   if (parser.parseEqual() || parser.parseGreater()) {
      return failure();
   }
   parser.parseAttribute(columns);
   result.addAttribute("columns", columns);

   mlir::db::TableType tableType;
   if (parser.parseColonType(tableType)) {
      return failure();
   }
   return parser.addTypeToList(tableType, result.types);
}
static void print(OpAsmPrinter& p, relalg::MaterializeOp& op) {
   p  << " " << op.rel() << " ";
   printAttributeRefArr(p, op.attrs());
   p << " => " << op.columns();
   p << " : " << op.getType();
}
///////////////////////////////////////////////////////////////////////////////////
// TmpOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseTmpOp(OpAsmParser& parser, OperationState& result) {
   parseRelationalInputs(parser, result, 1);
   Attribute attrs;
   parseAttributeRefArr(parser, result, attrs);
   result.addAttribute("attrs", attrs);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::TmpOp& op) {
   p  << " " << op.rel() << " ";
   printAttributeRefArr(p, op.attrs());
}
///////////////////////////////////////////////////////////////////////////////////
// InnerJoinOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseInnerJoinOp(OpAsmParser& parser, OperationState& result) {
   parseRelationalInputs(parser, result, 2);
   parseCustomRegion(parser, result);
   parser.parseOptionalAttrDictWithKeyword(result.attributes);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::InnerJoinOp& op) {
   p  << " " << op.left() << ", " << op.right() << " ";
   printCustomRegion(p, op.getRegion());
   p << " ";
   p.printOptionalAttrDictWithKeyword(op->getAttrs());
}
///////////////////////////////////////////////////////////////////////////////////
// FullOuterJoinOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseFullOuterJoinOp(OpAsmParser& parser, OperationState& result) {
   parseRelationalInputs(parser, result, 2);
   parseCustomRegion(parser, result);
   parser.parseOptionalAttrDictWithKeyword(result.attributes);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::FullOuterJoinOp& op) {
   p  << " " << op.left() << ", " << op.right() << " ";
   printCustomRegion(p, op.getRegion());
   p << " ";
   p.printOptionalAttrDictWithKeyword(op->getAttrs());
}
///////////////////////////////////////////////////////////////////////////////////
// OuterJoinOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseOuterJoinOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 2);
   parseCustomRegion(parser, result);
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());
   parseAttrMapping(parser, result);
   parser.parseOptionalAttrDictWithKeyword(result.attributes);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::OuterJoinOp& op) {
   p  << " ";
   p.printSymbolName(op.sym_name());
   p << " " << op.left() << ", " << op.right() << " ";
   printCustomRegion(p, op.getRegion());
   printMapping(p, op.mapping());
   p << " ";
   p.printOptionalAttrDictWithKeyword(op->getAttrs(), {"mapping", "sym_name"});
}
///////////////////////////////////////////////////////////////////////////////////
// SingleJoinOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseSingleJoinOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 2);
   parseCustomRegion(parser, result);
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());
   parseAttrMapping(parser, result);
   parser.parseOptionalAttrDictWithKeyword(result.attributes);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::SingleJoinOp& op) {
   p  << " ";
   p.printSymbolName(op.sym_name());
   p << " " << op.left() << ", " << op.right() << " ";
   printCustomRegion(p, op.getRegion());
   printMapping(p, op.mapping());
   p << " ";
   p.printOptionalAttrDictWithKeyword(op->getAttrs(), {"mapping", "sym_name"});
}
///////////////////////////////////////////////////////////////////////////////////
// NonCommutativeJoins: SemiJoin,AntiSemiJoin
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseNonCommutativeJoin(OpAsmParser& parser, OperationState& result) {
   if (parseRelationalInputs(parser, result, 2) ||
       parseCustomRegion(parser, result) ||
       addRelationOutput(parser, result) || parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return failure();
   }
   return success();
}
static void printNonCommutativeJoin(Operation* op, OpAsmPrinter& p) {
   p << " " << op->getOperand(0) << ", " << op->getOperand(1) << " ";
   printCustomRegion(p, op->getRegion(0));
   p << " ";
   p.printOptionalAttrDictWithKeyword(op->getAttrs());
}

///////////////////////////////////////////////////////////////////////////////////
// MarkJoinOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseMarkJoinOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());

   relalg::RelationalAttributeDefAttr defAttr;
   parseAttributeDefAttr(parser, result, defAttr);
   result.addAttribute("markattr", defAttr);
   parseRelationalInputs(parser, result, 2);
   parseCustomRegion(parser, result);
   addRelationOutput(parser, result);
   return parser.parseOptionalAttrDictWithKeyword(result.attributes);
}
static void print(OpAsmPrinter& p, relalg::MarkJoinOp& op) {
   p  << " ";
   p << " ";
   p.printSymbolName(op.sym_name());
   p << " ";
   printAttributeDefAttr(p, op.markattr());
   p << " " << op.left() << ", " << op.right() << " ";
   printCustomRegion(p, op.getRegion());
   p.printOptionalAttrDictWithKeyword(op->getAttrs(), {"markattr", "sym_name"});
}
///////////////////////////////////////////////////////////////////////////////////
// CollectionJoinOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseCollectionJoinOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());
   Attribute attrs;
   parseAttributeRefArr(parser, result, attrs);
   result.addAttribute("attrs", attrs);
   relalg::RelationalAttributeDefAttr defAttr;
   parseAttributeDefAttr(parser, result, defAttr);
   result.addAttribute("collAttr", defAttr);
   parseRelationalInputs(parser, result, 2);
   parseCustomRegion(parser, result);
   addRelationOutput(parser, result);
   return parser.parseOptionalAttrDictWithKeyword(result.attributes);
}
static void print(OpAsmPrinter& p, relalg::CollectionJoinOp& op) {
   p  << " ";
   p << " ";
   p.printSymbolName(op.sym_name());
   p << " ";
   printAttributeRefArr(p, op.attrs());
   p << " ";
   printAttributeDefAttr(p, op.collAttr());
   p << " " << op.left() << ", " << op.right() << " ";
   printCustomRegion(p, op.getRegion());
   p.printOptionalAttrDictWithKeyword(op->getAttrs(), {"collAttr", "sym_name", "attrs"});
}
///////////////////////////////////////////////////////////////////////////////////
// ProjectionOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseProjectionOp(OpAsmParser& parser, OperationState& result) {
   Attribute attrs;
   if (parseSetSemantic(parser, result)) {
      return failure();
   }
   parseAttributeRefArr(parser, result, attrs);
   result.addAttribute("attrs", attrs);
   parseRelationalInputs(parser, result, 1);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::ProjectionOp& op) {
   p  << " ";
   printSetSemantic(p, op.set_semantic());
   p << " ";
   printAttributeRefArr(p, op.attrs());
   p << " " << op.rel();
}
///////////////////////////////////////////////////////////////////////////////////
// SortOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseSortOp(OpAsmParser& parser, OperationState& result) {
   parseRelationalInputs(parser, result, 1);
   parseSortSpecs(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::SortOp& op) {
   p  << " " << op.rel() << " ";
   printSortSpecs(p, op.sortspecs());
}
///////////////////////////////////////////////////////////////////////////////////
// TopKOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseTopKOp(OpAsmParser& parser, OperationState& result) {
   mlir::IntegerAttr integerAttr;
   parser.parseAttribute(integerAttr, parser.getBuilder().getI32Type());
   result.addAttribute("rows", integerAttr);
   parseRelationalInputs(parser, result, 1);
   parseSortSpecs(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::TopKOp& op) {
   p  << " " << op.rows() << " " << op.rel() << " ";
   printSortSpecs(p, op.sortspecs());
}
///////////////////////////////////////////////////////////////////////////////////
// UnionOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseUnionOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());
   if (parseSetSemantic(parser, result)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 2);
   parseAttrMapping(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::UnionOp& op) {
   p  << " ";
   p.printSymbolName(op.sym_name());
   p << " ";
   printSetSemantic(p, op.set_semantic());
   p << " " << op.left() << ", " << op.right() << " ";
   printMapping(p, op.mapping());
}
///////////////////////////////////////////////////////////////////////////////////
// IntersectOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseIntersectOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());

   if (parseSetSemantic(parser, result)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 2);
   parseAttrMapping(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::IntersectOp& op) {
   p  << " ";
   p.printSymbolName(op.sym_name());
   p << " ";
   printSetSemantic(p, op.set_semantic());
   p << " " << op.left() << ", " << op.right() << " ";
   printMapping(p, op.mapping());
}
///////////////////////////////////////////////////////////////////////////////////
// ExceptOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseExceptOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());

   if (parseSetSemantic(parser, result)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 2);
   parseAttrMapping(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::ExceptOp& op) {
   p  << " ";
   p.printSymbolName(op.sym_name());
   p << " ";
   printSetSemantic(p, op.set_semantic());
   p << " " << op.left() << ", " << op.right() << " ";
   printMapping(p, op.mapping());
}

///////////////////////////////////////////////////////////////////////////////////
// ConstRelationOp
///////////////////////////////////////////////////////////////////////////////////
static void print(OpAsmPrinter& p, relalg::ConstRelationOp& op) {
   p << " ";
   p.printSymbolName(op.sym_name());
   if (op->getAttrs().size() > 1) p << ' ';
   p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"sym_name", "attributes", "values"});
   p << " attributes: ";
   printAttributeDefArr(p, op.attributes());
   p << " values: " << op.values();
}

static ParseResult parseConstRelationOp(OpAsmParser& parser,
                                        OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());
   Attribute attributes;
   if (parser.parseKeyword("attributes") || parser.parseColon() || parseAttributeDefArr(parser, result, attributes)) {
      return failure();
   }

   result.addAttribute("attributes", attributes);
   Attribute valueAttr;
   if (parser.parseKeyword("values") || parser.parseColon() || parser.parseAttribute(valueAttr, "values", result.attributes))
      return failure();

   return addRelationOutput(parser, result);
}

///////////////////////////////////////////////////////////////////////////////////
// RenamingOp
///////////////////////////////////////////////////////////////////////////////////
static void print(OpAsmPrinter& p, relalg::RenamingOp& op) {
   p << " ";
   p.printSymbolName(op.sym_name());
   p << " " << op.rel();
   if (op->getAttrs().size() > 1) p << ' ';
   p << " renamed: ";
   printAttributeDefArr(p, op.attributes());
   p.printOptionalAttrDictWithKeyword(op->getAttrs(), /*elidedAttrs=*/{"sym_name", "attributes"});
}

static ParseResult parseRenamingOp(OpAsmParser& parser,
                                   OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   getRelationalAttributeManager(parser).setCurrentScope(nameAttr.getValue());
   parseRelationalInputs(parser, result, 1);
   Attribute attributes;
   if (parser.parseKeyword("renamed") || parser.parseColon() || parseAttributeDefArr(parser, result, attributes)) {
      return failure();
   }

   result.addAttribute("attributes", attributes);
   parser.parseOptionalAttrDictWithKeyword(result.attributes);
   return addRelationOutput(parser, result);
}

///////////////////////////////////////////////////////////////////////////////////
// InOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseInOp(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::OperandType val;
   ::mlir::Type valType;
   if (parser.parseOperand(val) || parser.parseColonType(valType) || parser.parseComma()) {
      return failure();
   }
   if (parser.resolveOperand(val, valType, result.operands)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 1);
   return parser.addTypeToList(db::BoolType::get(parser.getBuilder().getContext()), result.types);
}
static void print(OpAsmPrinter& p, relalg::InOp& op) {
   p << " " << op.val() << " : " << op.val().getType() << ", " << op.rel();
}
///////////////////////////////////////////////////////////////////////////////////
// MaterializeOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseGetListOp(OpAsmParser& parser, OperationState& result) {
   parseRelationalInputs(parser, result, 1);
   Attribute attrs;
   parseAttributeRefArr(parser, result, attrs);
   result.addAttribute("attrs", attrs);

   mlir::db::CollectionType collectionType;
   if (parser.parseColonType(collectionType)) {
      return failure();
   }
   return parser.addTypeToList(collectionType, result.types);
}
static void print(OpAsmPrinter& p, relalg::GetListOp& op) {
   p  << " " << op.rel() << " ";
   printAttributeRefArr(p, op.attrs());
   p << " : " << op.getType();
}
#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"