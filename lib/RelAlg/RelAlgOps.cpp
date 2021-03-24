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

::mlir::ParseResult parseOuterJoinType(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   ::mlir::IntegerAttr typeAttr;

   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr, {"left", "right", "full"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
         parser.parseOptionalAttribute(attrVal,
                                       parser.getBuilder().getNoneType(),
                                       "type", attrStorage);
      if (parseResult.hasValue()) {
         if (failed(*parseResult))
            return ::mlir::failure();
         attrStr = attrVal.getValue();
      } else {
         return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'type' [left, right, full]");
      }
   }
   if (!attrStr.empty()) {
      auto attrOptional = ::mlir::relalg::symbolizeOuterJoinType(attrStr);
      if (!attrOptional)
         return parser.emitError(loc, "invalid ")
            << "type attribute specification: \"" << attrStr << '"';
      ;

      typeAttr = parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue()));
      result.addAttribute("type", typeAttr);
   }
   return success();
}
::mlir::ParseResult parseAggrFn(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   ::mlir::IntegerAttr typeAttr;

   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr, {"min", "max", "sum","avg","count"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
         parser.parseOptionalAttribute(attrVal,
                                       parser.getBuilder().getNoneType(),
                                       "type", attrStorage);
      if (parseResult.hasValue()) {
         if (failed(*parseResult))
            return ::mlir::failure();
         attrStr = attrVal.getValue();
      } else {
         return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'type' [min, max, sum, avg, count]");
      }
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
::mlir::ParseResult parseSortSpec(::mlir::OpAsmParser& parser,mlir::relalg::SortSpec& spec) {
   ::mlir::IntegerAttr typeAttr;

   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr, {"desc", "asc"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
         parser.parseOptionalAttribute(attrVal,
                                       parser.getBuilder().getNoneType(),
                                       "type", attrStorage);
      if (parseResult.hasValue()) {
         if (failed(*parseResult))
            return ::mlir::failure();
         attrStr = attrVal.getValue();
      } else {
         return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'sortSpec' [desc,asc]");
      }
   }
   if (!attrStr.empty()) {
      auto spec_ = ::mlir::relalg::symbolizeSortSpec(attrStr);
      if (!spec_)
         return parser.emitError(loc, "invalid ")
            << "type attribute specification: \"" << attrStr << '"';
      spec=spec_.getValue();
   }
   return success();
}
::mlir::ParseResult parseSetSemantic(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   ::mlir::IntegerAttr typeAttr;

   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr, {"distinct", "all"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
         parser.parseOptionalAttribute(attrVal,
                                       parser.getBuilder().getNoneType(),
                                       "set_semantic", attrStorage);
      if (parseResult.hasValue()) {
         if (failed(*parseResult))
            return ::mlir::failure();
         attrStr = attrVal.getValue();
      } else {
         return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'type' [distinct, all]");
      }
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
::mlir::ParseResult parseJoinType(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   ::mlir::IntegerAttr typeAttr;

   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr, {"inner", "semi", "antisemi"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
         parser.parseOptionalAttribute(attrVal,
                                       parser.getBuilder().getNoneType(),
                                       "type", attrStorage);
      if (parseResult.hasValue()) {
         if (failed(*parseResult))
            return ::mlir::failure();
         attrStr = attrVal.getValue();
      } else {
         return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'type' [inner, semi, antisemi]");
      }
   }
   if (!attrStr.empty()) {
      auto attrOptional = ::mlir::relalg::symbolizeNormalJoinType(attrStr);
      if (!attrOptional)
         return parser.emitError(loc, "invalid ")
            << "type attribute specification: \"" << attrStr << '"';
      ;

      typeAttr = parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue()));
      result.addAttribute("type", typeAttr);
   }
   return success();
}
static void createAttributeRefAttr(MLIRContext* context, const SymbolRefAttr& parsedSymbolRefAttr, relalg::RelationalAttributeRefAttr& attr) {
   attr = relalg::RelationalAttributeRefAttr::get(context, parsedSymbolRefAttr,
                                                  std::shared_ptr<relalg::RelationalAttribute>());
}
static ParseResult parseAttributeRefAttr(OpAsmParser& parser, OperationState& result, mlir::relalg::RelationalAttributeRefAttr& attr) {
   ::mlir::SymbolRefAttr parsedSymbolRefAttr;
   if (parser.parseAttribute(parsedSymbolRefAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   createAttributeRefAttr(parser.getBuilder().getContext(), parsedSymbolRefAttr, attr);
   return success();
}
static ParseResult parseCustomRegion(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::OperandType pred_argument;
   Type pred_arg_type;
   SmallVector<OpAsmParser::OperandType, 4> regionArgs;
   SmallVector<Type, 4> argTypes;
   if (parser.parseLParen()) {
      return failure();
   }
   while (true) {
      if (!parser.parseOptionalRParen()) {
         break;
      }
      if (parser.parseRegionArgument(pred_argument) || parser.parseColonType(pred_arg_type)) {
         return failure();
      }
      regionArgs.push_back(pred_argument);
      argTypes.push_back(pred_arg_type);
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
   ArrayAttr parsed_attr;
   std::vector<Attribute> attributes;
   if (parser.parseAttribute(parsed_attr, parser.getBuilder().getType<::mlir::NoneType>())) {
      return failure();
   }
   for (auto a : parsed_attr) {
      SymbolRefAttr parsedSymbolRefAttr = a.dyn_cast<SymbolRefAttr>();
      mlir::relalg::RelationalAttributeRefAttr attr;
      createAttributeRefAttr(parser.getBuilder().getContext(), parsedSymbolRefAttr, attr);
      attributes.push_back(attr);
   }
   attr = ArrayAttr::get(parser.getBuilder().getContext(), attributes);
   return success();
}
static void printAttributeRefArr(OpAsmPrinter& p, ArrayAttr arrayAttr) {
   ArrayAttr parsed_attr;
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
      mlir::relalg::RelationalAttributeRefAttr attr_ref_attr;
      if (parser.parseLParen()||parseAttributeRefAttr(parser, result, attr_ref_attr)||parser.parseComma()) {
         return failure();
      }
      mlir::relalg::SortSpec spec;
      if(parseSortSpec(parser,spec)||parser.parseRParen()){
         return failure();
      }
      mapping.push_back(mlir::relalg::SortSpecificationAttr::get(parser.getBuilder().getContext(),attr_ref_attr,spec));
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRSquare()) { return failure(); }
      break;
   }
   result.addAttribute("sortspecs", mlir::ArrayAttr::get(parser.getBuilder().getContext(), mapping));
   return success();
}
static void printSortSpecs(OpAsmPrinter& p, ArrayAttr arrayAttr) {
   ArrayAttr parsed_attr;
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
      p <<"("<<sortSpecificationAttr.getAttr().getName()<<","<<mlir::relalg::stringifySortSpec(sortSpecificationAttr.getSortSpec())<<")";
   }
   p << "]";
}
static ParseResult addRelationOutput(OpAsmParser& parser, OperationState& result) {
   return parser.addTypeToList(mlir::relalg::RelationType::get(parser.getBuilder().getContext()), result.types);
}
static ParseResult parseRelationalInputs(OpAsmParser& parser, OperationState& result, size_t inputs) {
   SmallVector<OpAsmParser::OperandType, 4> operands;
   if (parser.parseOperandList(operands)) {
      return failure();
   }
   if (parser.resolveOperands(operands, mlir::relalg::RelationType::get(parser.getBuilder().getContext()), result.operands)) {
      return failure();
   }
   return success();
}

static ParseResult parseAttributeDefAttr(OpAsmParser& parser, OperationState& result, mlir::relalg::RelationalAttributeDefAttr& attr) {
   SymbolRefAttr attr_symbolAttr;
   if (parser.parseAttribute(attr_symbolAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   std::string attr_name(attr_symbolAttr.getLeafReference());
   if (parser.parseLParen()) { return failure(); }
   DictionaryAttr dictAttr;
   if (parser.parseAttribute(dictAttr)) { return failure(); }
   auto prop_type = dictAttr.get("type").dyn_cast<TypeAttr>().getValue().dyn_cast<mlir::db::DBType>();
   auto relationalAttribute = std::make_shared<mlir::relalg::RelationalAttribute>(prop_type);
   Attribute from_existing;
   if (parser.parseRParen()) { return failure(); }
   if (parser.parseOptionalEqual().succeeded()) {
      if (parseAttributeRefArr(parser, result, from_existing)) {
         return failure();
      }
   }
   attr = mlir::relalg::RelationalAttributeDefAttr::get(parser.getBuilder().getContext(), attr_name, relationalAttribute, from_existing);
   return success();
}
static void printAttributeDefAttr(OpAsmPrinter& p, mlir::relalg::RelationalAttributeDefAttr attr) {
   p.printSymbolName(attr.getName());
   std::vector<mlir::NamedAttribute> rel_attr_def_props;
   MLIRContext* context = attr.getContext();
   const mlir::relalg::RelationalAttribute& relationalAttribute = attr.getRelationalAttribute();
   rel_attr_def_props.push_back({mlir::Identifier::get("type", context), mlir::TypeAttr::get(relationalAttribute.type)});
   p << "(" << mlir::DictionaryAttr::get(context, rel_attr_def_props) << ")";
   Attribute from_existing = attr.getFromExisting();
   if (from_existing) {
      ArrayAttr from_existing_arr = from_existing.dyn_cast_or_null<ArrayAttr>();
      p << "=";
      printAttributeRefArr(p, from_existing_arr);
   }
}

static ParseResult parseAttrMapping(OpAsmParser& parser, OperationState& result) {
   if (parser.parseKeyword("mapping") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      mlir::relalg::RelationalAttributeDefAttr attr_def_attr;
      if (parseAttributeDefAttr(parser, result, attr_def_attr)) {
         return failure();
      }
      mapping.push_back(attr_def_attr);
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
      auto relation_def_attr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      printAttributeDefAttr(p, relation_def_attr);
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
   if (parser.parseOptionalAttrDict(result.attributes)) return failure();
   if (parser.parseKeyword("columns") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::NamedAttribute> columns;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      StringRef col_name;
      if (parser.parseKeyword(&col_name)) { return failure(); }
      if (parser.parseEqual() || parser.parseGreater()) { return failure(); }
      mlir::relalg::RelationalAttributeDefAttr attr_def_attr;
      if (parseAttributeDefAttr(parser, result, attr_def_attr)) {
         return failure();
      }
      columns.push_back({Identifier::get(col_name, parser.getBuilder().getContext()), attr_def_attr});
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   result.addAttribute("columns", mlir::DictionaryAttr::get(parser.getBuilder().getContext(), columns));
   return parser.addTypeToList(mlir::relalg::RelationType::get(parser.getBuilder().getContext()), result.types);
}
static void print(OpAsmPrinter& p, relalg::BaseTableOp& op) {
   p << op.getOperationName();
   p << " ";
   p.printSymbolName(op.sym_name());
   if (op->getAttrs().size() > 1) p << ' ';
   p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"sym_name", "columns"});
   p << " columns: {";
   auto first = true;
   for (auto mapping : op.columns()) {
      auto [column_name, attr] = mapping;
      auto relation_def_attr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      p << column_name << " => ";
      printAttributeDefAttr(p, relation_def_attr);
   }
   p << "}";
}

///////////////////////////////////////////////////////////////////////////////////
// GetAttrOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseGetAttrOp(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::OperandType input_tuple;
   ::mlir::Type resultType;
   if (parser.parseOperand(input_tuple)) {
      return failure();
   }
   if (parser.resolveOperand(input_tuple, mlir::relalg::TupleType::get(parser.getBuilder().getContext()), result.operands)) {
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
   p << op.getOperationName();
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
   parseRelationalInputs(parser,result,1);
   if (parser.parseColon()) {
      return ::failure();
   }
   if (parser.parseType(resultType)) {
      return ::failure();
   }
   return parser.addTypeToList(resultType, result.types);
}
static void print(OpAsmPrinter& p, relalg::GetScalarOp& op) {
   p << op.getOperationName()<< " ";
   p.printAttributeWithoutType(op.attr().getName());

   p << " " << op.rel();
   p << " : ";
   p << op.getType();
}


///////////////////////////////////////////////////////////////////////////////////
// AddAttrOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseAddAttrOp(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::OperandType input;
   Type inputType;
   mlir::relalg::RelationalAttributeDefAttr defAttr;
   if (parseAttributeDefAttr(parser, result, defAttr)) {
      return failure();
   }
   result.addAttribute("attr", defAttr);
   if (parser.parseOperand(input)) {
      return failure();
   }
   inputType = defAttr.getRelationalAttribute().type;
   if (parser.resolveOperand(input, inputType, result.operands)) {
      return failure();
   }
   return success();
}
static void print(OpAsmPrinter& p, relalg::AddAttrOp& op) {
   p << op.getOperationName();
   p << " ";
   printAttributeDefAttr(p, op.attr());
   p << " " << op.val();
}

///////////////////////////////////////////////////////////////////////////////////
// SelectionOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseSelectionOp(OpAsmParser& parser, OperationState& result) {
   parseRelationalInputs(parser, result, 1);
   parseCustomRegion(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::SelectionOp& op) {
   p << op.getOperationName() << " " << op.rel() << " ";
   printCustomRegion(p, op.getRegion());
}
Region &mlir::relalg::SelectionOp::getLoopBody() { return getRegion(); }

bool mlir::relalg::SelectionOp::isDefinedOutsideOfLoop(Value value) {
   return !getRegion().isAncestor(value.getParentRegion());
}

LogicalResult mlir::relalg::SelectionOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
   for (auto op : ops)
      op->moveBefore(*this);
   return success();
}

///////////////////////////////////////////////////////////////////////////////////
// MapOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseMapOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 1);
   parseCustomRegion(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::MapOp& op) {
   p << op.getOperationName() << " ";
   p.printSymbolName(op.sym_name());
   p << " " << op.rel() << " ";
   printCustomRegion(p, op.getRegion());
}
Region &mlir::relalg::MapOp::getLoopBody() { return getRegion(); }

bool mlir::relalg::MapOp::isDefinedOutsideOfLoop(Value value) {
   return !getRegion().isAncestor(value.getParentRegion());
}

LogicalResult mlir::relalg::MapOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
   for (auto op : ops)
      op->moveBefore(*this);
   return success();
}

///////////////////////////////////////////////////////////////////////////////////
// AggregationOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseAggregationOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 1);
   Attribute group_by_attrs;
   parseAttributeRefArr(parser, result, group_by_attrs);
   result.addAttribute("group_by_attrs", group_by_attrs);
   parseCustomRegion(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::AggregationOp& op) {
   p << op.getOperationName() << " ";
   p.printSymbolName(op.sym_name());
   p << " " << op.rel() << " ";
   printAttributeRefArr(p, op.group_by_attrs());
   p << " ";
   printCustomRegion(p, op.getRegion());
}
Region &mlir::relalg::AggregationOp::getLoopBody() { return getRegion(); }

bool mlir::relalg::AggregationOp::isDefinedOutsideOfLoop(Value value) {
   return !getRegion().isAncestor(value.getParentRegion());
}

LogicalResult mlir::relalg::AggregationOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
   for (auto op : ops)
      op->moveBefore(*this);
   return success();
}

///////////////////////////////////////////////////////////////////////////////////
// AggrFuncOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseAggrFuncOp(OpAsmParser& parser, OperationState& result) {
   mlir::relalg::RelationalAttributeRefAttr refAttr;
   parseAggrFn(parser,result);
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
   p << op.getOperationName();
   p << " "<<fn<<" ";
   p.printAttributeWithoutType(op.attr().getName());
   p << " " << op.rel();
   p << " : ";
   p << op.getType();
}
///////////////////////////////////////////////////////////////////////////////////
// ForEachOp
///////////////////////////////////////////////////////////////////////////////////
static void print(mlir::OpAsmPrinter& p, mlir::relalg::ForEachOp op) {
   p << mlir::relalg::ForEachOp::getOperationName() << " " << op.rel() << " ";
   printAttributeRefArr(p, op.attrs());
   printCustomRegion(p, op.region());
}
static mlir::ParseResult parseForEachOp(mlir::OpAsmParser& parser, mlir::OperationState& result) {
   if (parseRelationalInputs(parser, result, 1)) {
      return failure();
   }
   Attribute attrs;
   if (parseAttributeRefArr(parser, result, attrs)) {
      return failure();
   }
   result.addAttribute("attrs", attrs);
   if (parseCustomRegion(parser, result)) {
      return failure();
   }
   return success();
}

///////////////////////////////////////////////////////////////////////////////////
// MaterializeOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseMaterializeOp(OpAsmParser& parser, OperationState& result) {
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
static void print(OpAsmPrinter& p, relalg::MaterializeOp& op) {
   p << op.getOperationName() << " " << op.rel() << " ";
   printAttributeRefArr(p, op.attrs());
   p << " : " << op.getType();
}

///////////////////////////////////////////////////////////////////////////////////
// JoinOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseJoinOp(OpAsmParser& parser, OperationState& result) {
   parseJoinType(parser, result);
   parseRelationalInputs(parser, result, 2);
   parseCustomRegion(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::JoinOp& op) {
   std::string jt(mlir::relalg::stringifyEnum(op.type()));
   p << op.getOperationName() << " " << jt << " " << op.left() << ", " << op.right();
   printCustomRegion(p, op.getRegion());
}
Region &mlir::relalg::JoinOp::getLoopBody() { return getRegion(); }

bool mlir::relalg::JoinOp::isDefinedOutsideOfLoop(Value value) {
   return !getRegion().isAncestor(value.getParentRegion());
}

LogicalResult mlir::relalg::JoinOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
   for (auto op : ops)
      op->moveBefore(*this);
   return success();
}
///////////////////////////////////////////////////////////////////////////////////
// OuterJoinOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseOuterJoinOp(OpAsmParser& parser, OperationState& result) {
   parseOuterJoinType(parser, result);
   parseRelationalInputs(parser, result, 2);
   parseCustomRegion(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::OuterJoinOp& op) {
   std::string ojt(mlir::relalg::stringifyEnum(op.type()));
   p << op.getOperationName() << " " << ojt << " " << op.left() << ", " << op.right();
   printCustomRegion(p, op.getRegion());
}
Region &mlir::relalg::OuterJoinOp::getLoopBody() { return getRegion(); }

bool mlir::relalg::OuterJoinOp::isDefinedOutsideOfLoop(Value value) {
   return !getRegion().isAncestor(value.getParentRegion());
}

LogicalResult mlir::relalg::OuterJoinOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
   for (auto op : ops)
      op->moveBefore(*this);
   return success();
}
///////////////////////////////////////////////////////////////////////////////////
// OuterJoinOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseSingleJoinOp(OpAsmParser& parser, OperationState& result) {
   parseOuterJoinType(parser, result);
   parseRelationalInputs(parser, result, 2);
   parseCustomRegion(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::SingleJoinOp& op) {
   std::string ojt(mlir::relalg::stringifyEnum(op.type()));
   p << op.getOperationName() << " " << ojt << " " << op.left() << ", " << op.right();
   printCustomRegion(p, op.getRegion());
}
Region &mlir::relalg::SingleJoinOp::getLoopBody() { return getRegion(); }

bool mlir::relalg::SingleJoinOp::isDefinedOutsideOfLoop(Value value) {
   return !getRegion().isAncestor(value.getParentRegion());
}

LogicalResult mlir::relalg::SingleJoinOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
   for (auto op : ops)
      op->moveBefore(*this);
   return success();
}
///////////////////////////////////////////////////////////////////////////////////
// OuterJoinOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseMarkJoinOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }

   relalg::RelationalAttributeDefAttr defAttr;
   parseAttributeDefAttr(parser,result,defAttr);
   result.addAttribute("markattr",defAttr);
   parseRelationalInputs(parser, result, 2);
   parseCustomRegion(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::MarkJoinOp& op) {
   p << op.getOperationName() << " ";
   p.printSymbolName(op.sym_name());
   p<<" ";
   printAttributeDefAttr(p,op.markattr());
   p<<" " << op.left() << ", " << op.right();
   printCustomRegion(p, op.getRegion());
}
Region &mlir::relalg::MarkJoinOp::getLoopBody() { return getRegion(); }

bool mlir::relalg::MarkJoinOp::isDefinedOutsideOfLoop(Value value) {
   return !getRegion().isAncestor(value.getParentRegion());
}

LogicalResult mlir::relalg::MarkJoinOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
   for (auto op : ops)
      op->moveBefore(*this);
   return success();
}

///////////////////////////////////////////////////////////////////////////////////
// DistinctOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseDistinctOp(OpAsmParser& parser, OperationState& result) {
   Attribute attrs;
   parseAttributeRefArr(parser, result, attrs);
   result.addAttribute("attrs", attrs);
   parseRelationalInputs(parser, result, 1);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::DistinctOp& op) {
   p << op.getOperationName() << " ";
   printAttributeRefArr(p, op.attrs());
   p << " " << op.rel();
}
///////////////////////////////////////////////////////////////////////////////////
// SortOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseSortOp(OpAsmParser& parser, OperationState& result) {
   parseRelationalInputs(parser, result, 1);
   parseSortSpecs(parser,result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::SortOp& op) {
   p << op.getOperationName() << " "<< op.rel()<<" ";
   printSortSpecs(p,op.sortspecs());
}
///////////////////////////////////////////////////////////////////////////////////
// UnionOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseUnionOp(OpAsmParser& parser, OperationState& result) {
   StringAttr nameAttr;
   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
      return failure();
   }
   if (parseSetSemantic(parser, result)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 2);
   parseAttrMapping(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::UnionOp& op) {
   p << op.getOperationName() << " ";
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
   if (parseSetSemantic(parser, result)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 2);
   parseAttrMapping(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::IntersectOp& op) {
   p << op.getOperationName() << " ";
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
   if (parseSetSemantic(parser, result)) {
      return failure();
   }
   parseRelationalInputs(parser, result, 2);
   parseAttrMapping(parser, result);
   return addRelationOutput(parser, result);
}
static void print(OpAsmPrinter& p, relalg::ExceptOp& op) {
   p << op.getOperationName() << " ";
   p.printSymbolName(op.sym_name());
   p << " ";
   printSetSemantic(p, op.set_semantic());
   p << " " << op.left() << ", " << op.right() << " ";
   printMapping(p, op.mapping());
}


///////////////////////////////////////////////////////////////////////////////////
// ConstRelationOp
///////////////////////////////////////////////////////////////////////////////////
static void print(OpAsmPrinter &p, relalg::ConstRelationOp &op) {
   p << op.getOperationName();
   p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

   if (op->getAttrs().size() > 1)
      p << ' ';
   p.printAttributeWithoutType(op.value());
   p << " : " << op.getType();
}

static ParseResult parseConstRelationOp(OpAsmParser &parser,
                                   OperationState &result) {
   Attribute valueAttr;
   if (parser.parseAttribute(valueAttr, "value", result.attributes))
      return failure();

   return addRelationOutput(parser,result);
}

///////////////////////////////////////////////////////////////////////////////////
// InOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseInOp(OpAsmParser& parser, OperationState& result) {
   ::mlir::db::DBType resultType;
   OpAsmParser::OperandType val;
   ::mlir::Type valType;
   if (parser.parseOperand(val)||parser.parseColonType(valType)||parser.parseComma()) {
      return failure();
   }
   if (parser.resolveOperand(val, valType, result.operands)) {
      return failure();
   }
   parseRelationalInputs(parser,result,1);
   return parser.addTypeToList(db::BoolType::get(parser.getBuilder().getContext()), result.types);
}
static void print(OpAsmPrinter& p, relalg::InOp& op) {
   p << op.getOperationName();
   p << " " << op.val() <<" : "<<op.val().getType() <<", "<<op.rel() <<":"<<op.getType();
}
#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"