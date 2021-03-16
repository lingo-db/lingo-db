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
   auto prop_name = dictAttr.get("name").dyn_cast<StringAttr>().getValue();
   auto prop_type = dictAttr.get("type").dyn_cast<TypeAttr>().getValue().dyn_cast<mlir::db::DBType>();
   auto relationalAttribute = std::make_shared<mlir::relalg::RelationalAttribute>(prop_name, prop_type);
   attr = mlir::relalg::RelationalAttributeDefAttr::get(parser.getBuilder().getContext(), attr_name, relationalAttribute);
   if (parser.parseRParen()) { return failure(); }
   return success();
}
static void printAttributeDefAttr(OpAsmPrinter& p, mlir::relalg::RelationalAttributeDefAttr attr) {
   p.printSymbolName(attr.getName());
   std::vector<mlir::NamedAttribute> rel_attr_def_props;
   MLIRContext* context = attr.getContext();
   const mlir::relalg::RelationalAttribute& relationalAttribute = attr.getRelationalAttribute();
   rel_attr_def_props.push_back({mlir::Identifier::get("name", context), mlir::StringAttr::get(context, relationalAttribute.name)});
   rel_attr_def_props.push_back({mlir::Identifier::get("type", context), mlir::TypeAttr::get(relationalAttribute.type)});
   p << "(" << mlir::DictionaryAttr::get(context, rel_attr_def_props) << ")";
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
      if(parseAttributeDefAttr(parser, result, attr_def_attr)){
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
   inputType=defAttr.getRelationalAttribute().type;
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

///////////////////////////////////////////////////////////////////////////////////
// SumAggrFuncOp
///////////////////////////////////////////////////////////////////////////////////
static ParseResult parseSumAggrFuncOp(OpAsmParser& parser, OperationState& result) {
   mlir::relalg::RelationalAttributeRefAttr refAttr;
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
static void print(OpAsmPrinter& p, relalg::SumAggrFuncOp& op) {
   p << op.getOperationName();
   p << " ";
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
   printCustomRegion(p,op.region());
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
#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.cpp.inc"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"