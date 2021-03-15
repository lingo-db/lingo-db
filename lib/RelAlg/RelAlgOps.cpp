#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/IR/OpImplementation.h"
#include <unordered_set>

#include <queue>

using namespace mlir;
static void print(OpAsmPrinter &p, relalg::BaseTableOp &op) {
  p << op.getOperationName();
  p << " ";
  p.printSymbolName(op.sym_name());
  if (op->getAttrs().size() > 1)
    p << ' ';
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"sym_name", "columns"});
  p << " columns: {";
  auto first = true;
  for (auto mapping : op.columns()) {
    auto [column_name, attr] = mapping;
    auto relation_def_attr =
        attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
    if (first) {
      first = false;
    } else {
      p << ", ";
    }
    p << column_name << " => ";
    p.printSymbolName(relation_def_attr.getName());
    std::vector<mlir::NamedAttribute> rel_attr_def_props;
    MLIRContext *context = op->getContext();
    const mlir::relalg::RelationalAttribute &relationalAttribute =
        relation_def_attr.getRelationalAttribute();
    rel_attr_def_props.push_back(
        {mlir::Identifier::get("name", context),
         mlir::StringAttr::get(context, relationalAttribute.name)});
    rel_attr_def_props.push_back(
        {mlir::Identifier::get("type", context),
         mlir::TypeAttr::get(relationalAttribute.type)});

    p << "(" << mlir::DictionaryAttr::get(context, rel_attr_def_props) << ")";
  }
  p << "}";
}

static void print(OpAsmPrinter &p, relalg::GetAttrOp &op) {
  p << op.getOperationName();
  p << " " << op.tuple() << ", ";
  p.printAttributeWithoutType(op.attr().getName());
  p << " : ";
  p << op.getType();
}

static ParseResult parseBaseTableOp(OpAsmParser &parser,
                                    OperationState &result) {
  Attribute valueAttr;
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseKeyword("columns") || parser.parseColon() ||
      parser.parseLBrace())
    return failure();
  std::vector<mlir::NamedAttribute> columns;

  while (true) {
    if (!parser.parseOptionalRBrace()) {
      break;
    }
    StringRef col_name;
    SymbolRefAttr attr_symbolAttr;
    if (parser.parseKeyword(&col_name)) {
      return failure();
    }
    if (parser.parseEqual() || parser.parseGreater()) {
      return failure();
    }
    if (parser.parseAttribute(
            attr_symbolAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
      return failure();
    }
    std::string attr_name(attr_symbolAttr.getLeafReference());
    if (parser.parseLParen()) {
      return failure();
    }
    DictionaryAttr dictAttr;
    if (parser.parseAttribute(dictAttr)) {
      return failure();
    }
    auto prop_name = dictAttr.get("name").dyn_cast<StringAttr>().getValue();
    auto prop_type = dictAttr.get("type")
                         .dyn_cast<TypeAttr>()
                         .getValue()
                         .dyn_cast<mlir::db::DBType>();
    auto relationalAttribute =
        std::make_shared<mlir::relalg::RelationalAttribute>(prop_name,
                                                            prop_type);
    auto attr_def_attr = mlir::relalg::RelationalAttributeDefAttr::get(
        parser.getBuilder().getContext(), attr_name, relationalAttribute);
    columns.push_back(
        {Identifier::get(col_name, parser.getBuilder().getContext()),
         attr_def_attr});
    if (parser.parseRParen()) {
      return failure();
    }
    if (!parser.parseOptionalComma()) {
      continue;
    }
    if (parser.parseRBrace()) {
      return failure();
    }
    break;
  }
  result.addAttribute(
      "columns",
      mlir::DictionaryAttr::get(parser.getBuilder().getContext(), columns));

  // Add the attribute type to the list.
  return parser.addTypeToList(
      mlir::relalg::RelationType::get(parser.getBuilder().getContext()),
      result.types);
}
static ParseResult parseGetAttrOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType input_tuple;
  ::mlir::SymbolRefAttr attr_symbolAttr;
  ::mlir::Type resultType;
  if (parser.parseOperand(input_tuple) || parser.parseComma()) {
    return mlir::failure();
  }
  if (parser.resolveOperand(
          input_tuple,
          mlir::relalg::TupleType::get(parser.getBuilder().getContext()),
          result.operands))
    return failure();
  if (parser.parseAttribute(attr_symbolAttr,
                            parser.getBuilder().getType<::mlir::NoneType>()))
    return ::mlir::failure();
  result.addAttribute(
      "attr", mlir::relalg::RelationalAttributeRefAttr::get(
                  parser.getBuilder().getContext(), attr_symbolAttr,
                  std::shared_ptr<mlir::relalg::RelationalAttribute>()));
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseType(resultType))
    return ::mlir::failure();
  return parser.addTypeToList(resultType, result.types);
}

static ParseResult parseSelectionOp(OpAsmParser &parser,
                                    OperationState &result) {
  OpAsmParser::OperandType input_rel, pred_argument;
  if (parser.parseOperand(input_rel)) {
    return failure();
  }
  Type pred_arg_type;
  if (parser.parseLParen() || parser.parseRegionArgument(pred_argument) ||
      parser.parseColonType(pred_arg_type) || parser.parseRParen()) {
    return failure();
  }
  SmallVector<OpAsmParser::OperandType, 4> regionArgs;
  regionArgs.push_back(pred_argument);
  SmallVector<Type, 4> argTypes;
  argTypes.push_back(pred_arg_type);
  Region *body = result.addRegion();
  if (parser.resolveOperand(
          input_rel,
          mlir::relalg::RelationType::get(parser.getBuilder().getContext()),
          result.operands))
    return failure();
  if (parser.parseRegion(*body, regionArgs, argTypes))
    return failure();

  return parser.addTypeToList(
      mlir::relalg::RelationType::get(parser.getBuilder().getContext()),
      result.types);
}

static void print(OpAsmPrinter &p, relalg::SelectionOp &op) {
  p << op.getOperationName();
  auto ba = op.getRegion().front().getArguments().front();
  p << " " << op.rel() << "(" << ba << ":" << ba.getType() << ")";
  p.printRegion(op.getRegion(), false, true);
}
#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.cpp.inc"
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"