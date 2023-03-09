#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/TupleStream/ColumnManager.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
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
      p << " => " << columnName.getValue();
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

ParseResult mlir::subop::CreateHeapOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   mlir::ArrayAttr sortBy;
   if (parser.parseAttribute(sortBy).failed()) {
      return failure();
   }
   result.addAttribute("sortBy", sortBy);
   mlir::subop::HeapType heapType;
   if (parser.parseArrow().failed() || parser.parseType(heapType)) {
      return failure();
   }
   std::vector<OpAsmParser::Argument> leftArgs(sortBy.size());
   std::vector<OpAsmParser::Argument> rightArgs(sortBy.size());
   if (parser.parseLParen() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < sortBy.size(); i++) {
      size_t j = 0;
      for (; j < heapType.getMembers().getNames().size(); j++) {
         if (sortBy[i] == heapType.getMembers().getNames()[j]) {
            break;
         }
      }
      leftArgs[i].type = heapType.getMembers().getTypes()[j].cast<mlir::TypeAttr>().getValue();
      if (i > 0 && parser.parseComma().failed()) return failure();
      if (parser.parseArgument(leftArgs[i])) return failure();
   }
   if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < sortBy.size(); i++) {
      size_t j = 0;
      for (; j < heapType.getMembers().getNames().size(); j++) {
         if (sortBy[i] == heapType.getMembers().getNames()[j]) {
            break;
         }
      }
      rightArgs[i].type = heapType.getMembers().getTypes()[j].cast<mlir::TypeAttr>().getValue();
      if (i > 0 && parser.parseComma().failed()) return failure();
      if (parser.parseArgument(rightArgs[i])) return failure();
   }
   if (parser.parseRSquare() || parser.parseRParen()) {
      return failure();
   }
   std::vector<OpAsmParser::Argument> args;
   args.insert(args.end(), leftArgs.begin(), leftArgs.end());
   args.insert(args.end(), rightArgs.begin(), rightArgs.end());
   Region* body = result.addRegion();
   if (parser.parseRegion(*body, args)) return failure();
   result.types.push_back(heapType);
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}

void subop::CreateHeapOp::print(OpAsmPrinter& p) {
   p << getSortBy() << " -> " << getType() << "\n";
   p << "([";
   bool first = true;
   for (size_t i = 0; i < getSortBy().size(); i++) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << getRegion().front().getArgument(i);
   }
   p << "],[";
   first = true;
   for (size_t i = 0; i < getSortBy().size(); i++) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << getRegion().front().getArgument(getSortBy().size() + i);
   }
   p << "])";
   p.printRegion(getRegion(), false, true);
   p.printOptionalAttrDict(getOperation()->getAttrs());
}
ParseResult mlir::subop::CreateSortedViewOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand getToSort;
   subop::BufferType vecType;
   if (parser.parseOperand(getToSort) || parser.parseColonType(vecType)) {
      return failure();
   }
   if (parser.resolveOperand(getToSort, vecType, result.operands).failed()) {
      return failure();
   }

   mlir::ArrayAttr sortBy;
   if (parser.parseAttribute(sortBy).failed()) {
      return failure();
   }
   result.addAttribute("sortBy", sortBy);
   std::vector<OpAsmParser::Argument> leftArgs(sortBy.size());
   std::vector<OpAsmParser::Argument> rightArgs(sortBy.size());
   if (parser.parseLParen() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < sortBy.size(); i++) {
      size_t j = 0;
      for (; j < vecType.getMembers().getNames().size(); j++) {
         if (sortBy[i] == vecType.getMembers().getNames()[j]) {
            break;
         }
      }
      leftArgs[i].type = vecType.getMembers().getTypes()[j].cast<mlir::TypeAttr>().getValue();
      if (i > 0 && parser.parseComma().failed()) return failure();
      if (parser.parseArgument(leftArgs[i])) return failure();
   }
   if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < sortBy.size(); i++) {
      size_t j = 0;
      for (; j < vecType.getMembers().getNames().size(); j++) {
         if (sortBy[i] == vecType.getMembers().getNames()[j]) {
            break;
         }
      }
      rightArgs[i].type = vecType.getMembers().getTypes()[j].cast<mlir::TypeAttr>().getValue();
      if (i > 0 && parser.parseComma().failed()) return failure();
      if (parser.parseArgument(rightArgs[i])) return failure();
   }
   if (parser.parseRSquare() || parser.parseRParen()) {
      return failure();
   }
   std::vector<OpAsmParser::Argument> args;
   args.insert(args.end(), leftArgs.begin(), leftArgs.end());
   args.insert(args.end(), rightArgs.begin(), rightArgs.end());
   Region* body = result.addRegion();
   if (parser.parseRegion(*body, args)) return failure();
   result.types.push_back(mlir::subop::SortedViewType::get(parser.getContext(), vecType));
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}

void subop::CreateSortedViewOp::print(OpAsmPrinter& p) {
   subop::CreateSortedViewOp& op = *this;
   p << " " << op.getToSort() << " : " << op.getToSort().getType() << " " << op.getSortBy() << " ";
   p << "([";
   bool first = true;
   for (size_t i = 0; i < op.getSortBy().size(); i++) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << op.getRegion().front().getArgument(i);
   }
   p << "],[";
   first = true;
   for (size_t i = 0; i < op.getSortBy().size(); i++) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << op.getRegion().front().getArgument(op.getSortBy().size() + i);
   }
   p << "])";
   p.printRegion(op.getRegion(), false, true);
   p.printOptionalAttrDict(getOperation()->getAttrs(), {"sortBy"});
}
ParseResult mlir::subop::LookupOrInsertOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   if (parser.resolveOperand(stream, mlir::tuples::TupleStreamType::get(parser.getContext()), result.operands).failed()) {
      return failure();
   }
   OpAsmParser::UnresolvedOperand state;
   if (parser.parseOperand(state)) {
      return failure();
   }

   mlir::ArrayAttr keys;
   if (parseCustRefArr(parser, keys).failed()) {
      return failure();
   }
   result.addAttribute("keys", keys);
   mlir::Type stateType;
   if (parser.parseColonType(stateType).failed()) {
      return failure();
   }
   if (parser.resolveOperand(state, stateType, result.operands).failed()) {
      return failure();
   }
   mlir::tuples::ColumnDefAttr reference;
   if (parseCustDef(parser, reference).failed()) {
      return failure();
   }
   result.addAttribute("ref", reference);
   std::vector<OpAsmParser::Argument> leftArgs(keys.size());
   std::vector<OpAsmParser::Argument> rightArgs(keys.size());
   Region* eqFn = result.addRegion();

   if (parser.parseOptionalKeyword("eq").succeeded()) {
      if (parser.parseColon() || parser.parseLParen() || parser.parseLSquare()) {
         return failure();
      }
      for (size_t i = 0; i < keys.size(); i++) {
         leftArgs[i].type = keys[i].cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
         if (i > 0 && parser.parseComma().failed()) return failure();
         if (parser.parseArgument(leftArgs[i])) return failure();
      }
      if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
         return failure();
      }
      for (size_t i = 0; i < keys.size(); i++) {
         rightArgs[i].type = keys[i].cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
         if (i > 0 && parser.parseComma().failed()) return failure();
         if (parser.parseArgument(rightArgs[i])) return failure();
      }
      if (parser.parseRSquare() || parser.parseRParen()) {
         return failure();
      }
      std::vector<OpAsmParser::Argument> args;
      args.insert(args.end(), leftArgs.begin(), leftArgs.end());
      args.insert(args.end(), rightArgs.begin(), rightArgs.end());
      if (parser.parseRegion(*eqFn, args)) return failure();
   }
   Region* initialFn = result.addRegion();

   if (parser.parseOptionalKeyword("initial").succeeded()) {
      if (parser.parseColon()) return failure();
      if (parser.parseRegion(*initialFn, {})) return failure();
   }
   result.addTypes(mlir::tuples::TupleStreamType::get(parser.getContext()));
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();

   return success();
}

void subop::LookupOrInsertOp::print(OpAsmPrinter& p) {
   subop::LookupOrInsertOp& op = *this;
   p << " " << op.getStream() << op.getState() << " ";
   printCustRefArr(p, op, op.getKeys());
   p << " : " << op.getState().getType() << " ";
   printCustDef(p, op, op.getRef());
   if (!op.getEqFn().empty()) {
      p << "eq: ([";
      bool first = true;
      for (size_t i = 0; i < op.getKeys().size(); i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.getEqFn().front().getArgument(i);
      }
      p << "],[";
      first = true;
      for (size_t i = 0; i < op.getKeys().size(); i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.getEqFn().front().getArgument(op.getKeys().size() + i);
      }
      p << "]) ";
      p.printRegion(op.getEqFn(), false, true);
   }
   if (!op.getInitFn().empty()) {
      p << "initial: ";
      p.printRegion(op.getInitFn(), false, true);
   }
   p.printOptionalAttrDict(getOperation()->getAttrs(), {"keys", "ref"});
}
ParseResult mlir::subop::InsertOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   if (parser.resolveOperand(stream, mlir::tuples::TupleStreamType::get(parser.getContext()), result.operands).failed()) {
      return failure();
   }
   OpAsmParser::UnresolvedOperand state;
   if (parser.parseOperand(state)) {
      return failure();
   }
   mlir::subop::LookupAbleState stateType;
   if (parser.parseColonType(stateType).failed()) {
      return failure();
   }
   if (parser.resolveOperand(state, stateType, result.operands).failed()) {
      return failure();
   }
   mlir::DictionaryAttr columnStateMapping;
   if(parseColumnStateMapping(parser,columnStateMapping).failed()){
      return failure();
   }
   result.addAttribute("mapping",columnStateMapping);
   auto keyTypes=stateType.getKeyMembers().getTypes();

   std::vector<OpAsmParser::Argument> leftArgs(keyTypes.size());
   std::vector<OpAsmParser::Argument> rightArgs(keyTypes.size());
   Region* eqFn = result.addRegion();

   if (parser.parseOptionalKeyword("eq").succeeded()) {
      if (parser.parseColon() || parser.parseLParen() || parser.parseLSquare()) {
         return failure();
      }
      for (size_t i = 0; i < keyTypes.size(); i++) {
         leftArgs[i].type = keyTypes[i].cast<mlir::TypeAttr>().getValue();
         if (i > 0 && parser.parseComma().failed()) return failure();
         if (parser.parseArgument(leftArgs[i])) return failure();
      }
      if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
         return failure();
      }
      for (size_t i = 0; i <  keyTypes.size(); i++) {
         rightArgs[i].type = keyTypes[i].cast<mlir::TypeAttr>().getValue();
         if (i > 0 && parser.parseComma().failed()) return failure();
         if (parser.parseArgument(rightArgs[i])) return failure();
      }
      if (parser.parseRSquare() || parser.parseRParen()) {
         return failure();
      }
      std::vector<OpAsmParser::Argument> args;
      args.insert(args.end(), leftArgs.begin(), leftArgs.end());
      args.insert(args.end(), rightArgs.begin(), rightArgs.end());
      if (parser.parseRegion(*eqFn, args)) return failure();
   }

   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();

   return success();
}
void subop::InsertOp::print(OpAsmPrinter& p) {
   subop::InsertOp& op = *this;
   p << " " << op.getStream() << op.getState() << " ";
   p << " : " << op.getState().getType() << " ";
   printColumnStateMapping(p, getOperation(),getMapping());
   auto keyTypes=getState().getType().getKeyMembers().getTypes();
   if (!op.getEqFn().empty()) {
      p << "eq: ([";
      bool first = true;
      for (size_t i = 0; i < keyTypes.size(); i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.getEqFn().front().getArgument(i);
      }
      p << "],[";
      first = true;
      for (size_t i = 0; i < keyTypes.size(); i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.getEqFn().front().getArgument(keyTypes.size() + i);
      }
      p << "]) ";
      p.printRegion(op.getEqFn(), false, true);
   }
   p.printOptionalAttrDict(getOperation()->getAttrs(), {"mapping"});
}
ParseResult mlir::subop::LookupOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   if (parser.resolveOperand(stream, mlir::tuples::TupleStreamType::get(parser.getContext()), result.operands).failed()) {
      return failure();
   }
   OpAsmParser::UnresolvedOperand state;
   if (parser.parseOperand(state)) {
      return failure();
   }

   mlir::ArrayAttr keys;
   if (parseCustRefArr(parser, keys).failed()) {
      return failure();
   }
   result.addAttribute("keys", keys);
   mlir::Type stateType;
   if (parser.parseColonType(stateType).failed()) {
      return failure();
   }
   if (parser.resolveOperand(state, stateType, result.operands).failed()) {
      return failure();
   }
   mlir::tuples::ColumnDefAttr reference;
   if (parseCustDef(parser, reference).failed()) {
      return failure();
   }
   result.addAttribute("ref", reference);
   std::vector<OpAsmParser::Argument> leftArgs(keys.size());
   std::vector<OpAsmParser::Argument> rightArgs(keys.size());
   Region* eqFn = result.addRegion();

   if (parser.parseOptionalKeyword("eq").succeeded()) {
      if (parser.parseColon() || parser.parseLParen() || parser.parseLSquare()) {
         return failure();
      }
      for (size_t i = 0; i < keys.size(); i++) {
         leftArgs[i].type = keys[i].cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
         if (i > 0 && parser.parseComma().failed()) return failure();
         if (parser.parseArgument(leftArgs[i])) return failure();
      }
      if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
         return failure();
      }
      for (size_t i = 0; i < keys.size(); i++) {
         rightArgs[i].type = keys[i].cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
         if (i > 0 && parser.parseComma().failed()) return failure();
         if (parser.parseArgument(rightArgs[i])) return failure();
      }
      if (parser.parseRSquare() || parser.parseRParen()) {
         return failure();
      }
      std::vector<OpAsmParser::Argument> args;
      args.insert(args.end(), leftArgs.begin(), leftArgs.end());
      args.insert(args.end(), rightArgs.begin(), rightArgs.end());
      if (parser.parseRegion(*eqFn, args)) return failure();
   }
   Region* initialFn = result.addRegion();

   if (parser.parseOptionalKeyword("initial").succeeded()) {
      if (parser.parseColon()) return failure();
      if (parser.parseRegion(*initialFn, {})) return failure();
   }
   result.addTypes(mlir::tuples::TupleStreamType::get(parser.getContext()));
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}

void subop::LookupOp::print(OpAsmPrinter& p) {
   subop::LookupOp& op = *this;
   p << " " << op.getStream() << op.getState() << " ";
   printCustRefArr(p, op, op.getKeys());
   p << " : " << op.getState().getType() << " ";
   printCustDef(p, op, op.getRef());
   if (!op.getEqFn().empty()) {
      p << "eq: ([";
      bool first = true;
      for (size_t i = 0; i < op.getKeys().size(); i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.getEqFn().front().getArgument(i);
      }
      p << "],[";
      first = true;
      for (size_t i = 0; i < op.getKeys().size(); i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.getEqFn().front().getArgument(op.getKeys().size() + i);
      }
      p << "]) ";
      p.printRegion(op.getEqFn(), false, true);
   }
   if (!op.getInitFn().empty()) {
      p << "initial: ";
      p.printRegion(op.getInitFn(), false, true);
   }
   p.printOptionalAttrDict(getOperation()->getAttrs(), {"keys", "ref"});
}
ParseResult mlir::subop::LoopOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   llvm::SmallVector<OpAsmParser::UnresolvedOperand> args;
   llvm::SmallVector<Type> argTypes;
   llvm::SmallVector<OpAsmParser::Argument> arguments;
   llvm::SmallVector<Type> argumentTypes;

   if (parser.parseOperandList(args) || parser.parseOptionalColonTypeList(argTypes)) {
      return failure();
   }
   if (parser.resolveOperands(args, argTypes, parser.getCurrentLocation(), result.operands).failed()) {
      return failure();
   }
   if (parser.parseLParen() || parser.parseArgumentList(arguments) || parser.parseRParen() || parser.parseOptionalArrowTypeList(argumentTypes)) {
      return failure();
   }
   if (arguments.size() != argumentTypes.size()) {
      return failure();
   }
   for (auto i = 0ul; i < arguments.size(); i++) {
      arguments[i].type = argumentTypes[i];
   }
   result.types.insert(result.types.end(), argumentTypes.begin(), argumentTypes.end());
   Region* body = result.addRegion();
   if (parser.parseRegion(*body, arguments)) return failure();
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}
void mlir::subop::LoopOp::print(::mlir::OpAsmPrinter& p) {
   if (!getArgs().empty()) {
      p << getArgs() << " : " << getArgs().getTypes();
   }
   p << " (";
   for (size_t i = 0; i < getRegion().getNumArguments(); i++) {
      if (i != 0) {
         p << " ,";
      }
      p << getRegion().getArguments()[i];
   }
   p << ")";
   if (!getResultTypes().empty()) {
      p << "-> " << getResultTypes();
   }
   p.printRegion(getRegion(), false, true);
   p.printOptionalAttrDict(getOperation()->getAttrs());
}

::mlir::ParseResult mlir::subop::CreateSegmentTreeView::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand source;
   ContinuousViewType continuousViewType;
   if (parser.parseOperand(source) || parser.parseColonType(continuousViewType) || parser.resolveOperand(source, continuousViewType, result.operands)) {
      return failure();
   }
   SegmentTreeViewType resultType;
   if (parser.parseArrow() || parser.parseType(resultType)) {
      return failure();
   }
   result.types.push_back(resultType);
   ArrayAttr relevantMembers;
   if (parser.parseKeyword("initial") || parser.parseAttribute(relevantMembers) || parser.parseColon()) {
      return failure();
   }
   result.addAttribute("relevant_members", relevantMembers);
   llvm::SmallVector<OpAsmParser::Argument> initialFnArguments;
   if (parser.parseLParen() || parser.parseArgumentList(initialFnArguments) || parser.parseRParen()) {
      return failure();
   }
   auto sourceMembers = continuousViewType.getMembers();
   for (size_t i = 0; i < relevantMembers.size(); i++) {
      size_t j = 0;
      for (; j < sourceMembers.getNames().size(); j++) {
         if (relevantMembers[i] == sourceMembers.getNames()[j]) {
            break;
         }
      }
      initialFnArguments[i].type = sourceMembers.getTypes()[j].cast<mlir::TypeAttr>().getValue();
   }
   Region* initialFn = result.addRegion();
   if (parser.parseRegion(*initialFn, initialFnArguments)) return failure();
   llvm::SmallVector<OpAsmParser::Argument> combineFnLeftArguments;
   llvm::SmallVector<OpAsmParser::Argument> combineFnRightArguments;
   if (parser.parseKeyword("combine") || parser.parseColon() || parser.parseLParen() || parser.parseLSquare()) {
      return failure();
   }
   if (parser.parseArgumentList(combineFnLeftArguments) || parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
      return failure();
   }
   if (parser.parseArgumentList(combineFnRightArguments) || parser.parseRSquare() || parser.parseRParen()) {
      return failure();
   }
   for (size_t i = 0; i < resultType.getValueMembers().getTypes().size(); i++) {
      auto t = resultType.getValueMembers().getTypes()[i].cast<mlir::TypeAttr>().getValue();
      combineFnLeftArguments[i].type = t;
      combineFnRightArguments[i].type = t;
   }
   std::vector<OpAsmParser::Argument> args;
   args.insert(args.end(), combineFnLeftArguments.begin(), combineFnLeftArguments.end());
   args.insert(args.end(), combineFnRightArguments.begin(), combineFnRightArguments.end());
   Region* combineFn = result.addRegion();
   if (parser.parseRegion(*combineFn, args)) return failure();
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}
void mlir::subop::CreateSegmentTreeView::print(::mlir::OpAsmPrinter& p) {
   p << getSource() << " : " << getSource().getType() << " -> " << getType() << " ";
   p << "initial" << getRelevantMembers() << ":"
     << "(";
   bool first = true;
   for (auto arg : getInitialFn().getArguments()) {
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      p << arg;
   }
   p << ")";
   p.printRegion(getInitialFn(), false, true);
   p << "combine: ([";
   auto argCount = getType().getValueMembers().getTypes().size();
   for (size_t i = 0; i < argCount; i++) {
      if (i > 0) {
         p << ", ";
      }
      p << getCombineFn().getArgument(i);
   }
   p << "],[";
   for (size_t i = 0; i < argCount; i++) {
      if (i > 0) {
         p << ", ";
      }
      p << getCombineFn().getArgument(i + argCount);
   }
   p << "])";
   p.printRegion(getCombineFn(), false, true);
   p.printOptionalAttrDict(getOperation()->getAttrs(), {"relevant_members"});
}
ParseResult mlir::subop::ReduceOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   if (parser.resolveOperand(stream, mlir::tuples::TupleStreamType::get(parser.getContext()), result.operands).failed()) {
      return failure();
   }
   mlir::tuples::ColumnRefAttr reference;
   if (parseCustRef(parser, reference).failed()) {
      return failure();
   }
   result.addAttribute("ref", reference);

   mlir::ArrayAttr columns;
   if (parseCustRefArr(parser, columns).failed()) {
      return failure();
   }
   result.addAttribute("columns", columns);
   mlir::ArrayAttr members;
   if (parser.parseAttribute(members).failed()) {
      return failure();
   }
   result.addAttribute("members", members);
   std::vector<OpAsmParser::Argument> leftArgs(columns.size());
   std::vector<OpAsmParser::Argument> rightArgs(members.size());
   if (parser.parseLParen() || parser.parseLSquare()) {
      return failure();
   }
   auto referenceType = reference.getColumn().type.cast<mlir::subop::StateEntryReference>();
   auto stateMembers = referenceType.getMembers();
   for (size_t i = 0; i < columns.size(); i++) {
      leftArgs[i].type = columns[i].cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
      if (i > 0 && parser.parseComma().failed()) return failure();
      if (parser.parseArgument(leftArgs[i])) return failure();
   }
   if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < members.size(); i++) {
      size_t j = 0;
      for (; j < stateMembers.getNames().size(); j++) {
         if (members[i] == stateMembers.getNames()[j]) {
            break;
         }
      }
      rightArgs[i].type = stateMembers.getTypes()[j].cast<mlir::TypeAttr>().getValue();
      if (i > 0 && parser.parseComma().failed()) return failure();
      if (parser.parseArgument(rightArgs[i])) return failure();
   }

   if (parser.parseRSquare() || parser.parseRParen()) {
      return failure();
   }
   std::vector<OpAsmParser::Argument> args;
   args.insert(args.end(), leftArgs.begin(), leftArgs.end());
   args.insert(args.end(), rightArgs.begin(), rightArgs.end());
   Region* body = result.addRegion();
   if (parser.parseRegion(*body, args)) return failure();
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}

void subop::ReduceOp::print(OpAsmPrinter& p) {
   subop::ReduceOp& op = *this;
   p << " " << op.getStream() << " ";
   printCustRef(p, op, op.getRef());
   printCustRefArr(p, op, op.getColumns());
   p << " " << op.getMembers() << " ";
   p << "([";
   bool first = true;
   for (size_t i = 0; i < op.getColumns().size(); i++) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << op.getRegion().front().getArgument(i);
   }
   p << "],[";
   first = true;
   for (size_t i = 0; i < op.getMembers().size(); i++) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << op.getRegion().front().getArgument(op.getColumns().size() + i);
   }
   p << "])";
   p.printRegion(op.getRegion(), false, true);
   p.printOptionalAttrDict(getOperation()->getAttrs(),{"columns","members","ref"});

}

ParseResult mlir::subop::NestedMapOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   if (parser.resolveOperand(stream, mlir::tuples::TupleStreamType::get(parser.getContext()), result.operands).failed()) {
      return failure();
   }

   OpAsmParser::Argument streamArg;
   std::vector<OpAsmParser::Argument> parameterArgs;
   mlir::ArrayAttr parameters;
   if (parseCustRefArr(parser, parameters).failed()) {
      return failure();
   }
   result.addAttribute("parameters", parameters);
   if (parser.parseLParen()) {
      return failure();
   }
   streamArg.type = mlir::tuples::TupleType::get(parser.getContext());
   if (parser.parseArgument(streamArg).failed()) {
      return failure();
   }
   for (auto x : parameters) {
      OpAsmParser::Argument arg;
      arg.type = x.cast<mlir::tuples::ColumnRefAttr>().getColumn().type;
      if (parser.parseComma() || parser.parseArgument(arg)) return failure();
      parameterArgs.push_back(arg);
   }
   if (parser.parseRParen()) {
      return failure();
   }
   Region* body = result.addRegion();
   std::vector<OpAsmParser::Argument> args;
   args.push_back(streamArg);
   args.insert(args.end(), parameterArgs.begin(), parameterArgs.end());
   if (parser.parseRegion(*body, args)) return failure();
   result.addTypes(mlir::tuples::TupleStreamType::get(parser.getContext()));
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}

void subop::NestedMapOp::print(OpAsmPrinter& p) {
   subop::NestedMapOp& op = *this;
   p << " " << op.getStream() << " ";
   printCustRefArr(p, this->getOperation(), getParameters());
   p << " (";
   p.printOperands(op.getRegion().front().getArguments());
   p << ") ";
   p.printRegion(op.getRegion(), false, true);
   p.printOptionalAttrDict(getOperation()->getAttrs(),{"parameters"});
}
ParseResult mlir::subop::GenerateOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   mlir::ArrayAttr createdColumns;
   if (parseCustDefArr(parser, createdColumns).failed()) {
      return failure();
   }
   result.addAttribute("generated_columns", createdColumns);

   Region* body = result.addRegion();
   if (parser.parseRegion(*body, {})) return failure();
   result.addTypes(mlir::tuples::TupleStreamType::get(parser.getContext()));
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}

void subop::GenerateOp::print(OpAsmPrinter& p) {
   subop::GenerateOp& op = *this;
   printCustDefArr(p, this->getOperation(), getGeneratedColumns());
   p.printRegion(op.getRegion(), false, true);
   p.printOptionalAttrDict(getOperation()->getAttrs(),{"generated_columns"});
}

std::vector<std::string> subop::ScanOp::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : getMapping()) {
      res.push_back(x.getName().str());
   }
   return res;
}
std::vector<std::string> subop::MaterializeOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : getMapping()) {
      res.push_back(x.getName().str());
   }
   return res;
}
std::vector<std::string> subop::NestedMapOp::getReadMembers() {
   std::vector<std::string> res;
   this->getRegion().walk([&](mlir::subop::SubOperator subop) {
      auto read = subop.getReadMembers();
      res.insert(res.end(), read.begin(), read.end());
   });
   return res;
}
std::vector<std::string> subop::NestedMapOp::getWrittenMembers() {
   std::vector<std::string> res;
   this->getRegion().walk([&](mlir::subop::SubOperator subop) {
      auto written = subop.getWrittenMembers();
      res.insert(res.end(), written.begin(), written.end());
   });
   return res;
}
std::vector<std::string> subop::LoopOp::getReadMembers() {
   std::vector<std::string> res;
   for(auto arg:getArgs()){
      if(auto stateType=mlir::dyn_cast_or_null<mlir::subop::State>(arg.getType())){
         for (auto x : stateType.getMembers().getNames()) {
            res.push_back(x.cast<mlir::StringAttr>().str());
         }
      }
   }
   this->getRegion().walk([&](mlir::subop::SubOperator subop) {
      auto read = subop.getReadMembers();
      res.insert(res.end(), read.begin(), read.end());
   });
   return res;
}
std::vector<std::string> subop::LoopOp::getWrittenMembers() {
   std::vector<std::string> res;
   this->getRegion().walk([&](mlir::subop::SubOperator subop) {
      auto written = subop.getWrittenMembers();
      res.insert(res.end(), written.begin(), written.end());
   });
   for(auto resT:getResultTypes()){
      if(auto stateType=mlir::dyn_cast_or_null<mlir::subop::State>(resT)){
         for (auto x : stateType.getMembers().getNames()) {
            res.push_back(x.cast<mlir::StringAttr>().str());
         }
      }
   }
   return res;
}
std::vector<std::string> subop::CreateArrayOp::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : getNumElements().getType().getMembers().getNames()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::CreateArrayOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : getRes().getType().getMembers().getNames()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::CreateSortedViewOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : getToSort().getType().cast<mlir::subop::BufferType>().getMembers().getNames()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::CreateHashIndexedView::getWrittenMembers() {
   return {getLinkMember().str(),getHashMember().str()};//todo: hack
}
std::vector<std::string> subop::CreateHashIndexedView::getReadMembers() {
   return {getHashMember().str()};
}
std::vector<std::string> subop::CreateSegmentTreeView::getWrittenMembers() {
   std::vector<std::string> res;
   auto names = getType().cast<mlir::subop::SegmentTreeViewType>().getValueMembers().getNames();
   for (auto name : names) {
      res.push_back(name.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::CreateSegmentTreeView::getReadMembers() {
   std::vector<std::string> res;
   for (auto name : getRelevantMembers()) {
      res.push_back(name.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::CreateContinuousView::getWrittenMembers() {
   std::vector<std::string> res;
   auto names = getSource().getType().cast<mlir::subop::State>().getMembers().getNames();
   for (auto name : names) {
      res.push_back(name.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::CreateContinuousView::getReadMembers() {
   std::vector<std::string> res;
   auto names = getSource().getType().cast<mlir::subop::State>().getMembers().getNames();
   for (auto name : names) {
      res.push_back(name.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::MaintainOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : getState().getType().cast<mlir::subop::State>().getMembers().getNames()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::MaintainOp::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : getState().getType().cast<mlir::subop::State>().getMembers().getNames()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::CreateSortedViewOp::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : getSortBy()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::ReduceOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : getMembers()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::ReduceOp::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : getMembers()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::ScatterOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : getMapping()) {
      res.push_back(x.getName().str());
   }
   return res;
}
std::vector<std::string> subop::LookupOrInsertOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : getState().getType().cast<mlir::subop::State>().getMembers().getNames()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::InsertOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : getState().getType().cast<mlir::subop::State>().getMembers().getNames()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::LookupOp::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : getState().getType().cast<mlir::subop::State>().getMembers().getNames()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::GatherOp::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : getMapping()) {
      res.push_back(x.getName().str());
   }
   return res;
}
static void replaceColumnUsesInLamda(mlir::MLIRContext* context, mlir::Block& block, const mlir::subop::ColumnFoldInfo& columnInfo) {
   auto& colManager = context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   block.walk([&columnInfo, &colManager](mlir::tuples::GetColumnOp getColumnOp) {
      auto* currColumn = &getColumnOp.getAttr().getColumn();
      if (columnInfo.directMappings.contains(currColumn)) {
         getColumnOp.setAttrAttr(colManager.createRef(columnInfo.directMappings.at(currColumn)));
      }
   });
}

mlir::LogicalResult subop::MapOp::foldColumns(mlir::subop::ColumnFoldInfo& columnInfo) {
   replaceColumnUsesInLamda(getContext(), getFn().front(), columnInfo);
   return mlir::success();
}
mlir::LogicalResult subop::FilterOp::foldColumns(mlir::subop::ColumnFoldInfo& columnInfo) {
   auto& colManager = getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   std::vector<mlir::Attribute> newConditions;
   for (auto f : getConditions()) {
      auto col = f.cast<mlir::tuples::ColumnRefAttr>();
      if (columnInfo.directMappings.contains(&col.getColumn())) {
         col = colManager.createRef(columnInfo.directMappings[&col.getColumn()]);
      }
      newConditions.push_back(col);
   }
   mlir::OpBuilder b(getContext());
   setConditionsAttr(b.getArrayAttr(newConditions));
   return mlir::success();
}
mlir::LogicalResult subop::MaterializeOp::foldColumns(mlir::subop::ColumnFoldInfo& columnInfo) {
   auto& colManager = getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

   std::vector<mlir::NamedAttribute> newMapping;
   mlir::OpBuilder b(getContext());
   for (auto m : getMapping()) {
      auto col = m.getValue().cast<mlir::tuples::ColumnRefAttr>();
      if (columnInfo.directMappings.contains(&col.getColumn())) {
         col = colManager.createRef(columnInfo.directMappings[&col.getColumn()]);
      }
      newMapping.push_back(b.getNamedAttr(m.getName().getValue(), col));
   }
   setMappingAttr(b.getDictionaryAttr(newMapping));
   return mlir::success();
}
mlir::LogicalResult subop::InsertOp::foldColumns(mlir::subop::ColumnFoldInfo& columnInfo) {
   auto& colManager = getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();

   std::vector<mlir::NamedAttribute> newMapping;
   mlir::OpBuilder b(getContext());
   for (auto m : getMapping()) {
      auto col = m.getValue().cast<mlir::tuples::ColumnRefAttr>();
      if (columnInfo.directMappings.contains(&col.getColumn())) {
         col = colManager.createRef(columnInfo.directMappings[&col.getColumn()]);
      }
      newMapping.push_back(b.getNamedAttr(m.getName().getValue(), col));
   }
   setMappingAttr(b.getDictionaryAttr(newMapping));
   return mlir::success();
}

void subop::NestedMapOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}

void subop::NestedMapOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   std::vector<Attribute> newColumns;
   for (auto col : getParameters()) {
      auto colRef = col.cast<mlir::tuples::ColumnRefAttr>();
      if (&colRef.getColumn() == oldColumn) {
         auto argumentPos = newColumns.size() + 1;
         transformer.updateValue(getRegion().getArgument(argumentPos), newColumn->type);
         getRegion().getArgument(argumentPos).setType(newColumn->type);
         newColumns.push_back(transformer.getColumnManager().createRef(newColumn));
      } else {
         newColumns.push_back(col);
      }
   }
   setParametersAttr(mlir::OpBuilder(getContext()).getArrayAttr(newColumns));
}

void subop::ScanListOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getList() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getElem().getColumn().type);
      setElemAttr(transformer.createReplacementColumn(getElemAttr(), newRefType));
   }
}
void subop::ScanListOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   assert(false && "should not happen");
}
void subop::ScanRefsOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getRef().getColumn().type);
      setRefAttr(transformer.createReplacementColumn(getRefAttr(), newRefType));
   }
}
void subop::ScanRefsOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   assert(false && "should not happen");
}

void subop::ScanOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      setMappingAttr(transformer.updateMapping(getMappingAttr()));
   }
}
void subop::ScanOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   assert(false && "should not happen");
}

void subop::GatherOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}
void subop::GatherOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   if (&getRef().getColumn() == oldColumn) {
      setRefAttr(transformer.getColumnManager().createRef(newColumn));
      setMappingAttr(transformer.updateMapping(getMappingAttr()));
   }
}
void subop::ScatterOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}
void subop::ScatterOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   if (&getRef().getColumn() == oldColumn) {
      setRefAttr(transformer.getColumnManager().createRef(newColumn));
      setMappingAttr(transformer.updateMapping(getMappingAttr()));
   }
}
void subop::LookupOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getRef().getColumn().type);
      setRefAttr(transformer.createReplacementColumn(getRefAttr(), newRefType));
   }
}
void subop::LookupOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   assert(false && "should not happen");
}
void subop::LookupOrInsertOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getRef().getColumn().type);
      setRefAttr(transformer.createReplacementColumn(getRefAttr(), newRefType));
   }
}
void subop::LookupOrInsertOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   assert(false && "should not happen");
}
void subop::InsertOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      setMappingAttr(transformer.updateMapping(getMapping()));
   }
}
void subop::InsertOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   assert(false && "should not happen");
}
void subop::ReduceOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}
void subop::ReduceOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   if (&getRef().getColumn() == oldColumn) {
      setRefAttr(transformer.getColumnManager().createRef(newColumn));
      setMembersAttr(transformer.updateMembers(getMembers()));
   }
}
void subop::UnwrapOptionalRefOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}
void subop::UnwrapOptionalRefOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   if (&getOptionalRef().getColumn() == oldColumn) {
      setOptionalRefAttr(transformer.getColumnManager().createRef(newColumn));
      auto newRefType = transformer.getNewRefType(this->getOperation(), getRef().getColumn().type);
      setRefAttr(transformer.createReplacementColumn(getRefAttr(), newRefType));
   }
}
void subop::MaterializeOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   assert(false && "should not happen");
}
void subop::MaterializeOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      setMappingAttr(transformer.updateMapping(getMapping()));
   }
}

void subop::SetTrackedCountOp::replaceColumns(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   assert(false && "should not happen");
}

void subop::SetTrackedCountOp::updateStateType(mlir::subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}

std::vector<std::string> subop::SetTrackedCountOp::getReadMembers() {
   return {getReadState().str()};
}
#define GET_OP_CLASSES
#include "mlir/Dialect/SubOperator/SubOperatorOps.cpp.inc"

#include "mlir/Dialect/SubOperator/SubOperatorOpsEnums.cpp.inc"
