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

ParseResult mlir::subop::SortOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand toSort;
   subop::VectorType vecType;
   if (parser.parseOperand(toSort) || parser.parseColonType(vecType)) {
      return failure();
   }
   parser.resolveOperand(toSort, vecType, result.operands);

   mlir::ArrayAttr sortBy;
   parser.parseAttribute(sortBy);
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
   return success();
}

void subop::SortOp::print(OpAsmPrinter& p) {
   subop::SortOp& op = *this;
   p << " " << op.toSort() << " : " << op.toSort().getType() << " " << op.sortBy() << " ";
   p << "([";
   bool first = true;
   for (size_t i = 0; i < op.sortBy().size(); i++) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << op.region().front().getArgument(i);
   }
   p << "],[";
   first = true;
   for (size_t i = 0; i < op.sortBy().size(); i++) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << op.region().front().getArgument(op.sortBy().size() + i);
   }
   p << "])";
   p.printRegion(op.region(), false, true);
}

ParseResult mlir::subop::LookupOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   parser.resolveOperand(stream, mlir::tuples::TupleStreamType::get(parser.getContext()), result.operands);
   OpAsmParser::UnresolvedOperand state;
   if (parser.parseOperand(state)) {
      return failure();
   }

   mlir::ArrayAttr keys;
   parseCustRefArr(parser, keys);
   result.addAttribute("keys", keys);
   mlir::Type stateType;
   parser.parseColonType(stateType);
   parser.resolveOperand(state, stateType, result.operands);
   mlir::tuples::ColumnDefAttr reference;
   parseCustDef(parser, reference);
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
   return success();
}

void subop::LookupOp::print(OpAsmPrinter& p) {
   subop::LookupOp& op = *this;
   p << " " << op.stream() << op.state() << " ";
   printCustRefArr(p, op, op.keys());
   p << " : " << op.state().getType() << " ";
   printCustDef(p, op, op.ref());
   if (!op.eqFn().empty()) {
      p << "eq: ([";
      bool first = true;
      for (size_t i = 0; i < op.keys().size(); i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.eqFn().front().getArgument(i);
      }
      p << "],[";
      first = true;
      for (size_t i = 0; i < op.keys().size(); i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.eqFn().front().getArgument(op.keys().size() + i);
      }
      p << "]) ";
      p.printRegion(op.eqFn(), false, true);
   }
   if (!op.initFn().empty()) {
      p << "initial: ";
      p.printRegion(op.initFn(), false, true);
   }
}

ParseResult mlir::subop::ReduceOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   parser.resolveOperand(stream, mlir::tuples::TupleStreamType::get(parser.getContext()), result.operands);
   mlir::tuples::ColumnRefAttr reference;
   parseCustRef(parser, reference);
   result.addAttribute("ref", reference);

   mlir::ArrayAttr columns;
   parseCustRefArr(parser, columns);
   result.addAttribute("columns", columns);
   mlir::ArrayAttr members;
   parser.parseAttribute(members);
   result.addAttribute("members", members);
   std::vector<OpAsmParser::Argument> leftArgs(columns.size());
   std::vector<OpAsmParser::Argument> rightArgs(members.size());
   if (parser.parseLParen() || parser.parseLSquare()) {
      return failure();
   }
   auto referenceType = reference.getColumn().type.cast<mlir::subop::EntryRefType>();
   auto stateMembers = referenceType.getT().cast<mlir::subop::State>().getMembers();
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
   return success();
}

void subop::ReduceOp::print(OpAsmPrinter& p) {
   subop::ReduceOp& op = *this;
   p << " " << op.stream() << " ";
   printCustRef(p, op, op.ref());
   printCustRefArr(p, op, op.columns());
   p << " " << op.members() << " ";
   p << "([";
   bool first = true;
   for (size_t i = 0; i < op.columns().size(); i++) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << op.region().front().getArgument(i);
   }
   p << "],[";
   first = true;
   for (size_t i = 0; i < op.members().size(); i++) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << op.region().front().getArgument(op.columns().size() + i);
   }
   p << "])";
   p.printRegion(op.region(), false, true);
}

ParseResult mlir::subop::NestedMapOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   parser.resolveOperand(stream, mlir::tuples::TupleStreamType::get(parser.getContext()), result.operands);

   OpAsmParser::Argument streamArg;
   if (parser.parseLParen()) {
      return failure();
   }
   streamArg.type = mlir::tuples::TupleType::get(parser.getContext());
   parser.parseArgument(streamArg);
   if (parser.parseRParen()) {
      return failure();
   }
   Region* body = result.addRegion();
   if (parser.parseRegion(*body, {streamArg})) return failure();
   result.addTypes(mlir::tuples::TupleStreamType::get(parser.getContext()));
   return success();
}

void subop::NestedMapOp::print(OpAsmPrinter& p) {
   subop::NestedMapOp& op = *this;
   p << " " << op.stream() << " ";
   p << " (";
   p.printOperands(op.region().front().getArguments());
   p << ") ";
   p.printRegion(op.region(), false, true);
}

std::vector<std::string> subop::ScanOp::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : mapping()) {
      res.push_back(x.getName().str());
   }
   return res;
}
std::vector<std::string> subop::MaterializeOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : mapping()) {
      res.push_back(x.getName().str());
   }
   return res;
}
std::vector<std::string> subop::NestedMapOp::getReadMembers() {
   std::vector<std::string> res;
   this->region().walk([&](mlir::subop::SubOperator subop) {
      auto read = subop.getReadMembers();
      res.insert(res.end(), read.begin(), read.end());
   });
   return res;
}
std::vector<std::string> subop::NestedMapOp::getWrittenMembers() {
   std::vector<std::string> res;
   this->region().walk([&](mlir::subop::SubOperator subop) {
      auto written = subop.getWrittenMembers();
      res.insert(res.end(), written.begin(), written.end());
   });
   return res;
}
std::vector<std::string> subop::SortOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : toSort().getType().cast<mlir::subop::VectorType>().getMembers().getNames()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::SortOp::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : sortBy()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::ConvertToExplicit::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : state().getType().cast<mlir::subop::TableType>().getMembers().getNames()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::ReduceOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : members()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::ReduceOp::getReadMembers() {
   std::vector<std::string> res;
   for (auto x : members()) {
      res.push_back(x.cast<mlir::StringAttr>().str());
   }
   return res;
}
std::vector<std::string> subop::ScatterOp::getWrittenMembers() {
   std::vector<std::string> res;
   for (auto x : mapping()) {
      res.push_back(x.getName().str());
   }
   return res;
}
#define GET_OP_CLASSES
#include "mlir/Dialect/SubOperator/SubOperatorOps.cpp.inc"


#include "mlir/Dialect/SubOperator/SubOperatorOpsEnums.cpp.inc"
