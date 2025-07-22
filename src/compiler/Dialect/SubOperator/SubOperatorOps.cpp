#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

#include <iostream>
#include <queue>

using namespace mlir;
using namespace lingodb::compiler::dialect;
namespace {
tuples::ColumnManager& getColumnManager(::mlir::OpAsmParser& parser) {
   return parser.getBuilder().getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
}
ParseResult parseCustRef(OpAsmParser& parser, tuples::ColumnRefAttr& attr) {
   ::mlir::SymbolRefAttr parsedSymbolRefAttr;
   if (parser.parseAttribute(parsedSymbolRefAttr, parser.getBuilder().getType<::mlir::NoneType>())) { return failure(); }
   attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
   return success();
}

void printCustRef(OpAsmPrinter& p, mlir::Operation* op, tuples::ColumnRefAttr attr) {
   p << attr.getName();
}
ParseResult parseCustRefArr(OpAsmParser& parser, ArrayAttr& attr) {
   ArrayAttr parsedAttr;
   std::vector<Attribute> attributes;
   if (parser.parseAttribute(parsedAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
      return failure();
   }
   for (auto a : parsedAttr) {
      SymbolRefAttr parsedSymbolRefAttr = mlir::dyn_cast<SymbolRefAttr>(a);
      tuples::ColumnRefAttr attr = getColumnManager(parser).createRef(parsedSymbolRefAttr);
      attributes.push_back(attr);
   }
   attr = ArrayAttr::get(parser.getBuilder().getContext(), attributes);
   return success();
}

void printCustRefArr(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   std::vector<Attribute> attributes;
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      tuples::ColumnRefAttr parsedSymbolRefAttr = mlir::dyn_cast<tuples::ColumnRefAttr>(a);
      p << parsedSymbolRefAttr.getName();
   }
   p << "]";
}
ParseResult parseCustDef(OpAsmParser& parser, tuples::ColumnDefAttr& attr) {
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
   auto propType = mlir::dyn_cast<TypeAttr>(dictAttr.get("type")).getValue();
   attr.getColumn().type = propType;
   return success();
}
void printCustDef(OpAsmPrinter& p, mlir::Operation* op, tuples::ColumnDefAttr attr) {
   p << attr.getName();
   std::vector<mlir::NamedAttribute> relAttrDefProps;
   MLIRContext* context = attr.getContext();
   const tuples::Column& relationalAttribute = attr.getColumn();
   relAttrDefProps.push_back({mlir::StringAttr::get(context, "type"), mlir::TypeAttr::get(relationalAttribute.type)});
   p << "(" << mlir::DictionaryAttr::get(context, relAttrDefProps) << ")";
   Attribute fromExisting = attr.getFromExisting();
   if (fromExisting) {
      ArrayAttr fromExistingArr = mlir::dyn_cast_or_null<ArrayAttr>(fromExisting);
      p << "=";
      printCustRefArr(p, op, fromExistingArr);
   }
}

ParseResult parseStateColumnMapping(OpAsmParser& parser, subop::ColumnDefMemberMappingAttr& attr) {
   auto& memberManager = parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   if (parser.parseLBrace()) return failure();
   llvm::SmallVector<subop::DefMappingPairT> columns;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      StringRef colName;
      if (parser.parseKeyword(&colName)) { return failure(); }
      if (parser.parseEqual() || parser.parseGreater()) { return failure(); }
      tuples::ColumnDefAttr attrDefAttr;
      if (parseCustDef(parser, attrDefAttr)) {
         return failure();
      }
      columns.push_back({memberManager.lookupMember(colName.str()), attrDefAttr});
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   attr = subop::ColumnDefMemberMappingAttr::get(parser.getBuilder().getContext(), columns);
   return success();
}
ParseResult parseColumnStateMapping(OpAsmParser& parser, subop::ColumnRefMemberMappingAttr& attr) {
   auto& memberManager = parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   if (parser.parseLBrace()) return failure();
   llvm::SmallVector<subop::RefMappingPairT> columns;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      tuples::ColumnRefAttr columnRefAttr;

      if (parseCustRef(parser, columnRefAttr)) {
         return failure();
      }
      if (parser.parseEqual() || parser.parseGreater()) { return failure(); }
      StringRef colName;
      if (parser.parseKeyword(&colName)) { return failure(); }

      columns.push_back({memberManager.lookupMember(colName.str()), columnRefAttr});
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRBrace()) { return failure(); }
      break;
   }
   attr = subop::ColumnRefMemberMappingAttr::get(parser.getBuilder().getContext(), columns);
   return success();
}
void printStateColumnMapping(OpAsmPrinter& p, mlir::Operation* op, subop::ColumnDefMemberMappingAttr attr) {
   auto& memberManager = op->getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   p << "{";
   auto first = true;
   for (auto mapping : attr.getMapping()) {
      auto columnName = memberManager.getName(mapping.first);
      auto relationDefAttr = mapping.second;
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      p << columnName << " => ";
      printCustDef(p, op, relationDefAttr);
   }
   p << "}";
}
void printColumnStateMapping(OpAsmPrinter& p, mlir::Operation* op, subop::ColumnRefMemberMappingAttr attr) {
   auto& memberManager = op->getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   p << "{";
   auto first = true;
   for (auto mapping : attr.getMapping()) {
      auto columnName = memberManager.getName(mapping.first);
      auto relationRefAttr = mapping.second;
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      printCustRef(p, op, relationRefAttr);
      p << " => " << columnName;
   }
   p << "}";
}
ParseResult parseCustRegion(OpAsmParser& parser, Region& result) {
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
void printCustRegion(OpAsmPrinter& p, Operation* op, Region& r) {
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
ParseResult parseCustDefArr(OpAsmParser& parser, ArrayAttr& attr) {
   std::vector<Attribute> attributes;
   if (parser.parseLSquare()) return failure();
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      tuples::ColumnDefAttr attrDefAttr;
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
void printCustDefArr(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      tuples::ColumnDefAttr parsedSymbolRefAttr = mlir::dyn_cast<tuples::ColumnDefAttr>(a);
      printCustDef(p, op, parsedSymbolRefAttr);
   }
   p << "]";
}

ParseResult parseCustMemberAttr(OpAsmParser& parser, subop::MemberAttr& attr) {
   auto& memberManager = parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   std::string memberName;
   if (parser.parseString(&memberName).failed()) {
      return failure();
   }
   attr = subop::MemberAttr::get(parser.getContext(), memberManager.lookupMember(memberName));
   return success();
}
void printCustMemberAttr(OpAsmPrinter& p, mlir::Operation* op, subop::MemberAttr attr) {
   auto& memberManager = op->getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   p.printString(memberManager.getName(attr.getMember()));
}
ParseResult parseCustMemberArrayAttr(OpAsmParser& parser, mlir::ArrayAttr& attr) {
   // first parse the array as arrayattr of string attributes
   mlir::ArrayAttr parsedAttr;
   if (parser.parseAttribute(parsedAttr)) {
      return failure();
   }
   std::vector<Attribute> attributes;
   for (auto a : parsedAttr) {
      mlir::StringAttr strAttr = mlir::dyn_cast<mlir::StringAttr>(a);
      if (!strAttr) {
         return parser.emitError(parser.getNameLoc(), "Expected string in member array");
      }
      subop::MemberAttr memberAttr = subop::MemberAttr::get(parser.getBuilder().getContext(),
                                                            parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager().lookupMember(strAttr.getValue().str()));
      attributes.push_back(memberAttr);
   }
   attr = parser.getBuilder().getArrayAttr(attributes);
   return success();
}
void printCustMemberArrayAttr(OpAsmPrinter& p, mlir::Operation* op, mlir::ArrayAttr arrayAttr) {
   auto& memberManager = op->getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   llvm::SmallVector<mlir::Attribute> attributes;
   for (auto a : arrayAttr) {
      subop::MemberAttr memberAttr = mlir::cast<subop::MemberAttr>(a);
      attributes.push_back(mlir::StringAttr::get(op->getContext(), memberManager.getName(memberAttr.getMember())));
   }
   p << mlir::ArrayAttr::get(op->getContext(), attributes);
}
} // namespace

ParseResult subop::CreateHeapOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   auto& memberManager = parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();

   subop::HeapType heapType;
   if (parser.parseType(heapType)) {
      return failure();
   }
   mlir::ArrayAttr sortBy;
   if (parseCustMemberArrayAttr(parser, sortBy).failed()) {
      return failure();
   }
   result.addAttribute("sortBy", sortBy);
   std::vector<OpAsmParser::Argument> leftArgs(sortBy.size());
   std::vector<OpAsmParser::Argument> rightArgs(sortBy.size());
   if (parser.parseLParen() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < sortBy.size(); i++) {
      leftArgs[i].type = memberManager.getType(mlir::cast<subop::MemberAttr>(sortBy[i]).getMember());
      if (i > 0 && parser.parseComma().failed()) return failure();
      if (parser.parseArgument(leftArgs[i])) return failure();
   }
   if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < sortBy.size(); i++) {
      rightArgs[i].type = memberManager.getType(mlir::cast<subop::MemberAttr>(sortBy[i]).getMember());
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
   p << " " << getType() << " ";
   printCustMemberArrayAttr(p, getOperation(), getSortBy());
   p << " ([";
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
   p.printOptionalAttrDict(getOperation()->getAttrs(), {getSortByAttrName()});
}
ParseResult subop::CreateSortedViewOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   auto& memberManager = parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   OpAsmParser::UnresolvedOperand getToSort;
   subop::BufferType vecType;
   if (parser.parseOperand(getToSort) || parser.parseColonType(vecType)) {
      return failure();
   }
   if (parser.resolveOperand(getToSort, vecType, result.operands).failed()) {
      return failure();
   }

   mlir::ArrayAttr sortBy;
   if (parseCustMemberArrayAttr(parser, sortBy).failed()) {
      return failure();
   }
   result.addAttribute("sortBy", sortBy);
   std::vector<OpAsmParser::Argument> leftArgs(sortBy.size());
   std::vector<OpAsmParser::Argument> rightArgs(sortBy.size());
   if (parser.parseLParen() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < sortBy.size(); i++) {
      leftArgs[i].type = memberManager.getType(mlir::cast<subop::MemberAttr>(sortBy[i]).getMember());
      if (i > 0 && parser.parseComma().failed()) return failure();
      if (parser.parseArgument(leftArgs[i])) return failure();
   }
   if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < sortBy.size(); i++) {
      rightArgs[i].type = memberManager.getType(mlir::cast<subop::MemberAttr>(sortBy[i]).getMember());
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
   result.types.push_back(subop::SortedViewType::get(parser.getContext(), vecType));
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}

void subop::CreateSortedViewOp::print(OpAsmPrinter& p) {
   subop::CreateSortedViewOp& op = *this;
   p << " " << op.getToSort() << " : " << op.getToSort().getType() << " ";
   printCustMemberArrayAttr(p, getOperation(), op.getSortBy());
   p << " ([";
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
ParseResult subop::LookupOrInsertOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   if (parser.resolveOperand(stream, tuples::TupleStreamType::get(parser.getContext()), result.operands).failed()) {
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
   tuples::ColumnDefAttr reference;
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
         leftArgs[i].type = mlir::cast<tuples::ColumnRefAttr>(keys[i]).getColumn().type;
         if (i > 0 && parser.parseComma().failed()) return failure();
         if (parser.parseArgument(leftArgs[i])) return failure();
      }
      if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
         return failure();
      }
      for (size_t i = 0; i < keys.size(); i++) {
         rightArgs[i].type = mlir::cast<tuples::ColumnRefAttr>(keys[i]).getColumn().type;
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
   result.addTypes(tuples::TupleStreamType::get(parser.getContext()));
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
      p << " eq: ([";
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
ParseResult subop::InsertOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   auto& memberManager = parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   if (parser.resolveOperand(stream, tuples::TupleStreamType::get(parser.getContext()), result.operands).failed()) {
      return failure();
   }
   OpAsmParser::UnresolvedOperand state;
   if (parser.parseOperand(state)) {
      return failure();
   }
   subop::LookupAbleState stateType;
   if (parser.parseColonType(stateType).failed()) {
      return failure();
   }
   if (parser.resolveOperand(state, stateType, result.operands).failed()) {
      return failure();
   }
   subop::ColumnRefMemberMappingAttr columnStateMapping;
   if (parseColumnStateMapping(parser, columnStateMapping).failed()) {
      return failure();
   }
   result.addAttribute("mapping", columnStateMapping);
   auto keyMembers = stateType.getKeyMembers().getMembers();

   std::vector<OpAsmParser::Argument> leftArgs(keyMembers.size());
   std::vector<OpAsmParser::Argument> rightArgs(keyMembers.size());
   Region* eqFn = result.addRegion();

   if (parser.parseOptionalKeyword("eq").succeeded()) {
      if (parser.parseColon() || parser.parseLParen() || parser.parseLSquare()) {
         return failure();
      }
      for (size_t i = 0; i < keyMembers.size(); i++) {
         leftArgs[i].type = memberManager.getType(keyMembers[i]);
         if (i > 0 && parser.parseComma().failed()) return failure();
         if (parser.parseArgument(leftArgs[i])) return failure();
      }
      if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
         return failure();
      }
      for (size_t i = 0; i < keyMembers.size(); i++) {
         rightArgs[i].type = memberManager.getType(keyMembers[i]);
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
   printColumnStateMapping(p, getOperation(), getMapping());
   auto numMembers = getState().getType().getKeyMembers().getMembers().size();
   if (!op.getEqFn().empty()) {
      p << " eq: ([";
      bool first = true;
      for (size_t i = 0; i < numMembers; i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.getEqFn().front().getArgument(i);
      }
      p << "],[";
      first = true;
      for (size_t i = 0; i < numMembers; i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.getEqFn().front().getArgument(numMembers + i);
      }
      p << "]) ";
      p.printRegion(op.getEqFn(), false, true);
   }
   p.printOptionalAttrDict(getOperation()->getAttrs(), {"mapping"});
}
ParseResult subop::LookupOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   auto& memberManager = parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   if (parser.resolveOperand(stream, tuples::TupleStreamType::get(parser.getContext()), result.operands).failed()) {
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
   LookupAbleState stateType;
   if (parser.parseColonType(stateType).failed()) {
      return failure();
   }
   if (parser.resolveOperand(state, stateType, result.operands).failed()) {
      return failure();
   }
   tuples::ColumnDefAttr reference;
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
         leftArgs[i].type = memberManager.getType(stateType.getKeyMembers().getMembers()[i]);
         if (i > 0 && parser.parseComma().failed()) return failure();
         if (parser.parseArgument(leftArgs[i])) return failure();
      }
      if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
         return failure();
      }
      for (size_t i = 0; i < keys.size(); i++) {
         rightArgs[i].type = mlir::cast<tuples::ColumnRefAttr>(keys[i]).getColumn().type;
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
   result.addTypes(tuples::TupleStreamType::get(parser.getContext()));
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
ParseResult subop::LoopOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
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
void subop::LoopOp::print(::mlir::OpAsmPrinter& p) {
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

::mlir::ParseResult subop::CreateSegmentTreeView::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   auto& memberManager = parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
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
   if (parser.parseKeyword("initial") || parseCustMemberArrayAttr(parser, relevantMembers) || parser.parseColon()) {
      return failure();
   }
   result.addAttribute("relevant_members", relevantMembers);
   llvm::SmallVector<OpAsmParser::Argument> initialFnArguments;
   if (parser.parseLParen() || parser.parseArgumentList(initialFnArguments) || parser.parseRParen()) {
      return failure();
   }
   auto sourceMembers = continuousViewType.getMembers().getMembers();
   for (size_t i = 0; i < relevantMembers.size(); i++) {
      initialFnArguments[i].type = memberManager.getType(mlir::cast<subop::MemberAttr>(relevantMembers[i]).getMember());
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
   auto resMembers = resultType.getValueMembers().getMembers();
   for (size_t i = 0; i < resMembers.size(); i++) {
      auto t = memberManager.getType(resMembers[i]);
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
void subop::CreateSegmentTreeView::print(::mlir::OpAsmPrinter& p) {
   p << " " << getSource() << " : " << getSource().getType() << " -> " << getType() << " ";
   p << "initial";
   printCustMemberArrayAttr(p, getOperation(), getRelevantMembers());
   p << ":(";
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
   auto argCount = getType().getValueMembers().getMembers().size();
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
ParseResult subop::ReduceOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   auto& memberManager = parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   if (parser.resolveOperand(stream, tuples::TupleStreamType::get(parser.getContext()), result.operands).failed()) {
      return failure();
   }
   tuples::ColumnRefAttr reference;
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
   if (parseCustMemberArrayAttr(parser, members).failed()) {
      return failure();
   }
   result.addAttribute("members", members);
   std::vector<OpAsmParser::Argument> leftArgs(columns.size());
   std::vector<OpAsmParser::Argument> rightArgs(members.size());
   if (parser.parseLParen() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < columns.size(); i++) {
      leftArgs[i].type = mlir::cast<tuples::ColumnRefAttr>(columns[i]).getColumn().type;
      if (i > 0 && parser.parseComma().failed()) return failure();
      if (parser.parseArgument(leftArgs[i])) return failure();
   }
   if (parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
      return failure();
   }
   for (size_t i = 0; i < members.size(); i++) {
      rightArgs[i].type = memberManager.getType(mlir::cast<subop::MemberAttr>(members[i]).getMember());
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
   Region* combineFn = result.addRegion();
   llvm::SmallVector<OpAsmParser::Argument> combineFnLeftArguments;
   llvm::SmallVector<OpAsmParser::Argument> combineFnRightArguments;
   if (parser.parseOptionalKeyword("combine").succeeded()) {
      if (parser.parseColon() || parser.parseLParen() || parser.parseLSquare()) {
         return failure();
      }
      if (parser.parseArgumentList(combineFnLeftArguments) || parser.parseRSquare() || parser.parseComma() || parser.parseLSquare()) {
         return failure();
      }
      if (parser.parseArgumentList(combineFnRightArguments) || parser.parseRSquare() || parser.parseRParen()) {
         return failure();
      }
      for (size_t i = 0; i < members.size(); i++) {
         auto t = memberManager.getType(mlir::cast<subop::MemberAttr>(members[i]).getMember());
         combineFnLeftArguments[i].type = t;
         combineFnRightArguments[i].type = t;
      }
      std::vector<OpAsmParser::Argument> combineArgs;
      combineArgs.insert(combineArgs.end(), combineFnLeftArguments.begin(), combineFnLeftArguments.end());
      combineArgs.insert(combineArgs.end(), combineFnRightArguments.begin(), combineFnRightArguments.end());
      if (parser.parseRegion(*combineFn, combineArgs)) return failure();
   }
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}

void subop::ReduceOp::print(OpAsmPrinter& p) {
   subop::ReduceOp& op = *this;
   p << " " << op.getStream() << " ";
   printCustRef(p, op, op.getRef());
   printCustRefArr(p, op, op.getColumns());
   p << " ";
   printCustMemberArrayAttr(p, getOperation(), op.getMembers());
   p << " ";
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
   if (!op.getCombine().empty()) {
      p << "combine: ([";
      bool first = true;
      for (size_t i = 0; i < op.getMembers().size(); i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.getCombine().front().getArgument(i);
      }
      p << "],[";
      first = true;
      for (size_t i = 0; i < op.getMembers().size(); i++) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << op.getCombine().front().getArgument(op.getMembers().size() + i);
      }
      p << "])";
      p.printRegion(op.getCombine(), false, true);
   }
   p.printOptionalAttrDict(getOperation()->getAttrs(), {"columns", "members", "ref"});
}

ParseResult subop::NestedMapOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand stream;
   if (parser.parseOperand(stream)) {
      return failure();
   }
   if (parser.resolveOperand(stream, tuples::TupleStreamType::get(parser.getContext()), result.operands).failed()) {
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
   streamArg.type = tuples::TupleType::get(parser.getContext());
   if (parser.parseArgument(streamArg).failed()) {
      return failure();
   }
   for (auto x : parameters) {
      OpAsmParser::Argument arg;
      arg.type = mlir::cast<tuples::ColumnRefAttr>(x).getColumn().type;
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
   result.addTypes(tuples::TupleStreamType::get(parser.getContext()));
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
   p.printOptionalAttrDict(getOperation()->getAttrs(), {"parameters"});
}
ParseResult subop::GenerateOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   mlir::ArrayAttr createdColumns;
   if (parseCustDefArr(parser, createdColumns).failed()) {
      return failure();
   }
   result.addAttribute("generated_columns", createdColumns);

   Region* body = result.addRegion();
   if (parser.parseRegion(*body, {})) return failure();
   //todo: count the number of emitters and add that many arguments
   result.addTypes(tuples::TupleStreamType::get(parser.getContext()));
   body->walk([&](subop::GenerateEmitOp) {
      result.addTypes(tuples::TupleStreamType::get(parser.getContext()));
   });
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   return success();
}

void subop::GenerateOp::print(OpAsmPrinter& p) {
   subop::GenerateOp& op = *this;
   printCustDefArr(p, this->getOperation(), getGeneratedColumns());
   p.printRegion(op.getRegion(), false, true);
   p.printOptionalAttrDict(getOperation()->getAttrs(), {"generated_columns"});
}

llvm::SmallVector<subop::Member> subop::ScanOp::getReadMembers() {
   return getMapping().getMembers();
}
llvm::SmallVector<subop::Member> subop::MaterializeOp::getWrittenMembers() {
   return getMapping().getMembers();
}
llvm::SmallVector<subop::Member> subop::LockOp::getReadMembers() {
   llvm::SmallVector<subop::Member> res;
   this->getNested().walk([&](subop::SubOperator subop) {
      auto read = subop.getReadMembers();
      res.insert(res.end(), read.begin(), read.end());
   });
   return res;
}
llvm::SmallVector<subop::Member> subop::LockOp::getWrittenMembers() {
   llvm::SmallVector<subop::Member> res;
   this->getNested().walk([&](subop::SubOperator subop) {
      auto written = subop.getWrittenMembers();
      res.insert(res.end(), written.begin(), written.end());
   });
   return res;
}
namespace {
void cloneRegionInto(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping, mlir::Region& region, mlir::Region& newRegion) {
   newRegion.getBlocks().clear();
   for (auto& block : region) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto* newBlock = builder.createBlock(&newRegion);
      for (auto arg : block.getArguments()) {
         auto newArg = newBlock->addArgument(arg.getType(), arg.getLoc());
         mapping.map(arg, newArg);
      }
      builder.setInsertionPointToStart(newBlock);
      for (auto& op : block) {
         if (auto subop = mlir::dyn_cast<subop::SubOperator>(op)) {
            subop.cloneSubOp(builder, mapping, columnMapping);

         } else {
            builder.clone(op, mapping);
         }
      }
   }
}

void mapResults(mlir::IRMapping& mapping, mlir::Operation* from, mlir::Operation* to) {
   for (auto i = 0ul; i < from->getNumResults(); i++) {
      mapping.map(from->getResult(i), to->getResult(i));
   }
}
} // namespace
mlir::Operation* subop::LockOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<LockOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), columnMapping.remap(getRef()));
   cloneRegionInto(builder, mapping, columnMapping, getNested(), newOp.getNested());
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
llvm::SmallVector<subop::Member> subop::NestedMapOp::getReadMembers() {
   llvm::SmallVector<subop::Member> res;
   getRegion().walk([&](subop::SubOperator subop) {
      auto read = subop.getReadMembers();
      res.insert(res.end(), read.begin(), read.end());
   });
   return res;
}
llvm::SmallVector<subop::Member> subop::NestedMapOp::getWrittenMembers() {
   llvm::DenseSet<Member> res;
   this->getRegion().walk([&](subop::SubOperator subop) {
      auto written = subop.getWrittenMembers();
      res.insert(written.begin(), written.end());
   });
   this->getRegion().walk([&](subop::StateCreator creator) {
      auto created = creator.getCreatedMembers();
      for (auto m : created) {
         res.erase(m);
      }
   });
   return llvm::SmallVector<Member>(res.begin(), res.end());
}
mlir::Operation* subop::NestedMapOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newMap = builder.create<NestedMapOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), columnMapping.remap(getParameters()));
   cloneRegionInto(builder, mapping, columnMapping, getRegion(), newMap.getRegion());
   mapResults(mapping, this->getOperation(), newMap.getOperation());
   return newMap;
}
llvm::SmallVector<subop::Member> subop::LoopOp::getReadMembers() {
   llvm::SmallVector<subop::Member> res;
   for (auto arg : getArgs()) {
      if (auto stateType = mlir::dyn_cast_or_null<subop::State>(arg.getType())) {
         auto members = stateType.getMembers().getMembers();
         res.insert(res.end(), members.begin(), members.end());
      }
   }
   this->getRegion().walk([&](subop::SubOperator subop) {
      auto read = subop.getReadMembers();
      res.insert(res.end(), read.begin(), read.end());
   });
   return res;
}
llvm::SmallVector<subop::Member> subop::LoopOp::getWrittenMembers() {
   llvm::SmallVector<subop::Member> res;
   this->getRegion().walk([&](subop::SubOperator subop) {
      auto written = subop.getWrittenMembers();
      res.insert(res.end(), written.begin(), written.end());
   });
   for (auto resT : getResultTypes()) {
      if (auto stateType = mlir::dyn_cast_or_null<subop::State>(resT)) {
         auto members = stateType.getMembers().getMembers();
         res.insert(res.end(), members.begin(), members.end());
      }
   }
   return res;
}
llvm::SmallVector<subop::Member> subop::CreateArrayOp::getReadMembers() {
   return getNumElements().getType().getMembers().getMembers();
}
llvm::SmallVector<subop::Member> subop::CreateArrayOp::getWrittenMembers() {
   return getRes().getType().getMembers().getMembers();
}
llvm::SmallVector<subop::Member> subop::CreateSortedViewOp::getWrittenMembers() {
   return mlir::cast<subop::BufferType>(getToSort().getType()).getMembers().getMembers();
}
llvm::SmallVector<subop::Member> subop::CreateHashIndexedView::getWrittenMembers() {
   return llvm::SmallVector<Member>{getLinkMember().getMember(), getHashMember().getMember()}; //todo: hack
}
llvm::SmallVector<subop::Member> subop::CreateHashIndexedView::getReadMembers() {
   return llvm::SmallVector<Member>{getHashMember().getMember()};
}
llvm::SmallVector<subop::Member> subop::MergeOp::getReadMembers() {
   return getThreadLocal().getType().getWrapped().getMembers().getMembers();
}
llvm::SmallVector<subop::Member> subop::MergeOp::getWrittenMembers() {
   return getThreadLocal().getType().getWrapped().getMembers().getMembers();
}
llvm::SmallVector<subop::Member> subop::CreateSegmentTreeView::getWrittenMembers() {
   return mlir::cast<subop::SegmentTreeViewType>(getType()).getValueMembers().getMembers();
}
llvm::SmallVector<subop::Member> subop::CreateSegmentTreeView::getReadMembers() {
   llvm::SmallVector<Member> res;
   for (auto name : getRelevantMembers()) {
      res.push_back(mlir::cast<MemberAttr>(name).getMember());
   }
   return res;
}
llvm::SmallVector<subop::Member> subop::CreateContinuousView::getWrittenMembers() {
   return mlir::cast<State>(getSource().getType()).getMembers().getMembers();
}
llvm::SmallVector<subop::Member> subop::CreateContinuousView::getReadMembers() {
   return mlir::cast<subop::State>(getSource().getType()).getMembers().getMembers();
}

llvm::SmallVector<subop::Member> subop::SimpleStateGetScalar::getReadMembers() {
   return {getMember().getMember()};
}
llvm::SmallVector<subop::Member> subop::CreateSortedViewOp::getReadMembers() {
   llvm::SmallVector<Member> res;
   for (auto x : getSortBy()) {
      res.push_back(mlir::cast<MemberAttr>(x).getMember());
   }
   return res;
}
llvm::SmallVector<subop::Member> subop::ReduceOp::getWrittenMembers() {
   llvm::SmallVector<Member> res;
   for (auto x : getMembers()) {
      res.push_back(mlir::cast<MemberAttr>(x).getMember());
   }
   return res;
}
llvm::SmallVector<subop::Member> subop::ReduceOp::getReadMembers() {
   llvm::SmallVector<Member> res;
   for (auto x : getMembers()) {
      res.push_back(mlir::cast<MemberAttr>(x).getMember());
   }
   return res;
}

mlir::Operation* subop::ReduceOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<ReduceOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), columnMapping.remap(getRef()), columnMapping.remap(getColumns()), getMembers());
   builder.cloneRegionBefore(getCombine(), newOp.getCombine(), newOp.getCombine().begin(), mapping);
   builder.cloneRegionBefore(getRegion(), newOp.getRegion(), newOp.getRegion().begin(), mapping);
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
llvm::SmallVector<subop::Member> subop::ScatterOp::getWrittenMembers() {
   return getMapping().getMembers();
}
mlir::Operation* subop::ScatterOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<ScatterOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), columnMapping.remap(getRef()), columnMapping.remap(getMapping()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
llvm::SmallVector<subop::Member> subop::LookupOrInsertOp::getWrittenMembers() {
   return mlir::cast<subop::State>(getState().getType()).getMembers().getMembers();
}
mlir::Operation* subop::LookupOrInsertOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<LookupOrInsertOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), mapping.lookupOrDefault(getState()), columnMapping.remap(getKeys()), columnMapping.clone(getRef()));
   builder.cloneRegionBefore(getInitFn(), newOp.getInitFn(), newOp.getInitFn().begin(), mapping);
   builder.cloneRegionBefore(getEqFn(), newOp.getEqFn(), newOp.getEqFn().begin(), mapping);
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
llvm::SmallVector<subop::Member> subop::InsertOp::getWrittenMembers() {
   return mlir::cast<subop::State>(getState().getType()).getMembers().getMembers();
}

mlir::Operation* subop::InsertOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<InsertOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), mapping.lookupOrDefault(getState()), columnMapping.remap(getMapping()));
   builder.cloneRegionBefore(getEqFn(), newOp.getEqFn(), newOp.getEqFn().begin(), mapping);
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
llvm::SmallVector<subop::Member> subop::LookupOp::getReadMembers() {
   if (auto lookableState = mlir::dyn_cast_or_null<subop::LookupAbleState>(getState().getType())) {
      return lookableState.getKeyMembers().getMembers();
   }
   return {};
}
mlir::Operation* subop::LookupOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<LookupOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), mapping.lookupOrDefault(getState()), columnMapping.remap(getKeys()), columnMapping.clone(getRef()));
   builder.cloneRegionBefore(getEqFn(), newOp.getEqFn(), newOp.getEqFn().begin(), mapping);
   builder.cloneRegionBefore(getInitFn(), newOp.getInitFn(), newOp.getInitFn().begin(), mapping);
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
llvm::SmallVector<subop::Member> subop::GatherOp::getReadMembers() {
   return getMapping().getMembers();
}

mlir::Operation* subop::GatherOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<GatherOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), columnMapping.remap(getRef()), columnMapping.clone(getMapping()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());

   return newOp;
}

mlir::LogicalResult subop::MapOp::foldColumns(subop::ColumnMapping& columnInfo) {
   setInputColsAttr(columnInfo.remap(getInputCols()));
   return mlir::success();
}

mlir::Operation* subop::MapOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<MapOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), columnMapping.clone(getComputedCols()), columnMapping.remap(getInputCols()));
   builder.cloneRegionBefore(getFn(), newOp.getFn(), newOp.getFn().begin(), mapping);
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
mlir::LogicalResult subop::FilterOp::foldColumns(subop::ColumnMapping& columnInfo) {
   setConditionsAttr(columnInfo.remap(getConditions()));
   return mlir::success();
}
mlir::Operation* subop::FilterOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<FilterOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), getFilterSemantic(), columnMapping.remap(getConditions()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
mlir::LogicalResult subop::MaterializeOp::foldColumns(subop::ColumnMapping& columnInfo) {
   setMappingAttr(columnInfo.remap(getMapping()));
   return mlir::success();
}
mlir::Operation* subop::MaterializeOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<MaterializeOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), mapping.lookupOrDefault(getState()), columnMapping.remap(getMapping()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
mlir::LogicalResult subop::InsertOp::foldColumns(subop::ColumnMapping& columnInfo) {
   setMappingAttr(columnInfo.remap(getMapping()));
   return mlir::success();
}

void subop::NestedMapOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}

void subop::NestedMapOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   std::vector<Attribute> newColumns;
   for (auto col : getParameters()) {
      auto colRef = mlir::cast<tuples::ColumnRefAttr>(col);
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

void subop::ScanListOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getList() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getElem().getColumn().type);
      setElemAttr(transformer.createReplacementColumn(getElemAttr(), newRefType));
   }
}
void subop::ScanListOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}
mlir::Operation* subop::ScanListOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<ScanListOp>(this->getLoc(), mapping.lookupOrDefault(getList()), columnMapping.clone(getElem()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());

   return newOp;
}
void subop::ScanRefsOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getRef().getColumn().type);
      setRefAttr(transformer.createReplacementColumn(getRefAttr(), newRefType));
   }
}
mlir::Operation* subop::ScanRefsOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<ScanRefsOp>(this->getLoc(), mapping.lookupOrDefault(getState()), columnMapping.clone(getRef()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());

   return newOp;
}

void subop::ScanRefsOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}

void subop::ScanOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      setMappingAttr(transformer.updateMapping(getMappingAttr()));
   }
}
void subop::ScanOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}

mlir::Operation* subop::ScanOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<ScanOp>(this->getLoc(), mapping.lookupOrDefault(getState()), columnMapping.clone(getMapping()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());

   return newOp;
}

void subop::GatherOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}
void subop::GatherOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   if (&getRef().getColumn() == oldColumn) {
      setRefAttr(transformer.getColumnManager().createRef(newColumn));
      setMappingAttr(transformer.updateMapping(getMappingAttr()));
   }
}
void subop::ScatterOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}
void subop::ScatterOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   if (&getRef().getColumn() == oldColumn) {
      setRefAttr(transformer.getColumnManager().createRef(newColumn));
      setMappingAttr(transformer.updateMapping(getMappingAttr()));
   }
}
void subop::LookupOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getRef().getColumn().type);
      setRefAttr(transformer.createReplacementColumn(getRefAttr(), newRefType));
   }
}
void subop::LookupOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}
void subop::LookupOrInsertOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      auto newRefType = transformer.getNewRefType(this->getOperation(), getRef().getColumn().type);
      setRefAttr(transformer.createReplacementColumn(getRefAttr(), newRefType));
   }
}
void subop::LookupOrInsertOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}
void subop::InsertOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      setMappingAttr(transformer.updateMapping(getMapping()));
   }
}
void subop::InsertOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}
void subop::ReduceOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}
void subop::ReduceOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   if (&getRef().getColumn() == oldColumn) {
      setRefAttr(transformer.getColumnManager().createRef(newColumn));
      setMembersAttr(transformer.updateMembers(getMembers()));
   }
}
void subop::UnwrapOptionalRefOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}
void subop::UnwrapOptionalRefOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   if (&getOptionalRef().getColumn() == oldColumn) {
      setOptionalRefAttr(transformer.getColumnManager().createRef(newColumn));
      auto newRefType = transformer.getNewRefType(this->getOperation(), getRef().getColumn().type);
      setRefAttr(transformer.createReplacementColumn(getRefAttr(), newRefType));
   }
}
mlir::Operation* subop::UnwrapOptionalRefOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<UnwrapOptionalRefOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), columnMapping.remap(getOptionalRef()), columnMapping.clone(getRef()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
void subop::LockOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}
void subop::LockOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   if (&getRef().getColumn() == oldColumn) {
      setRefAttr(transformer.getColumnManager().createRef(newColumn));
   }
}
void subop::MaterializeOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}
void subop::MaterializeOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   if (state == getState() && newType != state.getType()) {
      setMappingAttr(transformer.updateMapping(getMapping()));
   }
}

void subop::ExecutionStepOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}

void subop::ExecutionStepOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   for (auto [i, a, isThreadLocal] : llvm::zip(getInputs(), getSubOps().getArguments(), getIsThreadLocal())) {
      if (i == state) {
         if (mlir::cast<mlir::BoolAttr>(isThreadLocal).getValue()) {
            auto localType = mlir::cast<subop::ThreadLocalType>(newType).getWrapped();
            transformer.updateValue(a, localType);
            a.setType(localType);
         } else {
            transformer.updateValue(a, newType);
            a.setType(newType);
         }
      }
   }
}

void subop::SetTrackedCountOp::replaceColumns(subop::SubOpStateUsageTransformer& transformer, tuples::Column* oldColumn, tuples::Column* newColumn) {
   assert(false && "should not happen");
}

void subop::SetTrackedCountOp::updateStateType(subop::SubOpStateUsageTransformer& transformer, mlir::Value state, mlir::Type newType) {
   assert(false && "should not happen");
}

llvm::SmallVector<subop::Member> subop::SetTrackedCountOp::getReadMembers() {
   return {getReadState().getMember()};
}
llvm::SmallVector<subop::Member> subop::GenericCreateOp::getCreatedMembers() {
   if (auto stateType = mlir::cast<subop::State>(getRes().getType())) {
      return stateType.getMembers().getMembers();
   }
   return {};
}

llvm::SmallVector<subop::Member> subop::CreateFrom::getReadMembers() {
   if (auto stateType = mlir::cast<subop::State>(getState().getType())) {
      return stateType.getMembers().getMembers();
   }
   return {};
}
llvm::SmallVector<subop::Member> subop::CreateFrom::getWrittenMembers() {
   if (auto stateType = mlir::cast<subop::State>(getState().getType())) {
      return stateType.getMembers().getMembers();
   }
   return {};
}

mlir::Operation* subop::RenamingOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<RenamingOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), columnMapping.clone(getColumns()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());

   return newOp;
}
mlir::Operation* subop::EntriesBetweenOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<EntriesBetweenOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), columnMapping.remap(getLeftRef()), columnMapping.remap(getRightRef()), columnMapping.clone(getBetween()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());

   return newOp;
}
mlir::Operation* subop::GetBeginReferenceOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<GetBeginReferenceOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), mapping.lookupOrDefault(getState()), columnMapping.clone(getRef()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());

   return newOp;
}
mlir::Operation* subop::GetEndReferenceOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<GetEndReferenceOp>(this->getLoc(), mapping.lookupOrDefault(getStream()), mapping.lookupOrDefault(getState()), columnMapping.clone(getRef()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
mlir::Operation* subop::OffsetReferenceBy::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<OffsetReferenceBy>(this->getLoc(), mapping.lookupOrDefault(getStream()), columnMapping.remap(getRef()), columnMapping.remap(getIdx()), columnMapping.clone(getNewRef()));
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
mlir::Operation* subop::GenerateOp::cloneSubOp(mlir::OpBuilder& builder, mlir::IRMapping& mapping, subop::ColumnMapping& columnMapping) {
   auto newOp = builder.create<GenerateOp>(this->getLoc(), getResultTypes(), columnMapping.clone(getGeneratedColumns()));
   builder.cloneRegionBefore(getRegion(), newOp.getRegion(), newOp.getRegion().begin(), mapping);
   mapResults(mapping, this->getOperation(), newOp.getOperation());
   return newOp;
}
#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.cpp.inc"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsEnums.cpp.inc"
