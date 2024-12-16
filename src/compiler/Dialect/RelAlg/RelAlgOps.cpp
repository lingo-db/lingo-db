#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h"

#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include <llvm/ADT/TypeSwitch.h>
#include <iostream>
#include <queue>

using namespace mlir;

namespace {
using namespace lingodb::compiler::dialect;

tuples::ColumnManager& getColumnManager(::mlir::OpAsmParser& parser) {
   return parser.getBuilder().getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
}
::mlir::ParseResult parseSortSpec(::mlir::OpAsmParser& parser, relalg::SortSpec& spec) {
   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr, {"desc", "asc"})) {
      return parser.emitError(loc, "expected keyword containing one of the following enum values for attribute 'sortSpec' [desc,asc]");
   }
   if (!attrStr.empty()) {
      auto parsedSpec = ::relalg::symbolizeSortSpec(attrStr);
      if (!parsedSpec)
         return parser.emitError(loc, "invalid ")
            << "type attribute specification: \"" << attrStr << '"';
      spec = parsedSpec.value();
   }
   return success();
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
ParseResult parseSortSpecs(OpAsmParser& parser, mlir::ArrayAttr& result) {
   if (parser.parseLSquare()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      tuples::ColumnRefAttr attrRefAttr;
      if (parser.parseLParen() || parseCustRef(parser, attrRefAttr) || parser.parseComma()) {
         return failure();
      }
      relalg::SortSpec spec;
      if (parseSortSpec(parser, spec) || parser.parseRParen()) {
         return failure();
      }
      mapping.push_back(relalg::SortSpecificationAttr::get(parser.getBuilder().getContext(), attrRefAttr, spec));
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRSquare()) { return failure(); }
      break;
   }
   result = mlir::ArrayAttr::get(parser.getBuilder().getContext(), mapping);
   return success();
}
void printSortSpecs(OpAsmPrinter& p, mlir::Operation* op, ArrayAttr arrayAttr) {
   p << "[";
   std::vector<Attribute> attributes;
   bool first = true;
   for (auto a : arrayAttr) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      relalg::SortSpecificationAttr sortSpecificationAttr = mlir::dyn_cast<relalg::SortSpecificationAttr>(a);
      p << "(" << sortSpecificationAttr.getAttr().getName() << "," << relalg::stringifySortSpec(sortSpecificationAttr.getSortSpec()) << ")";
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

ParseResult parseCustAttrMapping(OpAsmParser& parser, ArrayAttr& res) {
   if (parser.parseKeyword("mapping") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::Attribute> mapping;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      tuples::ColumnDefAttr attrDefAttr;
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
void printCustAttrMapping(OpAsmPrinter& p, mlir::Operation* op, Attribute mapping) {
   p << " mapping: {";
   auto first = true;
   for (auto attr : mlir::dyn_cast_or_null<ArrayAttr>(mapping)) {
      auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      printCustDef(p, op, relationDefAttr);
   }
   p << "}";
}
} // namespace

///////////////////////////////////////////////////////////////////////////////////
// BaseTableOp
///////////////////////////////////////////////////////////////////////////////////
ParseResult relalg::BaseTableOp::parse(OpAsmParser& parser, OperationState& result) {
   if (parser.parseOptionalAttrDict(result.attributes)) return failure();
   if (parser.parseKeyword("columns") || parser.parseColon() || parser.parseLBrace()) return failure();
   std::vector<mlir::NamedAttribute> columns;
   while (true) {
      if (!parser.parseOptionalRBrace()) { break; }
      StringRef colName;
      if (parser.parseKeyword(&colName)) { return failure(); }
      if (parser.parseEqual() || parser.parseGreater()) { return failure(); }
      tuples::ColumnDefAttr attrDefAttr;
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
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(meta)) {
         result.attributes.set("meta", relalg::TableMetaDataAttr::get(parser.getContext(), runtime::TableMetaData::deserialize(strAttr.str())));
      } else {
         return failure();
      }
   } else {
      result.addAttribute("meta", relalg::TableMetaDataAttr::get(parser.getContext(), std::make_shared<runtime::TableMetaData>()));
   }
   result.addAttribute("columns", mlir::DictionaryAttr::get(parser.getBuilder().getContext(), columns));
   return parser.addTypeToList(tuples::TupleStreamType::get(parser.getBuilder().getContext()), result.types);
}
void relalg::BaseTableOp::print(OpAsmPrinter& p) {
   p << " ";
   std::vector<mlir::NamedAttribute> colsToPrint;
   for (auto attr : this->getOperation()->getAttrs()) {
      if (attr.getName().str() == "meta") {
         if (auto metaAttr = mlir::dyn_cast_or_null<relalg::TableMetaDataAttr>(attr.getValue())) {
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
   for (auto mapping : getColumns()) {
      auto columnName = mapping.getName();
      auto attr = mapping.getValue();
      auto relationDefAttr = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(attr);
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

::mlir::LogicalResult relalg::MapOp::verify() {
   if (getPredicate().empty() || getPredicate().front().empty()) {
      emitError("mapOp without body");
      return mlir::failure();
   }
   auto returnOp = mlir::cast<tuples::ReturnOp>(getPredicate().front().getTerminator());
   if (returnOp->getNumOperands() != getComputedCols().size()) {
      emitError("mapOp return vs computed cols mismatch");
      return mlir::failure();
   }
   for (auto z : llvm::zip(returnOp.getResults(), getComputedCols())) {
      if (auto colDef = mlir::dyn_cast_or_null<tuples::ColumnDefAttr>(std::get<1>(z))) {
         auto expected=std::get<0>(z).getType();
         if (colDef.getColumn().type != expected) {
            emitError("type mismatch between returned value and column definition");
            return mlir::failure();
         }
      } else {
         emitError("expected column definition for computed column");
         return mlir::failure();
      }
   }
   return mlir::success();
}

::mlir::ParseResult relalg::NestedOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> inputs;
   if (parser.parseOperandList(inputs)) {
      return mlir::failure();
   }
   auto tupleStreamType = tuples::TupleStreamType::get(parser.getContext());

   if (parser.resolveOperands(inputs, tupleStreamType, result.operands)) {
      return mlir::failure();
   }
   mlir::ArrayAttr usedCols, availableCols;
   if(parseCustRefArr(parser,usedCols).failed()||parser.parseArrow().failed()||parseCustRefArr(parser,availableCols).failed()){
      return mlir::failure();
   }
   result.addAttribute("used_cols",usedCols);
   result.addAttribute("available_cols",availableCols);
   llvm::SmallVector<mlir::OpAsmParser::Argument> regionArgs;

   if (parser.parseArgumentList(regionArgs, mlir::OpAsmParser::Delimiter::Paren)) {
      return mlir::failure();
   }
   for (auto& arg : regionArgs) {
      arg.type = tupleStreamType;
   }
   if (parser.parseRegion(*result.addRegion(), regionArgs)) return failure();
   result.addTypes(tupleStreamType);
   return mlir::success();
}

void relalg::NestedOp::print(::mlir::OpAsmPrinter& p) {
   p.printOperands(getInputs());
   printCustRefArr(p,this->getOperation(),getUsedCols());
   p<< " -> ";
   printCustRefArr(p,this->getOperation(),getAvailableCols());

   p << " (";
   p.printOperands(getNestedFn().front().getArguments());
   p << ") ";
   p.printRegion(getNestedFn(), false, true);
}
#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc"