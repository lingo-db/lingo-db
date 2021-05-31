#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/IR/OpImplementation.h"
#include <unordered_set>

#include <llvm/Support/Debug.h>
#include <queue>
using namespace mlir;

::mlir::ParseResult parseExtractableTimeUnit(::mlir::OpAsmParser& parser, mlir::db::ExtractableTimeUnitAttrAttr& result) {
   ::llvm::StringRef attrStr;
   ::mlir::NamedAttrList attrStorage;
   auto loc = parser.getCurrentLocation();
   if (parser.parseOptionalKeyword(&attrStr)) {
      return parser.emitError(loc, "expected extractable time unit");
   }
   if (!attrStr.empty()) {
      auto attrOptional = ::mlir::db::symbolizeExtractableTimeUnitAttr(attrStr);
      if (!attrOptional)
         return parser.emitError(loc, "invalid ")
            << "extractable time unit specification: \"" << attrStr << '"';
      result = mlir::db::ExtractableTimeUnitAttrAttr::get(parser.getBuilder().getContext(), attrOptional.getValue());
   }
   return success();
}
static void print(OpAsmPrinter& p, db::ConstantOp& op) {
   p << op.getOperationName();
   p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

   if (op->getAttrs().size() > 1)
      p << ' ';
   p << "( ";
   p.printAttributeWithoutType(op.getValue());
   p << " )";
   p << " : " << op.getType();
}

static ParseResult parseConstantOp(OpAsmParser& parser,
                                   OperationState& result) {
   Attribute valueAttr;
   if (parser.parseLParen() ||
       parser.parseAttribute(valueAttr, "value", result.attributes) ||
       parser.parseRParen())
      return failure();

   // If the attribute is a symbol reference, then we expect a trailing type.
   Type type;
   if (parser.parseColon()) {
      return failure();
   }
   if (parser.parseType(type)) {
      return failure();
   }
   // Add the attribute type to the list.
   return parser.addTypeToList(type, result.types);
}

static void buildDBCmpOp(OpBuilder& build, OperationState& result,
                         mlir::db::DBCmpPredicate predicate, Value left, Value right) {
   result.addOperands({left, right});
   bool nullable = left.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable() || right.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable();

   result.types.push_back(mlir::db::BoolType::get(build.getContext(), nullable));
   result.addAttribute("predicate",
                       build.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

static mlir::db::DBType inferResultType(OpAsmParser& parser, ArrayRef<mlir::db::DBType> types) {
   assert(types.size());
   bool anyNullables = llvm::any_of(types, [](mlir::db::DBType t) { return t.isNullable(); });
   db::DBType baseType = types[0].getBaseType();
   for (auto t : types) {
      if (t.getBaseType() != baseType) {
         parser.emitError(parser.getCurrentLocation(), "types do not have same base type");
         return mlir::db::DBType();
      }
   }
   if (anyNullables) {
      return baseType.asNullable();
   }
   return baseType;
}
static ParseResult parseImplicitResultSameOperandBaseTypeOp(OpAsmParser& parser,
                                                            OperationState& result) {
   SmallVector<OpAsmParser::OperandType, 4> args;
   SmallVector<mlir::db::DBType, 4> argTypes;

   while (true) {
      OpAsmParser::OperandType arg;
      mlir::db::DBType argType;
      auto parseRes = parser.parseOptionalOperand(arg);
      if (!parseRes.hasValue() || parseRes.getValue().failed()) {
         break;
      }
      if (parser.parseColonType(argType)) {
         return failure();
      }
      args.push_back(arg);
      argTypes.push_back(argType);
      if (!parser.parseOptionalComma()) { continue; }
      break;
   }
   if (parser.resolveOperands(args, argTypes, parser.getCurrentLocation(), result.operands).failed()) {
      return failure();
   }
   Type t = inferResultType(parser, argTypes);
   parser.addTypeToList(t, result.types);
   return success();
}

static void printImplicitResultSameOperandBaseTypeOp(Operation* op, OpAsmPrinter& p) {
   bool first = true;
   p << op->getName() << " ";
   for (auto operand : op->getOperands()) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << operand << ":" << operand.getType();
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
// CmpOp
//////////////////////////////////////////////////////////////////////////////////////////

::mlir::ParseResult parseCmpOp(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   ::mlir::IntegerAttr predicateAttr;
   ::mlir::OpAsmParser::OperandType leftOperand;
   ::llvm::SMLoc leftOperandsLoc;
   (void) leftOperandsLoc;
   ::mlir::db::DBType leftType, rightType;
   ::mlir::OpAsmParser::OperandType rightOperand;
   ::llvm::SMLoc rightOperandsLoc;
   (void) rightOperandsLoc;

   {
      ::llvm::StringRef attrStr;
      ::mlir::NamedAttrList attrStorage;
      auto loc = parser.getCurrentLocation();
      if (parser.parseOptionalKeyword(&attrStr, {"eq", "neq", "lt", "lte", "gt", "gte", "like"})) {
         return parser.emitError(loc, "expected keyword containing one of the following enum values for attribute 'predicate' [eq, neq, lt, lte, gt, gte,like]");
      }
      if (!attrStr.empty()) {
         auto attrOptional = ::mlir::db::symbolizeDBCmpPredicate(attrStr);
         if (!attrOptional)
            return parser.emitError(loc, "invalid ")
               << "predicate attribute specification: \"" << attrStr << '"';
         ;

         predicateAttr = parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue()));
         result.addAttribute("predicate", predicateAttr);
      }
   }

   leftOperandsLoc = parser.getCurrentLocation();
   if (parser.parseOperand(leftOperand))
      return ::mlir::failure();
   if (parser.parseColon())
      return ::mlir::failure();

   if (parser.parseType(leftType))
      return ::mlir::failure();
   if (parser.parseComma())
      return ::mlir::failure();

   rightOperandsLoc = parser.getCurrentLocation();
   if (parser.parseOperand(rightOperand))
      return ::mlir::failure();
   if (parser.parseColon())
      return ::mlir::failure();

   if (parser.parseType(rightType))
      return ::mlir::failure();
   if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
   if (parser.resolveOperand(leftOperand, leftType, result.operands))
      return ::mlir::failure();
   if (parser.resolveOperand(rightOperand, rightType, result.operands))
      return ::mlir::failure();
   bool nullable = rightType.isNullable() || leftType.isNullable();
   parser.addTypeToList(db::BoolType::get(parser.getBuilder().getContext(), nullable), result.types);
   return ::mlir::success();
}

static void print(::mlir::OpAsmPrinter& p, mlir::db::CmpOp& op) {
   p << "db.compare";
   p << ' ';
   {
      auto caseValue = op.predicate();
      auto caseValueStr = stringifyDBCmpPredicate(caseValue);
      p << caseValueStr;
   }
   p << ' ' << op.left() << " : " << op.left().getType() << ", " << op.right() << ' ' << ": " << op.right().getType();
   p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"predicate"});
}

static ParseResult parseDateOp(OpAsmParser& parser,
                               OperationState& result) {
   OpAsmParser::OperandType left, right;
   mlir::db::DBType leftType, rightType;
   if (parser.parseOperand(left) || parser.parseColonType(leftType) || parser.parseComma() || parser.parseOperand(right) || parser.parseColonType(rightType)) {
      return failure();
   }
   if (parser.resolveOperand(left, leftType, result.operands).failed() || parser.resolveOperand(right, rightType, result.operands).failed()) {
      return failure();
   }
   bool nullable = rightType.isNullable() || leftType.isNullable();
   parser.addTypeToList(db::DateType::get(parser.getBuilder().getContext(), nullable, leftType.dyn_cast_or_null<db::DateType>().getUnit()), result.types);
   return success();
}

static void printDateOp(Operation* op, OpAsmPrinter& p) {
   bool first = true;
   p << op->getName() << " ";
   for (auto operand : op->getOperands()) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << operand << ":" << operand.getType();
   }
}

static ParseResult parseIfOp(OpAsmParser& parser, OperationState& result) {
   // Create the regions for 'then'.
   result.regions.reserve(2);
   Region* thenRegion = result.addRegion();
   Region* elseRegion = result.addRegion();

   OpAsmParser::OperandType cond;
   Type condType;
   if (parser.parseOperand(cond) || parser.parseColonType(condType) ||
       parser.resolveOperand(cond, condType, result.operands))
      return failure();
   // Parse optional results type list.
   if (parser.parseOptionalArrowTypeList(result.types))
      return failure();
   // Parse the 'then' region.
   if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();

   // If we find an 'else' keyword then parse the 'else' region.
   if (!parser.parseOptionalKeyword("else")) {
      if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
         return failure();
   }

   // Parse the optional attribute list.
   if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
   return success();
}

static void print(OpAsmPrinter& p, mlir::db::IfOp op) {
   bool printBlockTerminators = false;

   p << mlir::db::IfOp::getOperationName() << " " << op.condition() << " : " << op.condition().getType();
   if (!op.results().empty()) {
      p << " -> (" << op.getResultTypes() << ")";
      // Print yield explicitly if the op defines values.
      printBlockTerminators = true;
   }
   p.printRegion(op.thenRegion(),
                 /*printEntryBlockArgs=*/false,
                 /*printBlockTerminators=*/printBlockTerminators);

   // Print the 'else' regions if it exists and has a block.
   auto& elseRegion = op.elseRegion();
   if (!elseRegion.empty()) {
      p << " else";
      p.printRegion(elseRegion,
                    /*printEntryBlockArgs=*/false,
                    /*printBlockTerminators=*/printBlockTerminators);
   }

   p.printOptionalAttrDict(op->getAttrs());
}

static ParseResult parseDateExtractOp(OpAsmParser& parser,
                                      OperationState& result) {
   OpAsmParser::OperandType date;
   mlir::db::DBType dateType;
   mlir::db::ExtractableTimeUnitAttrAttr unit;
   if (parseExtractableTimeUnit(parser, unit) || parser.parseComma() || parser.parseOperand(date) || parser.parseColonType(dateType)) {
      return failure();
   }
   if (parser.resolveOperand(date, dateType, result.operands).failed()) {
      return failure();
   }
   result.addAttribute("unit", unit);
   parser.addTypeToList(db::IntType::get(parser.getBuilder().getContext(), dateType.isNullable(), 64), result.types);
   return success();
}

static void print(OpAsmPrinter& p, mlir::db::DateExtractOp extractOp) {
   p << extractOp.getOperationName() << " " << mlir::db::stringifyExtractableTimeUnitAttr(extractOp.unit()) << ", " << extractOp.val() << " : " << extractOp.val().getType();
}
static void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
   assert(blocksArgs.size() == initializers.size() &&
          "expected same length of arguments and initializers");
   if (initializers.empty())
      return;

   p << prefix << '(';
   llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
     p << std::get<0>(it) << " = " << std::get<1>(it);
   });
   p << ")";
}
//adapted from scf::ForOp
static void print(OpAsmPrinter &p, mlir::db::ForOp op) {
   llvm::dbgs()<<op.region().getBlocks().size()<<"\n";
   p << op.getOperationName() << " " << op.getInductionVar() << " in "
     << op.collection() <<" : "<<op.collection().getType()<<" ";

   printInitializationList(p, op.getRegionIterArgs(), op.getIterOperands(),
                           " iter_args");
   if (!op.getIterOperands().empty())
      p << " -> (" << op.getIterOperands().getTypes() << ')';
   p.printRegion(op.region(),
      /*printEntryBlockArgs=*/false,
      /*printBlockTerminators=*/op.hasIterOperands());
   p.printOptionalAttrDict(op->getAttrs());
}
//adapted from scf::ForOp
static ParseResult parseForOp(OpAsmParser &parser, OperationState &result) {
   auto &builder = parser.getBuilder();
   OpAsmParser::OperandType inductionVariable, collection;
   Type collType;
   mlir::db::CollectionType collectionType;
   // Parse the induction variable followed by '='.
   if (parser.parseRegionArgument(inductionVariable) || parser.parseKeyword("in"))
      return failure();

   // Parse loop bounds.
   if (parser.parseOperand(collection) ||
       parser.parseColonType(collType))
      return failure();

   if(!(collectionType=collType.dyn_cast_or_null<mlir::db::CollectionType>())){
      return failure();
   }
   parser.resolveOperand(collection, collectionType, result.operands);

   // Parse the optional initial iteration arguments.
   SmallVector<OpAsmParser::OperandType, 4> regionArgs, operands;
   SmallVector<Type, 4> argTypes;
   regionArgs.push_back(inductionVariable);

   if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
      // Parse assignment list and results type list.
      if (parser.parseAssignmentList(regionArgs, operands) ||
          parser.parseArrowTypeList(result.types))
         return failure();
      // Resolve input operands.
      for (auto operandType : llvm::zip(operands, result.types))
         if (parser.resolveOperand(std::get<0>(operandType),
                                   std::get<1>(operandType), result.operands))
            return failure();
   }
   // Induction variable.
   argTypes.push_back(collectionType.getElementType());
   // Loop carried variables
   argTypes.append(result.types.begin(), result.types.end());
   // Parse the body region.
   Region *body = result.addRegion();
   if (regionArgs.size() != argTypes.size())
      return parser.emitError(
         parser.getNameLoc(),
         "mismatch in number of loop-carried values and defined values");

   if (parser.parseRegion(*body, regionArgs, argTypes))
      return failure();

   mlir::db::ForOp::ensureTerminator(*body, builder, result.location);

   // Parse the optional attribute list.
   if (parser.parseOptionalAttrDict(result.attributes))
      return failure();

   return success();
}


//taken from scf::WhileOp
static ParseResult parseWhileOp(OpAsmParser &parser, OperationState &result) {
   SmallVector<OpAsmParser::OperandType, 4> regionArgs, operands;
   Region *before = result.addRegion();
   Region *after = result.addRegion();

   OptionalParseResult listResult =
      parser.parseOptionalAssignmentList(regionArgs, operands);
   if (listResult.hasValue() && failed(listResult.getValue()))
      return failure();

   FunctionType functionType;
   llvm::SMLoc typeLoc = parser.getCurrentLocation();
   if (failed(parser.parseColonType(functionType)))
      return failure();

   result.addTypes(functionType.getResults());

   if (functionType.getNumInputs() != operands.size()) {
      return parser.emitError(typeLoc)
         << "expected as many input types as operands "
         << "(expected " << operands.size() << " got "
         << functionType.getNumInputs() << ")";
   }

   // Resolve input operands.
   if (failed(parser.resolveOperands(operands, functionType.getInputs(),
                                     parser.getCurrentLocation(),
                                     result.operands)))
      return failure();

   return failure(
      parser.parseRegion(*before, regionArgs, functionType.getInputs()) ||
      parser.parseKeyword("do") || parser.parseRegion(*after) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes));
}

//taken from scf::WhileOp
static void print(OpAsmPrinter &p, db::WhileOp op) {
   p << op.getOperationName();
   printInitializationList(p, op.before().front().getArguments(), op.inits(),
                           " ");
   p << " : ";
   p.printFunctionalType(op.inits().getTypes(), op.results().getTypes());
   p.printRegion(op.before(), /*printEntryBlockArgs=*/false);
   p << " do";
   p.printRegion(op.after());
   p.printOptionalAttrDictWithKeyword(op->getAttrs());
}


static ParseResult parseSortOp(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::OperandType toSort;
   db::VectorType vecType;
   if(parser.parseOperand(toSort)||parser.parseColonType(vecType)){
      return failure();
   }
   parser.resolveOperand(toSort,vecType,result.operands);
   parser.addTypeToList(vecType,result.types);
   OpAsmParser::OperandType left,right;
   if (parser.parseLParen()||parser.parseRegionArgument(left)||parser.parseComma()||parser.parseRegionArgument(right)||parser.parseRParen()){
      return failure();
   }
   Region* body = result.addRegion();
   if (parser.parseRegion(*body, {left,right}, {vecType.getElementType(),vecType.getElementType()})) return failure();
   return success();
}
static void print(OpAsmPrinter& p, db::SortOp& op) {
   p << op.getOperationName() << " " << op.toSort() <<":"<<op.toSort().getType() << " ";
   p << "(";
   bool first = true;
   for (auto arg : op.region().front().getArguments()) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << arg;
   }
   p << ")";
   p.printRegion(op.region(), false, true);
}
static ParseResult parseCreateAggrHTBuilder(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::OperandType toSort;
   db::AggrHTBuilderType builderType;

   OpAsmParser::OperandType left,right;
   Type leftArgType;
   Type rightArgType;
   if (parser.parseLParen()||parser.parseRegionArgument(left)||parser.parseColonType(leftArgType)||parser.parseComma()||parser.parseRegionArgument(right),parser.parseColonType(rightArgType)||parser.parseRParen()){
      return failure();
   }
   Region* body = result.addRegion();
   if (parser.parseRegion(*body, {left,right}, {leftArgType,rightArgType})) return failure();
   if(parser.parseArrow()||parser.parseType(builderType)) return failure();
   parser.addTypeToList(builderType,result.types);
   return success();
}
static void print(OpAsmPrinter& p, db::CreateAggrHTBuilder& op) {
   p << op.getOperationName() << " ";
   p << "(";
   bool first = true;
   for (auto arg : op.region().front().getArguments()) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << arg << ":" << arg.getType();
   }
   p << ")";
   p.printRegion(op.region(), false, true);
   p<< " -> "<< op.getType();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"
#include "mlir/Dialect/DB/IR/DBOpsInterfaces.cpp.inc"
