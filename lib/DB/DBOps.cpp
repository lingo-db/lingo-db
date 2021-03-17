#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/IR/OpImplementation.h"
#include <unordered_set>

#include <queue>
using namespace mlir;
static void print(OpAsmPrinter &p, db::ConstantOp &op) {
  p << op.getOperationName();
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

  if (op->getAttrs().size() > 1)
    p << ' ';
  p << "( ";
  p.printAttributeWithoutType(op.getValue());
  p << " )";
  p << " : " << op.getType();
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
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

static void buildDBCmpOp(OpBuilder &build, OperationState &result,
                        CmpIPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(mlir::db::BoolType::get(build.getContext(),false));//TODO
  result.addAttribute(CmpIOp::getPredicateAttrName(),
                      build.getI64IntegerAttr(static_cast<int64_t>(predicate)));
}

static mlir::db::DBType inferResultType(OpAsmParser &parser,ArrayRef<mlir::db::DBType> types){
   assert(types.size());
   bool anyNullables=llvm::any_of(types,[](mlir::db::DBType t){return t.isNullable();});
   db::DBType baseType=types[0].getBaseType();
   for(auto t:types){
      if(t.getBaseType()!=baseType){
         parser.emitError(parser.getCurrentLocation(),"types do not have same base type");
         return mlir::db::DBType();
      }
   }
   if(anyNullables){
      return baseType.asNullable();
   }
   return baseType;

}
static ParseResult parseImplicitResultSameOperandBaseTypeOp(OpAsmParser &parser,
                                                  OperationState &result) {
   SmallVector<OpAsmParser::OperandType, 4> args;
   SmallVector<mlir::db::DBType, 4> argTypes;

   while (true) {
      OpAsmParser::OperandType arg;
      mlir::db::DBType argType;
      auto parse_res=parser.parseOptionalOperand(arg);
      if (!parse_res.hasValue()||parse_res.getValue().failed()) {
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
   if(parser.resolveOperands(args,argTypes,parser.getCurrentLocation(),result.operands).failed()){
      return failure();
   }
   Type t=inferResultType(parser,argTypes);
   parser.addTypeToList(t,result.types);
   return success();
}

static void printImplicitResultSameOperandBaseTypeOp(Operation *op, OpAsmPrinter &p){
   bool first=true;
   p<<op->getName()<<" ";
   for(auto operand:op->getOperands()){

      if(first){
         first=false;
      }else{
         p<<",";
      }
      p<<operand<<":"<<operand.getType();
   }

}


//////////////////////////////////////////////////////////////////////////////////////////
// CmpOp
//////////////////////////////////////////////////////////////////////////////////////////

::mlir::ParseResult parseCmpOp(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
   ::mlir::IntegerAttr predicateAttr;
   ::mlir::OpAsmParser::OperandType leftOperand;::llvm::SMLoc lhsOperandsLoc;
   (void)lhsOperandsLoc;
   ::mlir::db::DBType leftType,rightType;
   ::mlir::OpAsmParser::OperandType rightOperand;
   ::llvm::SMLoc rhsOperandsLoc;
   (void)rhsOperandsLoc;

   {
      ::llvm::StringRef attrStr;
      ::mlir::NamedAttrList attrStorage;
      auto loc = parser.getCurrentLocation();
      if (parser.parseOptionalKeyword(&attrStr, {"eq","ne","slt","sle","sgt","sge","ult","ule","ugt","uge"})) {
         ::mlir::StringAttr attrVal;
         ::mlir::OptionalParseResult parseResult =
            parser.parseOptionalAttribute(attrVal,
                                          parser.getBuilder().getNoneType(),
                                          "predicate", attrStorage);
         if (parseResult.hasValue()) {
            if (failed(*parseResult))
               return ::mlir::failure();
            attrStr = attrVal.getValue();
         } else {
            return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'predicate' [eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge]");
         }
      }
      if (!attrStr.empty()) {
         auto attrOptional = ::mlir::db::symbolizeDBCmpPredicate(attrStr);
         if (!attrOptional)
            return parser.emitError(loc, "invalid ")
               << "predicate attribute specification: \"" << attrStr << '"';;

         predicateAttr = parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue()));
         result.addAttribute("predicate", predicateAttr);
      }
   }

   lhsOperandsLoc = parser.getCurrentLocation();
   if (parser.parseOperand(leftOperand))
      return ::mlir::failure();
   if (parser.parseColon())
      return ::mlir::failure();

   if (parser.parseType(leftType))
      return ::mlir::failure();
   if (parser.parseComma())
      return ::mlir::failure();

   rhsOperandsLoc = parser.getCurrentLocation();
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
   bool nullable=rightType.isNullable()||leftType.isNullable();
   parser.addTypeToList(db::BoolType::get(parser.getBuilder().getContext(),nullable),result.types);
   return ::mlir::success();
}

static void print(::mlir::OpAsmPrinter &p,mlir::db::CmpOp& op) {
   p << "db.compare";
   p << ' ';
   {
      auto caseValue = op.predicate();
      auto caseValueStr = stringifyDBCmpPredicate(caseValue);
      p << caseValueStr;
   }
   p << ' '<< op.lhs()<< " : "<<op.lhs().getType() << ", " << op.rhs()<<' ' << ": " << op.rhs().getType();
   p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"predicate"});
}

#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"
#include "mlir/Dialect/DB/IR/DBOpsInterfaces.cpp.inc"
