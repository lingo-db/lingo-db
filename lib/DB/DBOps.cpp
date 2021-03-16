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
#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"
#include "mlir/Dialect/DB/IR/DBOpsInterfaces.cpp.inc"
