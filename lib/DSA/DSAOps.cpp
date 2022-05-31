#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"

#include "mlir/IR/OpImplementation.h"
#include <unordered_set>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <queue>
using namespace mlir;

static void printInitializationList(OpAsmPrinter& p,
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
void mlir::dsa::ForOp::print(OpAsmPrinter& p) {
   mlir::dsa::ForOp& op = *this;
   p << " " << op.getInductionVar() << " in "
     << op.collection() << " : " << op.collection().getType() << " ";
   if (op.until()) {
      p << "until " << op.until() << " ";
   }
   printInitializationList(p, op.getRegionIterArgs(), op.getIterOperands(),
                           " iter_args");
   if (!op.getIterOperands().empty())
      p << " -> (" << op.getIterOperands().getTypes() << ')';
   p.printRegion(op.region(),
                 /*printEntryBlockArgs=*/false,
                 /*printBlockTerminators=*/op.hasIterOperands());
   p.printOptionalAttrDict(op->getAttrs(), {"operand_segment_sizes"});
}

//adapted from scf::ForOp
ParseResult dsa::ForOp::parse(OpAsmParser& parser, OperationState& result) {
   auto& builder = parser.getBuilder();
   OpAsmParser::UnresolvedOperand collection;
   OpAsmParser::Argument inductionVariable;
   Type collType;
   mlir::dsa::CollectionType collectionType;
   // Parse the induction variable followed by '='.
   if (parser.parseArgument(inductionVariable) || parser.parseKeyword("in"))
      return failure();

   // Parse loop bounds.
   if (parser.parseOperand(collection) ||
       parser.parseColonType(collType))
      return failure();

   if (!(collectionType = collType.dyn_cast_or_null<mlir::dsa::CollectionType>())) {
      return failure();
   }
   parser.resolveOperand(collection, collectionType, result.operands);

   // Parse the optional initial iteration arguments.
   SmallVector<OpAsmParser::Argument, 4> regionArgs;
   SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
   SmallVector<Type, 4> argTypes;
   regionArgs.push_back(inductionVariable);
   bool hasUntil = false;
   if (succeeded(parser.parseOptionalKeyword("until"))) {
      OpAsmParser::UnresolvedOperand until;
      if (parser.parseOperand(until) || parser.resolveOperand(until, mlir::dsa::FlagType::get(parser.getBuilder().getContext()), result.operands)) {
         return failure();
      }
      hasUntil = true;
   }
   size_t iterArgs = 0;
   if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
      // Parse assignment list and results type list.
      if (parser.parseAssignmentList(regionArgs, operands) ||
          parser.parseArrowTypeList(result.types))
         return failure();
      // Resolve input operands.
      for (auto unresolvedOperand : llvm::zip(operands, result.types)) {
         if (parser.resolveOperand(std::get<0>(unresolvedOperand),
                                   std::get<1>(unresolvedOperand), result.operands)) {
            return failure();
         }
         iterArgs++;
      }
   }
   // Induction variable.
   argTypes.push_back(collectionType.getElementType());
   // Loop carried variables
   argTypes.append(result.types.begin(), result.types.end());
   // Parse the body region.
   Region* body = result.addRegion();
   if (regionArgs.size() != argTypes.size())
      return parser.emitError(
         parser.getNameLoc(),
         "mismatch in number of loop-carried values and defined values");

   for (auto i = 0ull; i < regionArgs.size(); i++) {
      regionArgs[i].type = argTypes[i];
   }
   if (parser.parseRegion(*body, regionArgs))
      return failure();

   mlir::dsa::ForOp::ensureTerminator(*body, builder, result.location);
   result.addAttribute("operand_segment_sizes", builder.getI32VectorAttr({1, (hasUntil ? 1 : 0), static_cast<int32_t>(iterArgs)}));

   // Parse the optional attribute list.
   if (parser.parseOptionalAttrDict(result.attributes))
      return failure();

   return success();
}

ParseResult mlir::dsa::SortOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) {
   OpAsmParser::UnresolvedOperand toSort;
   dsa::VectorType vecType;
   if (parser.parseOperand(toSort) || parser.parseColonType(vecType)) {
      return failure();
   }
   parser.resolveOperand(toSort, vecType, result.operands);
   OpAsmParser::Argument left, right;
   left.type = vecType.getElementType();
   right.type = vecType.getElementType();
   if (parser.parseLParen() || parser.parseArgument(left) || parser.parseComma() || parser.parseArgument(right) || parser.parseRParen()) {
      return failure();
   }
   Region* body = result.addRegion();
   if (parser.parseRegion(*body, {left, right})) return failure();
   return success();
}

void dsa::SortOp::print(OpAsmPrinter& p) {
   dsa::SortOp& op = *this;
   p << " " << op.toSort() << ":" << op.toSort().getType() << " ";
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

ParseResult mlir::dsa::HashtableInsert::parse(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::UnresolvedOperand ht, key, val;
   Type htType;
   Type keyType;
   Type valType;
   if (parser.parseOperand(ht) || parser.parseColonType(htType) || parser.parseComma() || parser.parseOperand(key) || parser.parseColonType(keyType)) {
      return failure();
   }
   parser.resolveOperand(ht, htType, result.operands);
   parser.resolveOperand(key, keyType, result.operands);
   if (parser.parseOptionalComma().succeeded()) {
      if (parser.parseOperand(val) || parser.parseColonType(valType)) {
         return failure();
      }
      parser.resolveOperand(val, valType, result.operands);
   }
   Region* hash = result.addRegion();
   Region* equal = result.addRegion();
   Region* reduce = result.addRegion();

   if (parser.parseOptionalKeyword("hash").succeeded()) {
      OpAsmParser::Argument left;
      Type leftArgType;
      if (parser.parseColon() || parser.parseLParen() || parser.parseArgument(left) || parser.parseColonType(leftArgType) || parser.parseRParen()) {
         return failure();
      }
      left.type = leftArgType;
      if (parser.parseRegion(*hash, {left})) return failure();
   } else {
   }
   if (parser.parseOptionalKeyword("eq").succeeded()) {
      OpAsmParser::Argument left, right;
      Type leftArgType;
      Type rightArgType;
      if (parser.parseColon() || parser.parseLParen() || parser.parseArgument(left) || parser.parseColonType(leftArgType) || parser.parseComma() || parser.parseArgument(right), parser.parseColonType(rightArgType) || parser.parseRParen()) {
         return failure();
      }
      left.type = leftArgType;
      right.type = rightArgType;
      if (parser.parseRegion(*equal, {left, right})) return failure();
   }
   if (parser.parseOptionalKeyword("reduce").succeeded()) {
      OpAsmParser::Argument left, right;
      Type leftArgType;
      Type rightArgType;
      if (parser.parseColon() || parser.parseLParen() || parser.parseArgument(left) || parser.parseColonType(leftArgType) || parser.parseComma() || parser.parseArgument(right), parser.parseColonType(rightArgType) || parser.parseRParen()) {
         return failure();
      }
      left.type = leftArgType;
      right.type = rightArgType;
      if (parser.parseRegion(*reduce, {left, right})) return failure();
   }
   return success();
}

void dsa::HashtableInsert::print(OpAsmPrinter& p) {
   dsa::HashtableInsert& op = *this;
   p << " " << op.ht() << " : " << op.ht().getType() << ", " << op.key() << " : " << op.key().getType();
   if (op.val()) {
      p << ", " << op.val() << " : " << op.val().getType();
   }
   if (!op.hash().empty()) {
      p << " hash: (";
      bool first = true;
      for (auto arg : op.hash().front().getArguments()) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << arg << ":" << arg.getType();
      }
      p << ")";
      p.printRegion(op.hash(), false, true);
   }
   if (!op.equal().empty()) {
      p << " eq: (";
      bool first = true;
      for (auto arg : op.equal().front().getArguments()) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << arg << ":" << arg.getType();
      }
      p << ")";
      p.printRegion(op.equal(), false, true);
   }
   if (!op.reduce().empty()) {
      p << " reduce: (";
      bool first = true;
      for (auto arg : op.reduce().front().getArguments()) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << arg << ":" << arg.getType();
      }
      p << ")";
      p.printRegion(op.reduce(), false, true);
   }
}

#define GET_OP_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOps.cpp.inc"
#include "mlir/Dialect/DSA/IR/DSAOpsInterfaces.cpp.inc"
