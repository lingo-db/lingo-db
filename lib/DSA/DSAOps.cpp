#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/util/UtilOps.h"

#include "mlir/IR/OpImplementation.h"
#include <unordered_set>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <queue>
using namespace mlir;
namespace {
void printInitializationList(OpAsmPrinter& p,
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
} // namespace
//adapted from scf::ForOp
void mlir::dsa::ForOp::print(OpAsmPrinter& p) {
   mlir::dsa::ForOp& op = *this;
   p << " " << op.getInductionVar() << " in "
     << op.getCollection() << " : " << op.getCollection().getType() << " ";
   printInitializationList(p, op.getRegionIterArgs(), op.getIterOperands(),
                           " iter_args");
   if (!op.getIterOperands().empty())
      p << " -> (" << op.getIterOperands().getTypes() << ')';
   p.printRegion(op.getRegion(),
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
   mlir::util::CollectionType collectionType;
   // Parse the induction variable followed by '='.
   if (parser.parseArgument(inductionVariable) || parser.parseKeyword("in"))
      return failure();

   // Parse loop bounds.
   if (parser.parseOperand(collection) ||
       parser.parseColonType(collType))
      return failure();

   if (!(collectionType = collType.dyn_cast_or_null<mlir::util::CollectionType>())) {
      return failure();
   }
   if (parser.resolveOperand(collection, collectionType, result.operands).failed()) {
      return failure();
   }

   // Parse the optional initial iteration arguments.
   SmallVector<OpAsmParser::Argument, 4> regionArgs;
   SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
   SmallVector<Type, 4> argTypes;
   regionArgs.push_back(inductionVariable);

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

   // Parse the optional attribute list.
   if (parser.parseOptionalAttrDict(result.attributes))
      return failure();

   return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/DSA/IR/DSAOps.cpp.inc"
#include "mlir/Dialect/DSA/IR/DSAOpsInterfaces.cpp.inc"
