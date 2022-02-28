#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/IR/OpImplementation.h"
#include <unordered_set>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Debug.h>
#include <queue>
using namespace mlir;

mlir::Type constructNullableBool(MLIRContext* context, ValueRange operands) {
   bool nullable = llvm::any_of(operands, [](auto operand) { return operand.getType().template isa<mlir::db::NullableType>(); });
   mlir::Type restype = IntegerType::get(context, 1);
   if (nullable) {
      restype = mlir::db::NullableType::get(context, restype);
   }
   return restype;
}
mlir::Type getBaseType(mlir::Type t) {
   if (auto nullableT = t.dyn_cast_or_null<mlir::db::NullableType>()) {
      return nullableT.getType();
   }
   return t;
}
bool isIntegerType(mlir::Type type, unsigned int width) {
   auto asStdInt = type.dyn_cast_or_null<mlir::IntegerType>();
   return asStdInt && asStdInt.getWidth() == width;
}
int getIntegerWidth(mlir::Type type, bool isUnSigned) {
   auto asStdInt = type.dyn_cast_or_null<mlir::IntegerType>();
   if (asStdInt && asStdInt.isUnsigned() == isUnSigned) {
      return asStdInt.getWidth();
   }
   return 0;
}
LogicalResult mlir::db::CmpOp::inferReturnTypes(
   MLIRContext* context, Optional<Location> location, ValueRange operands,
   DictionaryAttr attributes, RegionRange regions,
   SmallVectorImpl<Type>& inferredReturnTypes) {
   inferredReturnTypes.assign({constructNullableBool(context, operands)});
   return success();
}
static void buildDBAndOp(OpBuilder& builder, OperationState& result, ValueRange operands) {
   result.addOperands(operands);
   result.types.push_back(constructNullableBool(builder.getContext(), operands));
}
static void buildDBOrOp(OpBuilder& builder, OperationState& result, ValueRange operands) {
   result.addOperands(operands);
   result.types.push_back(constructNullableBool(builder.getContext(), operands));
}

static Type inferResultType(OpAsmParser& parser, ArrayRef<Type> types) {
   assert(types.size());
   bool anyNullables = llvm::any_of(types, [](Type t) { return t.isa<mlir::db::NullableType>(); });
   auto firstNullable = types[0].dyn_cast_or_null<mlir::db::NullableType>();
   Type baseType = firstNullable ? firstNullable.getType() : types[0];
   for (auto t : types) {
      auto asNullable = t.dyn_cast_or_null<mlir::db::NullableType>();
      Type currBaseType = asNullable ? asNullable.getType() : t;
      if (baseType != currBaseType) {
         parser.emitError(parser.getCurrentLocation(), "types do not have same base type");
         return Type();
      }
   }
   if (anyNullables) {
      return mlir::db::NullableType::get(parser.getContext(), baseType);
   }
   return baseType;
}
static ParseResult parseImplicitResultSameOperandBaseTypeOp(OpAsmParser& parser,
                                                            OperationState& result) {
   SmallVector<OpAsmParser::OperandType, 4> args;
   SmallVector<mlir::Type, 4> argTypes;

   while (true) {
      OpAsmParser::OperandType arg;
      mlir::Type argType;
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
   p << " ";
   for (auto operand : op->getOperands()) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << operand << ":" << operand.getType();
   }
}

static ParseResult parseDateOp(OpAsmParser& parser,
                               OperationState& result) {
   OpAsmParser::OperandType left, right;
   mlir::Type leftType, rightType;
   if (parser.parseOperand(left) || parser.parseColonType(leftType) || parser.parseComma() || parser.parseOperand(right) || parser.parseColonType(rightType)) {
      return failure();
   }
   if (parser.resolveOperand(left, leftType, result.operands).failed() || parser.resolveOperand(right, rightType, result.operands).failed()) {
      return failure();
   }
   bool nullable = rightType.isa<mlir::db::NullableType>() || leftType.isa<mlir::db::NullableType>();
   mlir::Type resType = db::DateType::get(parser.getBuilder().getContext(), leftType.dyn_cast_or_null<db::DateType>().getUnit());
   if (nullable) resType = mlir::db::NullableType::get(parser.getContext(), resType);
   parser.addTypeToList(resType, result.types);
   return success();
}

static void printDateOp(Operation* op, OpAsmPrinter& p) {
   bool first = true;
   p << " ";
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

   p << " " << op.condition() << " : " << op.condition().getType();
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
static void print(OpAsmPrinter& p, mlir::db::ForOp op) {
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
static ParseResult parseForOp(OpAsmParser& parser, OperationState& result) {
   auto& builder = parser.getBuilder();
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

   if (!(collectionType = collType.dyn_cast_or_null<mlir::db::CollectionType>())) {
      return failure();
   }
   parser.resolveOperand(collection, collectionType, result.operands);

   // Parse the optional initial iteration arguments.
   SmallVector<OpAsmParser::OperandType, 4> regionArgs, operands;
   SmallVector<Type, 4> argTypes;
   regionArgs.push_back(inductionVariable);
   bool hasUntil = false;
   if (succeeded(parser.parseOptionalKeyword("until"))) {
      OpAsmParser::OperandType until;
      if (parser.parseOperand(until) || parser.resolveOperand(until, mlir::db::FlagType::get(parser.getBuilder().getContext()), result.operands)) {
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
      for (auto operandType : llvm::zip(operands, result.types)) {
         if (parser.resolveOperand(std::get<0>(operandType),
                                   std::get<1>(operandType), result.operands)) {
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

   if (parser.parseRegion(*body, regionArgs, argTypes))
      return failure();

   mlir::db::ForOp::ensureTerminator(*body, builder, result.location);
   result.addAttribute("operand_segment_sizes", builder.getI32VectorAttr({1, (hasUntil ? 1 : 0), static_cast<int32_t>(iterArgs)}));

   // Parse the optional attribute list.
   if (parser.parseOptionalAttrDict(result.attributes))
      return failure();

   return success();
}


static ParseResult parseSortOp(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::OperandType toSort;
   db::VectorType vecType;
   if (parser.parseOperand(toSort) || parser.parseColonType(vecType)) {
      return failure();
   }
   parser.resolveOperand(toSort, vecType, result.operands);
   OpAsmParser::OperandType left, right;
   if (parser.parseLParen() || parser.parseRegionArgument(left) || parser.parseComma() || parser.parseRegionArgument(right) || parser.parseRParen()) {
      return failure();
   }
   Region* body = result.addRegion();
   if (parser.parseRegion(*body, {left, right}, {vecType.getElementType(), vecType.getElementType()})) return failure();
   return success();
}
static void print(OpAsmPrinter& p, db::SortOp& op) {
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
static ParseResult parseBuilderMerge(OpAsmParser& parser, OperationState& result) {
   OpAsmParser::OperandType builder, val;
   Type builderType;
   Type valType;
   if (parser.parseOperand(builder) || parser.parseColonType(builderType) || parser.parseComma() || parser.parseOperand(val) || parser.parseColonType(valType)) {
      return failure();
   }
   Region* body = result.addRegion();
   if (parser.parseOptionalLParen().succeeded()) {
      OpAsmParser::OperandType left, right;
      Type leftArgType;
      Type rightArgType;
      if (parser.parseRegionArgument(left) || parser.parseColonType(leftArgType) || parser.parseComma() || parser.parseRegionArgument(right), parser.parseColonType(rightArgType) || parser.parseRParen()) {
         return failure();
      }
      if (parser.parseRegion(*body, {left, right}, {leftArgType, rightArgType})) return failure();
   }
   parser.resolveOperand(builder, builderType, result.operands);
   parser.resolveOperand(val, valType, result.operands);
   parser.addTypeToList(builderType, result.types);
   return success();
}
static void print(OpAsmPrinter& p, db::BuilderMerge& op) {
   p << " " << op.builder() << " : " << op.builder().getType() << ", " << op.val() << " : " << op.val().getType();
   if (!op.fn().empty()) {
      p << "(";
      bool first = true;
      for (auto arg : op.fn().front().getArguments()) {
         if (first) {
            first = false;
         } else {
            p << ",";
         }
         p << arg << ":" << arg.getType();
      }
      p << ")";
      p.printRegion(op.fn(), false, true);
   }
}

LogicalResult mlir::db::OrOp::canonicalize(mlir::db::OrOp orOp, mlir::PatternRewriter& rewriter) {
   llvm::SmallDenseMap<mlir::Value, size_t> usage;
   for (auto val : orOp.vals()) {
      if (!val.getDefiningOp()) return failure();
      if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(val.getDefiningOp())) {
         llvm::SmallPtrSet<mlir::Value, 4> alreadyUsed;
         for (auto andOperand : andOp.vals()) {
            if (!alreadyUsed.contains(andOperand)) {
               usage[andOperand]++;
               alreadyUsed.insert(andOperand);
            }
         }
      } else {
         return failure();
      }
   }
   size_t totalAnds = orOp.vals().size();
   llvm::SmallPtrSet<mlir::Value, 4> extracted;
   std::vector<mlir::Value> newOrOperands;
   for (auto val : orOp.vals()) {
      if (auto andOp = mlir::dyn_cast_or_null<mlir::db::AndOp>(val.getDefiningOp())) {
         std::vector<mlir::Value> keep;
         for (auto andOperand : andOp.vals()) {
            if (usage[andOperand] == totalAnds) {
               extracted.insert(andOperand);
            } else {
               keep.push_back(andOperand);
            }
         }
         if (keep.size() != andOp.vals().size()) {
            if (keep.size()) {
               newOrOperands.push_back(rewriter.create<mlir::db::AndOp>(andOp->getLoc(), keep));
            }
         } else {
            newOrOperands.push_back(andOp);
         }
      }
   }
   std::vector<Value> extractedAsVec;
   extractedAsVec.insert(extractedAsVec.end(), extracted.begin(), extracted.end());
   if (!extracted.empty()) {
      Value newOrOp = rewriter.create<mlir::db::OrOp>(orOp->getLoc(), newOrOperands);
      extractedAsVec.push_back(newOrOp);
      rewriter.replaceOpWithNewOp<mlir::db::AndOp>(orOp, extractedAsVec);
      return success();
   }
   return failure();
}
LogicalResult mlir::db::AndOp::canonicalize(mlir::db::AndOp andOp, mlir::PatternRewriter& rewriter) {
   std::vector<mlir::Value> rawValues;
   std::queue<mlir::Value> queue;
   queue.push(andOp);
   while (!queue.empty()) {
      auto current = queue.front();
      queue.pop();
      if (auto* definingOp = current.getDefiningOp()) {
         if (auto nestedAnd = mlir::dyn_cast_or_null<mlir::db::AndOp>(definingOp)) {
            for (auto v : nestedAnd.vals()) {
               queue.push(v);
            }
         } else {
            rawValues.push_back(current);
         }
      } else {
         rawValues.push_back(current);
      }
   }
   if (rawValues.size() != andOp.vals().size()) {
      rewriter.replaceOpWithNewOp<mlir::db::AndOp>(andOp, rawValues);
      return success();
   }
   return failure();
}
void mlir::db::IfOp::build(OpBuilder& builder, OperationState& result,
                           TypeRange resultTypes, Value cond,
                           function_ref<void(OpBuilder&, Location)> thenBuilder,
                           function_ref<void(OpBuilder&, Location)> elseBuilder) {
   assert(thenBuilder && "the builder callback for 'then' must be present");

   result.addOperands(cond);
   result.addTypes(resultTypes);

   OpBuilder::InsertionGuard guard(builder);
   Region* thenRegion = result.addRegion();
   builder.createBlock(thenRegion);
   thenBuilder(builder, result.location);

   Region* elseRegion = result.addRegion();
   if (!elseBuilder)
      return;

   builder.createBlock(elseRegion);
   elseBuilder(builder, result.location);
}
#define GET_OP_CLASSES
#include "mlir/Dialect/DB/IR/DBOps.cpp.inc"
#include "mlir/Dialect/DB/IR/DBOpsInterfaces.cpp.inc"
