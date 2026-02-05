#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LogicalResult.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>
#include <lingodb/compiler/Dialect/graphalg/SemiringTypes.h>

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/graphalg/GraphAlgOps.cpp.inc"

namespace graphalg {

// === TransposeOp ===
mlir::LogicalResult TransposeOp::inferReturnTypes(
   mlir::MLIRContext* ctx, std::optional<mlir::Location> location,
   Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
   auto inputType = llvm::cast<MatrixType>(adaptor.getInput().getType());
   inferredReturnTypes.emplace_back(MatrixType::get(
      ctx, inputType.getCols(), inputType.getRows(), inputType.getSemiring()));
   return mlir::success();
}

// === DiagOp ===
mlir::LogicalResult DiagOp::inferReturnTypes(
   mlir::MLIRContext* ctx, std::optional<mlir::Location> location,
   Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
   auto inputType = llvm::cast<MatrixType>(adaptor.getInput().getType());
   // Get the non-1 dimension if there is one.
   auto dim =
      inputType.getRows().isOne() ? inputType.getCols() : inputType.getRows();

   inferredReturnTypes.emplace_back(
      MatrixType::get(ctx, dim, dim, inputType.getSemiring()));
   return mlir::success();
}

// === ApplyUnaryOp ===

// Keep the dimensions of the first input, but take the semiring of the function
// result type.
static MatrixType buildApplyResultType(mlir::func::FuncOp func,
                                       mlir::Value firstInput) {
   auto firstInputType = llvm::cast<MatrixType>(firstInput.getType());

   assert(func.getFunctionType().getNumResults() == 1);
   auto funcResultType =
      llvm::cast<MatrixType>(func.getFunctionType().getResult(0));

   return MatrixType::get(func->getContext(), firstInputType.getRows(),
                          firstInputType.getCols(),
                          funcResultType.getSemiring());
}

static mlir::LogicalResult
verifyCalleeSignature(mlir::SymbolTableCollection& symbolTable,
                      mlir::CallOpInterface callOp,
                      mlir::FunctionType expected) {
   auto callable =
      llvm::dyn_cast_if_present<mlir::func::FuncOp>(callOp.resolveCallable());
   if (!callable) {
      return callOp->emitOpError("is expecting a function declaration with name ")
         << llvm::cast<mlir::SymbolRefAttr>(callOp.getCallableForCallee());
   }

   auto actual = callable.getFunctionType();
   if (actual != expected) {
      auto diag = callOp->emitOpError("is expecting callable to have type ")
         << expected << ", got " << actual;
      diag.attachNote(callable->getLoc()) << "Call target";
      return diag;
   }

   return mlir::success();
}

void ApplyUnaryOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                         mlir::func::FuncOp func, mlir::Value input) {
   build(builder, state, buildApplyResultType(func, input), func.getSymName(),
         input);
}

mlir::LogicalResult
ApplyUnaryOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
   auto expectedFunctionType = mlir::FunctionType::get(
      getContext(), getInput().getType().asScalar(), getType().asScalar());
   return verifyCalleeSignature(symbolTable, *this, expectedFunctionType);
}

// Return the callee of the generic call operation, this is required by the
// call interface.
mlir::CallInterfaceCallable ApplyUnaryOp::getCallableForCallee() {
   return getFuncAttr();
}

static mlir::FlatSymbolRefAttr
castToFunctionRef(mlir::CallInterfaceCallable callee) {
   return llvm::cast<mlir::FlatSymbolRefAttr>(
      llvm::cast<mlir::SymbolRefAttr>(callee));
}

// Set the callee for the generic call operation, this is required by the call
// interface.
void ApplyUnaryOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
   setFuncAttr(castToFunctionRef(callee));
}

// Get the argument operands to the called function, this is required by the
// call interface.
mlir::Operation::operand_range ApplyUnaryOp::getArgOperands() {
   return getOperation()->getOperands();
}

mlir::MutableOperandRange ApplyUnaryOp::getArgOperandsMutable() {
   return getOperation();
}

// === ApplyBinaryOp ===
void ApplyBinaryOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                          mlir::func::FuncOp func, mlir::Value lhs,
                          mlir::Value rhs) {
   build(builder, state, buildApplyResultType(func, lhs), func.getSymName(), lhs,
         rhs);
}

mlir::LogicalResult
ApplyBinaryOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
   auto expectedFunctionType = mlir::FunctionType::get(
      getContext(),
      std::array<mlir::Type, 2>{getLhs().getType().asScalar(),
                                getRhs().getType().asScalar()},
      getType().asScalar());
   return verifyCalleeSignature(symbolTable, *this, expectedFunctionType);
}

mlir::CallInterfaceCallable ApplyBinaryOp::getCallableForCallee() {
   return getFuncAttr();
}

void ApplyBinaryOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
   setFuncAttr(castToFunctionRef(callee));
}

mlir::Operation::operand_range ApplyBinaryOp::getArgOperands() {
   return getOperation()->getOperands();
}

mlir::MutableOperandRange ApplyBinaryOp::getArgOperandsMutable() {
   return getOperation();
}

// === ApplyElementWiseOp ===
void ApplyElementWiseOp::build(mlir::OpBuilder& builder,
                               mlir::OperationState& state,
                               mlir::func::FuncOp func, mlir::Value lhs,
                               mlir::Value rhs) {
   build(builder, state, buildApplyResultType(func, lhs), func.getSymName(), lhs,
         rhs);
}

mlir::LogicalResult
ApplyElementWiseOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
   auto expectedFunctionType = mlir::FunctionType::get(
      getContext(),
      std::array<mlir::Type, 2>{getLhs().getType().asScalar(),
                                getRhs().getType().asScalar()},
      getType().asScalar());
   return verifyCalleeSignature(symbolTable, *this, expectedFunctionType);
}

mlir::CallInterfaceCallable ApplyElementWiseOp::getCallableForCallee() {
   return getFuncAttr();
}

void ApplyElementWiseOp::setCalleeFromCallable(
   mlir::CallInterfaceCallable callee) {
   setFuncAttr(castToFunctionRef(callee));
}

mlir::Operation::operand_range ApplyElementWiseOp::getArgOperands() {
   return getOperation()->getOperands();
}

mlir::MutableOperandRange ApplyElementWiseOp::getArgOperandsMutable() {
   return getOperation();
}

// === SelectUnaryOp ===
mlir::LogicalResult
SelectUnaryOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
   auto expectedFunctionType = mlir::FunctionType::get(
      getContext(), getInput().getType().asScalar(),
      MatrixType::scalarOf(SemiringTypes::forBool(getContext())));
   return verifyCalleeSignature(symbolTable, *this, expectedFunctionType);
}

mlir::CallInterfaceCallable SelectUnaryOp::getCallableForCallee() {
   return getFuncAttr();
}

void SelectUnaryOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
   setFuncAttr(castToFunctionRef(callee));
}

mlir::Operation::operand_range SelectUnaryOp::getArgOperands() {
   return getOperation()->getOperands();
}

mlir::MutableOperandRange SelectUnaryOp::getArgOperandsMutable() {
   return getOperation();
}

// === SelectBinaryOp ===
mlir::LogicalResult
SelectBinaryOp::verifySymbolUses(mlir::SymbolTableCollection& symbolTable) {
   // Verify function type matches
   auto lhsType = getLhs().getType();
   auto rhsType = getRhs().getType();
   auto expectedFunctionType = mlir::FunctionType::get(
      getContext(),
      std::array<mlir::Type, 2>{lhsType.asScalar(), rhsType.asScalar()},
      MatrixType::scalarOf(SemiringTypes::forBool(getContext())));
   return verifyCalleeSignature(symbolTable, *this, expectedFunctionType);
}

mlir::CallInterfaceCallable SelectBinaryOp::getCallableForCallee() {
   return getFuncAttr();
}

void SelectBinaryOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
   setFuncAttr(castToFunctionRef(callee));
}

mlir::Operation::operand_range SelectBinaryOp::getArgOperands() {
   return getOperation()->getOperands();
}

mlir::MutableOperandRange SelectBinaryOp::getArgOperandsMutable() {
   return getOperation();
}

// === MatMulOp ===
static MatrixType inferMatMulReturnType(mlir::Value lhs, mlir::Value rhs) {
   auto lhsType = llvm::cast<MatrixType>(lhs.getType());
   auto rhsType = llvm::cast<MatrixType>(rhs.getType());
   return MatrixType::get(lhsType.getContext(), lhsType.getRows(),
                          rhsType.getCols(), lhsType.getSemiring());
}

mlir::LogicalResult MatMulOp::inferReturnTypes(
   mlir::MLIRContext* ctx, std::optional<mlir::Location> location,
   Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
   inferredReturnTypes.emplace_back(
      inferMatMulReturnType(adaptor.getLhs(), adaptor.getRhs()));
   return mlir::success();
}

mlir::LogicalResult MatMulOp::verify() {
   // Verify the dimensions
   if (getLhs().getType().getCols() != getRhs().getType().getRows()) {
      return emitOpError("is expecting shapes of the LHS (")
         << getLhs().getType() << ") and the RHS (" << getRhs().getType()
         << ") to be compatible, but the number of columns in the LHS ("
         << getLhs().getType().getCols()
         << ") does not match the number of rows in the RHS ("
         << getRhs().getType().getRows() << ")";
   }

   return mlir::success();
}

// === VecMatMulOp ===
mlir::LogicalResult VecMatMulOp::inferReturnTypes(
   mlir::MLIRContext* ctx, std::optional<mlir::Location> location,
   Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
   auto lhsType = llvm::cast<MatrixType>(adaptor.getLhs().getType());
   auto rhsType = llvm::cast<MatrixType>(adaptor.getRhs().getType());
   inferredReturnTypes.emplace_back(MatrixType::get(
      ctx, rhsType.getCols(), DimAttr::getOne(ctx), lhsType.getSemiring()));
   return mlir::success();
}

mlir::LogicalResult VecMatMulOp::verify() {
   // Verify the dimensions
   if (getLhs().getType().getRows() != getRhs().getType().getRows()) {
      return emitError("The LHS and RHS shapes are incompatible: ")
         << "The number of rows in the LHS do not match the number of "
         << "rows in the RHS";
   }

   return mlir::success();
}

// === ElementWiseOp ===
mlir::LogicalResult ElementWiseOp::inferReturnTypes(
   mlir::MLIRContext* ctx, std::optional<mlir::Location> location,
   Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
   // Note: assumes lhs and rhs type match.
   auto lhsType = llvm::cast<MatrixType>(adaptor.getLhs().getType());
   MatrixType returnType;
   if (binaryOpIsCompare(adaptor.getOp())) {
      // Same shape as the input, but with boolean semiring.
      returnType = MatrixType::get(ctx, lhsType.getRows(), lhsType.getCols(),
                                   SemiringTypes::forBool(ctx));
   } else {
      returnType = lhsType;
   }

   inferredReturnTypes.emplace_back(returnType);
   return mlir::success();
}

// === ReduceOp ===
static mlir::LogicalResult verifyCollapseDim(ReduceOp op,
                                             llvm::StringRef dimName,
                                             DimAttr resultDim,
                                             DimAttr inputDim) {
   if (!resultDim.isOne() && resultDim != inputDim) {
      op->emitOpError("result ")
         << dimName << " " << resultDim
         << " are not collapsed to 1 and do not match the input " << dimName
         << " " << inputDim;
   }

   return mlir::success();
}

mlir::LogicalResult ReduceOp::verify() {
   auto inputType = getInput().getType();
   if (mlir::failed(verifyCollapseDim(*this, "rows", getType().getRows(),
                                      inputType.getRows()))) {
      return mlir::failure();
   }

   if (mlir::failed(verifyCollapseDim(*this, "columns", getType().getCols(),
                                      inputType.getCols()))) {
      return mlir::failure();
   }

   return mlir::success();
}

// === CastOp ===
static mlir::LogicalResult verifyCast(mlir::Operation* op, mlir::Type from,
                                      mlir::Type to) {
   auto* dialect = op->getContext()->getLoadedDialect<GraphAlgDialect>();
   if (!dialect->isCastLegal(from, to)) {
      return op->emitError("Cast from semiring ")
         << from << " to " << to << " is not allowed";
   }

   return mlir::success();
}

mlir::LogicalResult CastOp::verify() {
   auto from = getInput().getType().getSemiring();
   auto to = getType().getSemiring();
   return verifyCast(getOperation(), from, to);
}

// === LiteralOp ===
mlir::LogicalResult LiteralOp::inferReturnTypes(
   mlir::MLIRContext* ctx, std::optional<mlir::Location> location,
   Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
   inferredReturnTypes.emplace_back(
      MatrixType::scalarOf(adaptor.getValue().getType()));
   return mlir::success();
}

// === ConstantMatrixOp ===
mlir::LogicalResult ConstantMatrixOp::verify() {
   if (getType().getSemiring() != getValue().getType()) {
      return emitOpError("is expecting a constant value of type ")
         << getType().getSemiring() << ", got " << getValue().getType();
   }

   return mlir::success();
}

mlir::LogicalResult verifyLoop(mlir::Operation* op, mlir::ValueRange initArgs,
                               mlir::Region& bodyRegion,
                               mlir::Region& untilRegion) {
   llvm::SmallVector<mlir::Type> initArgTypes;
   for (auto arg : initArgs) {
      initArgTypes.emplace_back(arg.getType());
   }

   if (op->getResultTypes() != initArgTypes) {
      return op->emitOpError("result types ")
         << op->getResultTypes() << " do not match init args "
         << initArgTypes;
   }

   // Block arguments must start with an iteration counter
   auto* ctx = op->getContext();
   auto iterType = MatrixType::scalarOf(SemiringTypes::forInt(ctx));
   for (auto& region : op->getRegions()) {
      if (region.empty()) {
         continue;
      }

      mlir::TypeRange argTypes = region.getArgumentTypes();
      if (argTypes.empty() || argTypes.front() != iterType) {
         return op->emitOpError("region types ")
            << argTypes << "do not include the iteration variable";
      }
   }

   // Body must have YieldOp as terminator
   auto& body = bodyRegion.front();
   if (!body.mightHaveTerminator()) {
      return op->emitOpError("body region does not have a terminator");
   }
   auto bodyYield = llvm::dyn_cast_if_present<YieldOp>(body.getTerminator());
   if (!bodyYield) {
      return op->emitOpError("body region is not terminated with a YieldOp");
   }

   // If there is an until block, it should return a boolean.
   if (!untilRegion.empty()) {
      auto& until = untilRegion.front();
      if (!until.mightHaveTerminator()) {
         return op->emitOpError("until region does not have a terminator");
      }
      auto untilYield = llvm::dyn_cast_if_present<YieldOp>(until.getTerminator());
      if (!untilYield) {
         return op->emitOpError("until region is not terminated with a YieldOp");
      }

      auto expectedType = MatrixType::scalarOf(SemiringTypes::forBool(ctx));
      if (untilYield->getOperandTypes() != mlir::TypeRange{expectedType}) {
         return op->emitOpError("until block does not return a bool scalar: ")
            << untilYield->getOperandTypes();
      }
   }

   return mlir::success();
}

void getLoopSuccessorRegions(
   mlir::Operation* op, mlir::Region& body, mlir::Region& until,
   mlir::RegionBranchPoint point,
   llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions) {
   if (body.args_empty() || (!until.empty() && until.args_empty())) {
      // Missing iter var. Let the verifier deal with it.
      return;
   }

   if (point.isParent()) {
      // Can go straight to the result
      regions.emplace_back(mlir::RegionSuccessor(op->getResults()));
      // Or into the body
      regions.emplace_back(
         mlir::RegionSuccessor(&body, body.getArguments().drop_front()));
   } else if (point.getRegionOrNull() == &body) {
      // Terminate the op.
      regions.emplace_back(mlir::RegionSuccessor(op->getResults()));
      // Recurse into body.
      regions.emplace_back(
         mlir::RegionSuccessor(&body, body.getArguments().drop_front()));
      // Go to until check
      if (!until.empty()) {
         regions.emplace_back(
            mlir::RegionSuccessor(&until, until.getArguments().drop_front()));
      }
   } else if (point.getRegionOrNull() == &until) {
      // Not passing any modified state to other blocks, so in terms of
      // dataflow we do not have successor regions.
   }
}

// === BroadcastOp ===
mlir::LogicalResult BroadcastOp::verify() {
   auto [inRows, inCols] = getInput().getType().getDims();
   auto [outRows, outCols] = getType().getDims();
   if (inRows != outRows && !inRows.isOne()) {
      return emitOpError("output rows ")
         << outRows << " are incompatible with input rows " << inRows;
   } else if (inCols != outCols && !inCols.isOne()) {
      return emitOpError("output cols ")
         << outCols << " are incompatible with input cols " << inCols;
   }

   return mlir::success();
}

// === ForConstOp ===
mlir::LogicalResult ForConstOp::verifyRegions() {
   return verifyLoop(getOperation(), getInitArgs(), getBody(), getUntil());
}

void ForConstOp::getSuccessorRegions(
   mlir::RegionBranchPoint point,
   llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions) {
   getLoopSuccessorRegions(getOperation(), getBody(), getUntil(), point,
                           regions);
}

mlir::OperandRange
ForConstOp::getEntrySuccessorOperands(mlir::RegionBranchPoint point) {
   return getInitArgs();
}

// === ForDimOp ===
mlir::LogicalResult ForDimOp::verifyRegions() {
   return verifyLoop(getOperation(), getInitArgs(), getBody(), getUntil());
}

void ForDimOp::getSuccessorRegions(
   mlir::RegionBranchPoint point,
   llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions) {
   getLoopSuccessorRegions(getOperation(), getBody(), getUntil(), point,
                           regions);
}

mlir::OperandRange
ForDimOp::getEntrySuccessorOperands(mlir::RegionBranchPoint point) {
   return getInitArgs();
}

// === YieldOp ===
mlir::MutableOperandRange
YieldOp::getMutableSuccessorOperands(mlir::RegionBranchPoint point) {
   return getInputsMutable();
}

// === ApplyInlineOp ===
void ApplyInlineOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                          mlir::ValueRange inputs, mlir::Type semiring) {
   assert(0 < inputs.size());
   // Derive output shape from the first argument.
   auto firstInputType = llvm::cast<MatrixType>(inputs[0].getType());
   build(builder, state, firstInputType.withSemiring(semiring), inputs);
   auto& block = state.regions.front()->emplaceBlock();
   for (auto input : inputs) {
      auto inputType = llvm::cast<MatrixType>(input.getType());
      assert(firstInputType.getDims() == inputType.getDims());
      block.addArgument(inputType.asScalar(), input.getLoc());
   }
}

static mlir::LogicalResult
verifyInputDimensionsMatchOutput(mlir::Operation* op, MatrixType output,
                                 mlir::ValueRange inputs) {
   auto dims = output.getDims();
   for (auto [i, input] : llvm::enumerate(inputs)) {
      auto inputType = llvm::cast<MatrixType>(input.getType());
      if (inputType.getDims() != dims) {
         return op->emitOpError("is expecting all inputs to have the same ")
            << "dimensions as the result, but the input at index " << i
            << " has type " << inputType << " (expected dimensions to match "
            << output << ")";
      }
   }

   return mlir::success();
}

mlir::LogicalResult ApplyInlineOp::verify() {
   if (mlir::failed(
          verifyInputDimensionsMatchOutput(*this, getType(), getInputs()))) {
      return mlir::failure();
   }

   // Body arguments must be scalar matrix versions of the input types.
   auto& body = getBody().front();
   auto args = body.getArguments();
   if (args.size() != getInputs().size()) {
      return emitOpError("has ") << getInputs().size() << " inputs, but body has "
                                 << args.size() << " arguments";
   }

   for (auto [i, input, arg] : llvm::enumerate(getInputs(), args)) {
      auto inputType = llvm::cast<MatrixType>(input.getType());
      auto argType = llvm::dyn_cast<MatrixType>(arg.getType());
      if (!argType || !argType.isScalar()) {
         return emitOpError(" is expecting all body arguments to be scalar ")
            << "matrices, but the argument at index " << i << " is of type "
            << argType;
      }

      if (inputType.getSemiring() != argType.getSemiring()) {
         return emitOpError("semirings of input and body argument at index ")
            << i << " do not match (" << inputType << " vs. " << argType
            << ")";
      }
   }

   return mlir::success();
}

mlir::LogicalResult ApplyInlineOp::verifyRegions() {
   auto& body = getBody().front();
   if (!body.mightHaveTerminator()) {
      return emitOpError("body does not have a terminator");
   }

   auto returnOp =
      llvm::dyn_cast_if_present<ApplyInlineReturnOp>(body.getTerminator());
   if (!returnOp) {
      return emitOpError("body is not terminated with ")
         << ApplyInlineReturnOp::getOperationName();
   }

   auto expectType = MatrixType::scalarOf(getType().getSemiring());
   auto returnType = returnOp.getValue().getType();
   if (returnType != expectType) {
      return emitOpError("is expecting the final ")
         << ApplyInlineReturnOp::getOperationName()
         << " to return a value of type " << expectType << ", but got "
         << returnType;
   }

   return mlir::success();
}

// === ApplyOp ===
mlir::Block& ApplyOp::createBody() {
   auto& block = getBody().emplaceBlock();
   for (auto input : getInputs()) {
      auto inputType = llvm::cast<MatrixType>(input.getType());
      block.addArgument(inputType.getSemiring(), input.getLoc());
   }

   return block;
}

mlir::LogicalResult ApplyOp::verify() {
   auto [outRows, outCols] = getType().getDims();
   for (auto input : getInputs()) {
      auto type = llvm::cast<MatrixType>(input.getType());
      auto [inRows, inCols] = type.getDims();
      if (inRows != outRows && !inRows.isOne()) {
         auto diag =
            emitOpError("input and output row dimensions are incompatible");
         diag.attachNote(input.getLoc()) << "incompatible input is " << input;
      }

      if (inCols != outCols && !inCols.isOne()) {
         auto diag =
            emitOpError("input and output column dimensions are incompatible");
         diag.attachNote(input.getLoc()) << "incompatible input is " << input;
      }
   }

   // Body arguments must be scalar versions of the input types.
   auto& body = getBody().front();
   auto args = body.getArguments();
   if (args.size() != getInputs().size()) {
      return emitOpError("has ") << getInputs().size() << " inputs, but body has "
                                 << args.size() << " arguments";
   }

   for (auto [i, input, arg] : llvm::enumerate(getInputs(), args)) {
      auto inputType = llvm::cast<MatrixType>(input.getType());
      auto argType = arg.getType();
      if (!llvm::isa<SemiringTypeInterface>(argType)) {
         return emitOpError(" is expecting all body arguments to have ")
            << "semiring types, but the argument at index " << i
            << " is of type " << argType;
      }

      if (inputType.getSemiring() != argType) {
         return emitOpError("semirings of input and body argument at index ")
            << i << " do not match (" << inputType << " vs. " << argType
            << ")";
      }
   }

   return mlir::success();
}

mlir::LogicalResult ApplyOp::verifyRegions() {
   auto& body = getBody().front();
   if (!body.mightHaveTerminator()) {
      return emitOpError("body does not have a terminator");
   }

   auto returnOp =
      llvm::dyn_cast_if_present<ApplyReturnOp>(body.getTerminator());
   if (!returnOp) {
      return emitOpError("body is not terminated with ")
         << ApplyReturnOp::getOperationName();
   }

   auto expectType = getType().getSemiring();
   auto returnType = returnOp.getValue().getType();
   if (returnType != expectType) {
      return emitOpError("is expecting the final ")
         << ApplyReturnOp::getOperationName() << " to return a value of type "
         << expectType << ", but got " << returnType;
   }

   return mlir::success();
}

// === CastScalarOp ===
mlir::LogicalResult CastScalarOp::verify() {
   return verifyCast(getOperation(), getInput().getType(), getType());
}

// === MakeDenseOp ===
mlir::LogicalResult MakeDenseOp::verifyRegions() {
   // Body args are scalar semiring types.
   for (auto arg : getBody().getArguments()) {
      if (!llvm::isa<SemiringTypeInterface>(arg.getType())) {
         return emitOpError("is expecting body arguments to have semiring types")
            << ", but argument " << arg.getArgNumber() << " has type "
            << arg.getType() << ", which is not a semiring";
      }
   }

   // Terminated with MakeDenseReturnOp
   auto& body = getBody().front();
   if (!body.mightHaveTerminator()) {
      return emitOpError("body does not have a terminator");
   }

   auto returnOp =
      llvm::dyn_cast_if_present<MakeDenseReturnOp>(body.getTerminator());
   if (!returnOp) {
      return emitOpError("body is not terminated with ")
         << MakeDenseReturnOp::getOperationName();
   }

   return mlir::success();
}

mlir::LogicalResult MatMulJoinOp::inferReturnTypes(
   mlir::MLIRContext* ctx, std::optional<mlir::Location> location,
   Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
   inferredReturnTypes.emplace_back(
      inferMatMulReturnType(adaptor.getLhs(), adaptor.getRhs()));
   return mlir::success();
}

static mlir::LogicalResult verifyUnionOperandType(mlir::Operation* op) {
   auto resultType = llvm::cast<MatrixType>(op->getResult(0).getType());
   auto outRows = resultType.getRows();
   auto outCols = resultType.getCols();

   for (auto input : op->getOperands()) {
      auto inputType = llvm::cast<MatrixType>(input.getType());
      auto rows = inputType.getRows();
      auto cols = inputType.getCols();

      if (!outRows.isOne() && outRows != rows) {
         return op->emitOpError("input ")
            << input << " has " << rows
            << " rows, which is incompatible with the output type "
            << resultType;
      }

      if (!outCols.isOne() && outCols != cols) {
         return op->emitOpError("input ")
            << input << " has " << cols
            << " columns, which is incompatible with the output type "
            << resultType;
      }
   }

   return mlir::success();
}

mlir::LogicalResult UnionOp::verify() { return verifyUnionOperandType(*this); }

mlir::LogicalResult DeferredReduceOp::verify() {
   return verifyUnionOperandType(*this);
}

} // namespace graphalg
