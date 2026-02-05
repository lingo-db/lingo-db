/** Canonicalization and folding for graphalg ops. */
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgInterfaces.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>
#include <lingodb/compiler/Dialect/graphalg/SemiringTypes.h>

namespace graphalg {

mlir::OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
   if (auto constInput = adaptor.getInput()) {
      return constInput;
   }

   if (getType().isScalar()) {
      return getInput();
   }

   // e.T.T => e
   if (auto childTransOp = getInput().getDefiningOp<TransposeOp>()) {
      return childTransOp.getInput();
   }

   return nullptr;
}

mlir::OpFoldResult DiagOp::fold(FoldAdaptor adaptor) {
   if (getType().isScalar()) {
      return getInput();
   }

   return nullptr;
}

// Turn a matrix multiply with scalar arguments into a scalar multiply.
static mlir::LogicalResult matMulScalar(MatMulOp op,
                                        mlir::PatternRewriter& rewriter) {
   if (!op.getLhs().getType().isScalar() || !op.getRhs().getType().isScalar()) {
      return mlir::failure();
   }

   auto applyOp = rewriter.replaceOpWithNewOp<ApplyOp>(
      op, op.getType(), mlir::ValueRange{op.getLhs(), op.getRhs()});
   auto& body = applyOp.createBody();

   rewriter.setInsertionPointToStart(&body);

   auto lhsArg = body.getArgument(0);
   auto rhsArg = body.getArgument(1);
   auto mulOp = rewriter.create<MulOp>(applyOp->getLoc(), lhsArg, rhsArg);
   rewriter.create<ApplyReturnOp>(applyOp->getLoc(), mulOp);

   return mlir::success();
}

static bool isMulIdentityMatrix(mlir::Value v) {
   auto constOp = v.getDefiningOp<ConstantMatrixOp>();
   if (!constOp) {
      return false;
   }

   auto sring =
      llvm::cast<SemiringTypeInterface>(constOp.getType().getSemiring());
   return constOp.getValue() == sring.mulIdentity();
}

static mlir::LogicalResult matMulBroadcast(MatMulOp op,
                                           mlir::PatternRewriter& rewriter) {
   if (!op.getLhs().getType().isColumnVector() ||
       !op.getRhs().getType().isRowVector()) {
      return mlir::failure();
   }

   if (isMulIdentityMatrix(op.getLhs())) {
      rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), op.getRhs());
      return mlir::success();
   } else if (isMulIdentityMatrix(op.getRhs())) {
      rewriter.replaceOpWithNewOp<BroadcastOp>(op, op.getType(), op.getLhs());
      return mlir::success();
   }

   return mlir::failure();
}

void MatMulOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                           mlir::MLIRContext* context) {
   patterns.add(matMulScalar);
   patterns.add(matMulBroadcast);
}

mlir::OpFoldResult ReduceOp::fold(FoldAdaptor adaptor) {
   if (getInput().getType() == getType()) {
      // Not reducing along any dimension.
      return getInput();
   }

   return nullptr;
}

mlir::OpFoldResult CastDimOp::fold(FoldAdaptor adaptor) {
   if (getInput().isConcrete()) {
      return mlir::IntegerAttr::get(SemiringTypes::forInt(getContext()),
                                    getInput().getConcreteDim());
   }

   return nullptr;
}

static mlir::LogicalResult applyConstantInput(ApplyOp op,
                                              mlir::PatternRewriter& rewriter) {
   ConstantMatrixOp constantInput;
   unsigned argNum = 0;
   for (auto& oper : op->getOpOperands()) {
      auto constOp = oper.get().getDefiningOp<ConstantMatrixOp>();
      if (constOp) {
         constantInput = constOp;
         argNum = oper.getOperandNumber();
         break;
      }
   }

   if (!constantInput) {
      return mlir::failure();
   }

   // Insert the constant into the apply body
   auto& body = op.getBody().front();
   rewriter.setInsertionPointToStart(&body);
   auto constOp = rewriter.create<ConstantOp>(constantInput.getLoc(),
                                              constantInput.getValue());

   // Replace all uses of the block argument with the in-body constant.
   rewriter.replaceAllUsesWith(body.getArgument(argNum), constOp);

   // Note: Folding will remove the now-redundant input and block argument.
   return mlir::success();
}

// Inlines the body of producer into consumer, creating a single ApplyOp that
// is equivalent to consumer without depending on producer.
mlir::LogicalResult inlineApply(ApplyOp consumer, ApplyOp producer,
                                unsigned int producerIdx,
                                mlir::PatternRewriter& rewriter) {
   assert(consumer->getOperand(producerIdx) == producer);

   // NOTE: Keep the producer as an input, and let the folder get rid of it.
   llvm::SmallVector<mlir::Value> newInputs(consumer.getInputs());
   newInputs.append(producer.getInputs().begin(), producer.getInputs().end());

   auto newOp = rewriter.create<ApplyOp>(consumer->getLoc(), consumer.getType(),
                                         newInputs);
   auto& newBody = newOp.getBody().emplaceBlock();

   mlir::IRMapping mapping;
   // Create block arguments
   llvm::SmallVector<mlir::Value> consumerArgReplacements;
   for (auto cArg : consumer.getBody().getArguments()) {
      auto arg = newBody.addArgument(cArg.getType(), consumer.getLoc());
      consumerArgReplacements.emplace_back(arg);
   }

   for (auto pArg : producer.getBody().getArguments()) {
      auto arg = newBody.addArgument(pArg.getType(), pArg.getLoc());
      mapping.map(pArg, arg);
   }

   rewriter.setInsertionPointToStart(&newBody);

   // Clone the producer body first: We want its output for the body of the
   // consumer.
   mlir::Value producerResult;
   for (mlir::Operation& op : producer.getBody().front()) {
      if (auto returnOp = llvm::dyn_cast<ApplyReturnOp>(op)) {
         producerResult = mapping.lookup(returnOp.getValue());
         break;
      }

      auto* newOp = rewriter.clone(op, mapping);
      mapping.map(&op, newOp);
   }

   // Now inline the consumer body
   consumerArgReplacements[producerIdx] = producerResult;
   rewriter.inlineBlockBefore(&consumer.getBody().front(), &newBody,
                              newBody.end(), consumerArgReplacements);

   rewriter.replaceOp(consumer, newOp);
   return mlir::success();
}

// Find ApplyOp in the input that we can merge into this one.
mlir::LogicalResult applyMerge(ApplyOp op, mlir::PatternRewriter& rewriter) {
   for (auto& oper : op->getOpOperands()) {
      if (auto inputOp = oper.get().getDefiningOp<ApplyOp>()) {
         if (inputOp->hasOneUse()) {
            // Can inline into this op.
            return inlineApply(op, inputOp, oper.getOperandNumber(), rewriter);
         }
      }
   }

   return mlir::failure();
}

void ApplyOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                          mlir::MLIRContext* context) {
   patterns.add(applyConstantInput);
   patterns.add(applyMerge);
}

mlir::OpFoldResult LiteralOp::fold(FoldAdaptor adaptor) { return getValue(); }

mlir::OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
   if (auto input = adaptor.getInput()) {
      return input;
   } else if (getInput().getType() == getType()) {
      // Nothing to broadcast
      return getInput();
   } else if (auto childBroadcast = getInput().getDefiningOp<BroadcastOp>()) {
      // broadcast of broadcast is redundant.
      getInputMutable().assign(childBroadcast.getInput());
      return getResult();
   }

   return nullptr;
}

static mlir::LogicalResult forDimConst(ForDimOp op,
                                       mlir::PatternRewriter& rewriter) {
   if (!op.getDim().isConcrete()) {
      return mlir::failure();
   }

   // The number of iterations is known, so we can replace with a ForConstOp.
   auto end = op.getDim().getConcreteDim();

   // Range from 0 to dim.
   auto intType =
      MatrixType::scalarOf(SemiringTypes::forInt(rewriter.getContext()));
   auto beginOp = rewriter.create<ConstantMatrixOp>(
      op->getLoc(), intType, rewriter.getI64IntegerAttr(0));
   auto endOp = rewriter.create<ConstantMatrixOp>(
      op->getLoc(), intType, rewriter.getI64IntegerAttr(end));

   auto forConstOp = rewriter.create<ForConstOp>(
      op->getLoc(), op->getResultTypes(), op.getInitArgs(), beginOp, endOp);
   rewriter.inlineRegionBefore(op.getBody(), forConstOp.getBody(),
                               forConstOp.getBody().begin());
   rewriter.replaceOp(op, forConstOp);

   return mlir::success();
}

void ForDimOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                           mlir::MLIRContext* context) {
   patterns.add(forDimConst);
}

mlir::OpFoldResult PickAnyOp::fold(FoldAdaptor adaptor) {
   if (getType().isColumnVector()) {
      return getInput();
   }

   return nullptr;
}

// Omit reduction over boolean matrices if it is directly followed by PickAnyOp.
static mlir::LogicalResult pickAnyReduceBool(PickAnyOp op,
                                             mlir::PatternRewriter& rewriter) {
   if (!op.getType().isBoolean()) {
      return mlir::failure();
   }

   auto reduceOp = op.getInput().getDefiningOp<DeferredReduceOp>();
   if (!reduceOp) {
      return mlir::failure();
   }

   // Relax to a union
   auto unionOp = rewriter.createOrFold<UnionOp>(op.getLoc(), reduceOp.getType(),
                                                 reduceOp.getInputs());
   rewriter.modifyOpInPlace(op, [&]() { op.getInputMutable().assign(unionOp); });

   return mlir::failure();
}

void PickAnyOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                            mlir::MLIRContext* context) {
   patterns.add(pickAnyReduceBool);
}

mlir::OpFoldResult TrilOp::fold(FoldAdaptor adaptor) {
   if (getType().isScalar()) {
      return getInput();
   }

   return nullptr;
}

mlir::OpFoldResult TriuOp::fold(FoldAdaptor adaptor) {
   if (getType().isScalar()) {
      return getInput();
   }

   return nullptr;
}

mlir::OpFoldResult ApplyOp::fold(FoldAdaptor adaptor) {
   if (!getBody().hasOneBlock()) {
      // No body.
      return nullptr;
   }

   auto& body = getBody().front();
   if (!body.mightHaveTerminator()) {
      // No terminator.
      return nullptr;
   }

   if (body.getNumArguments() != getNumOperands()) {
      // Invalid input to body arg mapping.
      return nullptr;
   }

   auto returnOp =
      llvm::dyn_cast_if_present<ApplyReturnOp>(body.getTerminator());
   if (!returnOp) {
      // Not the expected return op.
      return nullptr;
   }

   // Fold if body reduces to a constant
   mlir::Attribute constantValue;
   if (mlir::matchPattern(returnOp.getValue(),
                          mlir::m_Constant(&constantValue))) {
      return constantValue;
   }

   // If we are returning an input unchanged, refer to that input directly.
   if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(returnOp.getValue())) {
      auto input = getInputs()[arg.getArgNumber()];
      if (input.getType() == getType()) {
         // Only possible if we are not broadcasting the input.
         return getInputs()[arg.getArgNumber()];
      }
   }

   // If one of the inputs is not used in the body, remove it.
   for (auto arg : body.getArguments()) {
      if (arg.use_empty()) {
         getInputsMutable().erase(arg.getArgNumber());
         body.eraseArgument(arg.getArgNumber());
         // Modified in-place.
         return getResult();
      }
   }

   // If one of the inputs is a broadcast, refer to the child instead.
   for (auto [i, input] : llvm::enumerate(getInputs())) {
      if (auto broadcastOp = input.getDefiningOp<BroadcastOp>()) {
         llvm::SmallVector<mlir::Value> newInputs(getInputs());
         newInputs[i] = broadcastOp.getInput();
         getInputsMutable().assign(newInputs);
         return getResult();
      }
   }

   return nullptr;
}

mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

mlir::OpFoldResult ConstantMatrixOp::fold(FoldAdaptor adaptor) {
   return getValue();
}

mlir::OpFoldResult CastScalarOp::fold(FoldAdaptor adaptor) {
   auto inRing = llvm::cast<SemiringTypeInterface>(getInput().getType());
   auto outRing = llvm::cast<SemiringTypeInterface>(getType());

   // Casting to the input type is redundant.
   if (inRing == outRing) {
      return getInput();
   }

   // Constant folding.
   if (auto typedAttr =
          llvm::dyn_cast_if_present<mlir::TypedAttr>(adaptor.getInput())) {
      auto* dialect = getContext()->getLoadedDialect<lingodb/compiler/Dialect/graphalgDialect>();
      // Note: castAttribute returns nullptr if the cast failed, preventing a
      // fold if the cast fails.
      return dialect->castAttribute(typedAttr, outRing);
   }

   return nullptr;
}

mlir::OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
   auto sring = llvm::cast<SemiringTypeInterface>(getType());
   auto lhs = llvm::dyn_cast_if_present<mlir::TypedAttr>(adaptor.getLhs());
   auto rhs = llvm::dyn_cast_if_present<mlir::TypedAttr>(adaptor.getRhs());
   if (lhs && rhs) {
      return sring.add(lhs, rhs);
   }

   // Add with identity does nothing.
   if (lhs == sring.addIdentity()) {
      return getRhs();
   } else if (rhs == sring.addIdentity()) {
      return getLhs();
   }

   return nullptr;
}

mlir::OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
   auto sring = llvm::cast<SemiringTypeInterface>(getType());
   auto lhs = llvm::dyn_cast_if_present<mlir::TypedAttr>(adaptor.getLhs());
   auto rhs = llvm::dyn_cast_if_present<mlir::TypedAttr>(adaptor.getRhs());
   if (lhs && rhs) {
      return sring.mul(lhs, rhs);
   }

   // Multiply with identity does nothing.
   if (adaptor.getLhs() == sring.mulIdentity()) {
      return getRhs();
   } else if (adaptor.getRhs() == sring.mulIdentity()) {
      return getLhs();
   }

   // Multiply with additive identity annihilates.
   if (adaptor.getLhs() == sring.addIdentity() ||
       adaptor.getRhs() == sring.addIdentity()) {
      return sring.addIdentity();
   }

   return nullptr;
}

namespace {

//** Matcher for (x == false) */
struct IsNot {
   mlir::Value input;
   bool match(mlir::Operation* op);
};

} // namespace

static bool isFalse(mlir::Value v) {
   mlir::Attribute attr;
   if (mlir::matchPattern(v, mlir::m_Constant(&attr))) {
      return attr == mlir::BoolAttr::get(v.getContext(), false);
   }

   return false;
}

bool IsNot::match(mlir::Operation* op) {
   auto eqOp = llvm::dyn_cast<EqOp>(op);
   if (!eqOp) {
      return false;
   }

   if (isFalse(eqOp.getLhs())) {
      input = eqOp.getRhs();
      return true;
   }

   if (isFalse(eqOp.getRhs())) {
      input = eqOp.getLhs();
      return true;
   }

   return false;
}

mlir::OpFoldResult EqOp::fold(FoldAdaptor adaptor) {
   if (getLhs() == getRhs()) {
      return mlir::BoolAttr::get(getContext(), true);
   }

   auto lhs = adaptor.getLhs();
   auto rhs = adaptor.getRhs();
   if (lhs && rhs) {
      // Note: Two attributes with the same value are uniqued and mapped to
      // the same constant value. If the attribute values indeed match,
      // usually the condition above would have already folded this to true.
      return mlir::BoolAttr::get(getContext(), lhs == rhs);
   }

   // ((x == false) == false) => x
   IsNot isNot;
   if (mlir::matchPattern(this->getOperation(), isNot)) {
      IsNot nested;
      if (mlir::matchPattern(isNot.input, nested)) {
         return nested.input;
      }
   }

   return nullptr;
}

// Test if the value is equal to the additive identity.
static bool isAdditiveIdentity(mlir::Value v) {
   mlir::Attribute constantValue;
   if (!mlir::matchPattern(v, mlir::m_Constant(&constantValue))) {
      return false;
   }

   auto semiring = llvm::dyn_cast<SemiringTypeInterface>(v.getType());
   if (!semiring) {
      return false;
   }

   return constantValue == semiring.addIdentity();
}

mlir::OpFoldResult MakeDenseOp::fold(FoldAdaptor adaptor) {
   if (!getBody().hasOneBlock()) {
      // No body.
      return nullptr;
   }

   auto& body = getBody().front();
   if (!body.mightHaveTerminator()) {
      // No terminator.
      return nullptr;
   }

   auto returnOp =
      llvm::dyn_cast_if_present<MakeDenseReturnOp>(body.getTerminator());
   if (!returnOp) {
      // Not the expected return op.
      return nullptr;
   }

   if (isAdditiveIdentity(returnOp.getValue())) {
      // Unconditionally returns zero, so we can omit making the input dense.
      return getInput();
   }

   // If one of the inputs is not used in the body, remove it.
   for (auto arg : body.getArguments()) {
      if (arg.use_empty()) {
         body.eraseArgument(arg.getArgNumber());
         // Modified in-place.
         return getResult();
      }
   }

   return nullptr;
}

mlir::OpFoldResult UnionOp::fold(FoldAdaptor adaptor) {
   if (getInputs().size() == 1 && getInputs()[0].getType() == getType()) {
      return getInputs()[0];
   }

   bool hasUnionInputs = llvm::any_of(
      getInputs(), [](mlir::Value v) { return v.getDefiningOp<UnionOp>(); });
   if (hasUnionInputs) {
      // Flatten nested union.
      llvm::SmallVector<mlir::Value> flatInputs;
      for (auto input : getInputs()) {
         if (auto nestedOp = input.getDefiningOp<UnionOp>()) {
            flatInputs.append(nestedOp.getInputs().begin(),
                              nestedOp.getInputs().end());
         } else {
            flatInputs.emplace_back(input);
         }
      }

      getInputsMutable().assign(flatInputs);
      return getResult();
   }

   return nullptr;
}

static bool allConstant(llvm::ArrayRef<mlir::Attribute> values) {
   return llvm::all_of(values, [](mlir::Attribute attr) { return !!attr; });
}

static bool isOverScalars(DeferredReduceOp op) {
   return op.getType().isScalar() &&
      llvm::all_of(op.getInputs(), [&](mlir::Value v) {
             return llvm::cast<MatrixType>(v.getType()).isScalar();
          });
}

mlir::OpFoldResult DeferredReduceOp::fold(FoldAdaptor adaptor) {
   // Fold to a constant
   if (allConstant(adaptor.getInputs()) && isOverScalars(*this)) {
      auto sring = llvm::cast<SemiringTypeInterface>(getType().getSemiring());
      auto value = sring.addIdentity();
      for (auto input : adaptor.getInputs()) {
         value = sring.add(value, llvm::cast<mlir::TypedAttr>(input));
      }

      return value;
   }

   // Merge nested reduce ops
   bool haveNested = llvm::any_of(getInputs(), [](mlir::Value v) {
      return v.getDefiningOp<DeferredReduceOp>();
   });
   if (haveNested) {
      llvm::SmallVector<mlir::Value> newInputs;
      for (auto input : getInputs()) {
         if (auto nestedOp = input.getDefiningOp<DeferredReduceOp>()) {
            auto nestedInputs = nestedOp.getInputs();
            newInputs.append(nestedInputs.begin(), nestedInputs.end());
         } else {
            newInputs.emplace_back(input);
         }
      }

      getInputsMutable().assign(newInputs);
      return getResult();
   }

   return nullptr;
}

} // namespace graphalg
