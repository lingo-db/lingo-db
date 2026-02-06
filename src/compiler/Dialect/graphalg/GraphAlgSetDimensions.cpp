#include <array>

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgAttr.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgDialect.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgOps.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h>
#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>

namespace llvm::cl {

bool parser<graphalg::CallArgumentDimensions>::parse(
   Option& o, StringRef argName, StringRef arg,
   graphalg::CallArgumentDimensions& out) {
   auto err = [&]() {
      return o.error("Call argument dimensions '" + arg + "' are invalid",
                     argName);
   };

   auto remaining = arg;
   unsigned long long rows;
   if (llvm::consumeUnsignedInteger(remaining, 10, rows)) {
      return err();
   }

   if (remaining.front() != 'x') {
      return err();
   }

   remaining = remaining.substr(1);

   unsigned long long cols;
   if (llvm::consumeUnsignedInteger(remaining, 10, cols)) {
      return err();
   }

   out.rows = rows;
   out.cols = cols;

   // Success
   return false;
}

void parser<graphalg::CallArgumentDimensions>::print(
   llvm::raw_ostream& os, const graphalg::CallArgumentDimensions& value) {
   os << value.rows << 'x' << value.cols;
}

} // namespace llvm::cl

namespace graphalg {

#define GEN_PASS_DEF_GRAPHALGSETDIMENSIONS
#include "lingodb/compiler/Dialect/graphalg/GraphAlgPasses.h.inc"

namespace {

/**
 * Set concrete dimensions for the parameters of a function.
 *
 * After this pass, ops in the body of the resulting function exclusively use
 * concrete matrix dimensions.
 */
class GraphAlgSetDimensions
   : public impl::GraphAlgSetDimensionsBase<GraphAlgSetDimensions> {
   using impl::GraphAlgSetDimensionsBase<
      GraphAlgSetDimensions>::GraphAlgSetDimensionsBase;

   void runOnOperation() final;
};

/** Maps abstract dimensions symbols to concrete dimensions. */
class DimMapper {
   private:
   llvm::SmallDenseMap<DimAttr, std::uint64_t> _mapping;

   public:
   static mlir::FailureOr<DimMapper>
   build(mlir::func::FuncOp func, llvm::ArrayRef<CallArgumentDimensions> args);

   mlir::LogicalResult tryMap(mlir::Location loc, DimAttr sym,
                              std::uint64_t dim);

   DimAttr convertAttr(DimAttr attr) const;
};

/** Type converter to change abstract matrix types into concrete ones. */
class DimMapperTypeConverter : public mlir::TypeConverter {
   private:
   const DimMapper& _mapper;

   public:
   DimMapperTypeConverter(const DimMapper& mapper);
};

/** Template for op-specific conversions. */
template <typename T>
class OpConversion : public mlir::OpConversionPattern<T> {
   using mlir::OpConversionPattern<T>::OpConversionPattern;

   mlir::LogicalResult
   matchAndRewrite(T op,
                   typename mlir::OpConversionPattern<T>::OpAdaptor adaptor,
                   mlir::ConversionPatternRewriter& rewriter) const override;
};

/**
 * The standard conversion pattern.
 *
 * Replaces abstract operand, result and block argument types with concrete
 * ones.
 */
class DimConversionPattern : public mlir::ConversionPattern {
   using mlir::ConversionPattern::ConversionPattern;

   public:
   DimConversionPattern(const DimMapperTypeConverter& converter,
                        mlir::MLIRContext* ctx)
      : mlir::ConversionPattern(converter, MatchAnyOpTypeTag{},
                                /*benefit=*/1, ctx) {}

   mlir::LogicalResult
   matchAndRewrite(mlir::Operation* op, llvm::ArrayRef<mlir::Value> operands,
                   mlir::ConversionPatternRewriter& rewriter) const override;
};

/** Template for rewrites (without type conversion). */
template <typename T>
class DimOpRewritePattern : public mlir::OpRewritePattern<T> {
   private:
   const DimMapper& _mapper;

   mlir::LogicalResult
   matchAndRewrite(T op, mlir::PatternRewriter& rewriter) const override;

   public:
   DimOpRewritePattern(const DimMapper& mapper, mlir::MLIRContext* ctx)
      : mlir::OpRewritePattern<T>(ctx), _mapper(mapper) {}
};

} // namespace

mlir::FailureOr<DimMapper>
DimMapper::build(mlir::func::FuncOp func,
                 llvm::ArrayRef<CallArgumentDimensions>
                    args) {
   auto type = func.getFunctionType();
   if (type.getNumInputs() != args.size()) {
      return func->emitOpError("has ")
         << type.getNumInputs() << " parameters, expected " << args.size();
   }

   DimMapper mapper;
   for (auto [i, param, arg] : llvm::enumerate(type.getInputs(), args)) {
      auto matrix = llvm::dyn_cast<MatrixType>(param);
      if (!matrix) {
         return func->emitOpError("parameter ")
            << i << " has type " << param << ", expected " << MatrixType::name;
      }

      if (mlir::failed(
             mapper.tryMap(func->getLoc(), matrix.getRows(), arg.rows))) {
         return mlir::failure();
      }

      if (mlir::failed(
             mapper.tryMap(func->getLoc(), matrix.getCols(), arg.cols))) {
         return mlir::failure();
      }
   }

   return mapper;
}

mlir::LogicalResult DimMapper::tryMap(mlir::Location loc, DimAttr sym,
                                      std::uint64_t dim) {
   if (sym.isConcrete() && sym.getConcreteDim() != dim) {
      return mlir::emitError(loc) << "Concrete dimension " << sym
                                  << " must be mapped to itself, but got " << dim;
   }

   auto [it, newlyAdded] = _mapping.insert({sym, dim});
   if (!newlyAdded && it->second != dim) {
      return mlir::emitError(loc)
         << "Attempt to map abstract dimension " << sym
         << " to concrete dimension " << dim
         << ", but it is already mapped to another concrete dimension "
         << it->second;
   }

   return mlir::success();
}

DimAttr DimMapper::convertAttr(DimAttr dim) const {
   if (_mapping.contains(dim)) {
      return DimAttr::getConcrete(dim.getContext(), _mapping.lookup(dim));
   }

   return nullptr;
}

DimMapperTypeConverter::DimMapperTypeConverter(const DimMapper& mapper)
   : _mapper(mapper) {
   addConversion([](mlir::Type type) {
      // Do not convert non-matrix types
      return type;
   });

   // Convert to concrete dimensions.
   addConversion([&](MatrixType type) {
      auto* ctx = type.getContext();
      std::array<DimAttr, 2> dims = {type.getRows(), type.getCols()};
      for (auto& dim : dims) {
         if (auto newDim = _mapper.convertAttr(dim)) {
            dim = newDim;
         }
      }

      return MatrixType::get(ctx, dims[0], dims[1], type.getSemiring());
   });
}

static bool hasAbstractDimension(mlir::Type type) {
   auto matrix = llvm::dyn_cast<MatrixType>(type);
   if (!matrix) {
      return false;
   }

   return matrix.getRows().isAbstract() || matrix.getCols().isAbstract();
}

static bool anyHasAbstractDimensions(mlir::TypeRange types) {
   return llvm::any_of(types, hasAbstractDimension);
}

static bool doesNotUseAbstractDimensions(mlir::Operation* op) {
   return !anyHasAbstractDimensions(op->getResultTypes()) &&
      !anyHasAbstractDimensions(op->getOperandTypes());
}

static bool doesNotHaveAbstractInputs(mlir::func::FuncOp op) {
   return !anyHasAbstractDimensions(op.getFunctionType().getInputs());
}

template <>
mlir::LogicalResult OpConversion<mlir::func::FuncOp>::matchAndRewrite(
   mlir::func::FuncOp op, OpAdaptor adaptor,
   mlir::ConversionPatternRewriter& rewriter) const {
   // Convert signature.
   auto funcType = op.getFunctionType();
   llvm::SmallVector<mlir::Type> inputTypes;
   if (mlir::failed(
          typeConverter->convertTypes(funcType.getInputs(), inputTypes))) {
      return op->emitOpError("cannot convert input types");
   }

   llvm::SmallVector<mlir::Type> resultTypes;
   if (mlir::failed(
          typeConverter->convertTypes(funcType.getResults(), resultTypes))) {
      return op->emitOpError("cannot convert result types");
   }

   funcType = rewriter.getFunctionType(inputTypes, resultTypes);
   rewriter.modifyOpInPlace(op, [&]() {
      // Apply signature conversion.
      op.setFunctionType(funcType);
   });

   // Convert body.
   if (mlir::failed(
          rewriter.convertRegionTypes(&op.getFunctionBody(), *typeConverter))) {
      return op->emitOpError("cannot convert body types");
   }

   return mlir::success();
}

mlir::LogicalResult DimConversionPattern::matchAndRewrite(
   mlir::Operation* op, llvm::ArrayRef<mlir::Value> operands,
   mlir::ConversionPatternRewriter& rewriter) const {
   // Convert result types.
   llvm::SmallVector<mlir::Type> newResultTypes;
   for (auto res : op->getResults()) {
      if (auto newType = typeConverter->convertType(res.getType())) {
         newResultTypes.emplace_back(newType);
      } else {
         return op->emitOpError("cannot convert result ") << res;
      }
   }

   rewriter.modifyOpInPlace(op, [&]() {
      op->setOperands(operands);

      // Update result types
      for (auto [res, newType] :
           llvm::zip_equal(op->getResults(), newResultTypes)) {
         res.setType(newType);
      }
   });

   // Update regions
   for (auto& region : op->getRegions()) {
      if (mlir::failed(rewriter.convertRegionTypes(&region, *typeConverter))) {
         return op->emitOpError("Cannot convert types of region ")
            << region.getRegionNumber();
      }
   }

   return mlir::success();
}

template <>
mlir::LogicalResult DimOpRewritePattern<CastDimOp>::matchAndRewrite(
   CastDimOp op, mlir::PatternRewriter& rewriter) const {
   auto dim = _mapper.convertAttr(op.getInput());
   if (!dim) {
      return mlir::failure();
   }

   // The folder on CastDimOp should turn this into a constant.
   auto newOp = rewriter.createOrFold<CastDimOp>(op->getLoc(), dim);
   rewriter.replaceOp(op, newOp);

   return mlir::success();
}

template <>
mlir::LogicalResult DimOpRewritePattern<ForDimOp>::matchAndRewrite(
   ForDimOp op, mlir::PatternRewriter& rewriter) const {
   auto dim = _mapper.convertAttr(op.getDim());
   if (!dim) {
      return mlir::failure();
   }

   rewriter.modifyOpInPlace(op, [&]() { op.setDimAttr(dim); });

   return mlir::success();
}

void GraphAlgSetDimensions::runOnOperation() {
   if (functionName.empty()) {
      getOperation().emitError("Missing value for required option 'func'");
      return signalPassFailure();
   }

   auto func = llvm::dyn_cast_if_present<mlir::func::FuncOp>(
      getOperation().lookupSymbol(functionName));
   if (!func) {
      getOperation()->emitOpError("does not contain a function named '")
         << functionName << "'";
      return signalPassFailure();
   }

   auto dimMapper = DimMapper::build(func, argDims);
   if (mlir::failed(dimMapper)) {
      return signalPassFailure();
   }

   mlir::ConversionTarget target(getContext());
   target.addDynamicallyLegalDialect<GraphAlgDialect>(
      doesNotUseAbstractDimensions);
   target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      doesNotUseAbstractDimensions);
   target.addDynamicallyLegalOp<mlir::func::FuncOp>(doesNotHaveAbstractInputs);
   target.addIllegalOp<CastDimOp>();
   target.addIllegalOp<ForDimOp>();

   mlir::RewritePatternSet patterns(&getContext());

   // Convert function signature.
   DimMapperTypeConverter typeConverter(*dimMapper);
   patterns.add<OpConversion<mlir::func::FuncOp>>(typeConverter, &getContext());

   // Convert all result types and block argument types.
   patterns.add<DimConversionPattern>(typeConverter, &getContext());

   // Convert ops that have a special dependency on DimAttr.
   patterns.add<DimOpRewritePattern<CastDimOp>, DimOpRewritePattern<ForDimOp>>(
      *dimMapper, &getContext());
   // Use the canonicalization pattern to rewrite ForDimOp into ForConstOp.
   ForDimOp::getCanonicalizationPatterns(patterns, &getContext());

   if (mlir::failed(
          mlir::applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
   }
}

} // namespace graphalg
