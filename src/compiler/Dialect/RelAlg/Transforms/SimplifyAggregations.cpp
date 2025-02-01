#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"

namespace {
using namespace lingodb::compiler::dialect;
//Pattern that optimizes the join order
class WrapAggrFuncPattern : public mlir::RewritePattern {
   public:
   WrapAggrFuncPattern(mlir::MLIRContext* context)
      : RewritePattern(relalg::AggrFuncOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& attributeManager = getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

      relalg::AggrFuncOp aggrFuncOp = mlir::cast<relalg::AggrFuncOp>(op);
      if (mlir::isa<relalg::AggregationOp, relalg::WindowOp, relalg::GroupJoinOp>(op->getParentOp())) {
         return mlir::success(false);
      }
      std::string scopeName = attributeManager.getUniqueScope("aggr");
      std::string attributeName = "aggrAttr";

      auto def = attributeManager.createDef(scopeName, attributeName);
      def.getColumn().type = aggrFuncOp.getType();
      auto aggrOp = rewriter.create<relalg::AggregationOp>(op->getLoc(), tuples::TupleStreamType::get(getContext()), aggrFuncOp.getRel(), rewriter.getArrayAttr({}), rewriter.getArrayAttr({def}));
      auto* block = new mlir::Block;
      aggrOp.getAggrFunc().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto tplType = tuples::TupleType::get(getContext());
         block->addArgument(tuples::TupleStreamType::get(getContext()), op->getLoc());
         block->addArgument(tplType, op->getLoc());

         auto relArgument = block->getArgument(0);
         auto val = rewriter.create<relalg::AggrFuncOp>(op->getLoc(), aggrFuncOp.getType(), aggrFuncOp.getFn(), relArgument, aggrFuncOp.getAttr());
         rewriter.create<tuples::ReturnOp>(op->getLoc(), mlir::ValueRange({val}));
      }
      auto nullableType = mlir::dyn_cast_or_null<db::NullableType>(aggrFuncOp.getType());
      mlir::Value getScalarOp = rewriter.replaceOpWithNewOp<relalg::GetScalarOp>(op, nullableType, attributeManager.createRef(&def.getColumn()), aggrOp.asRelation());
      mlir::Value res = getScalarOp;
      if (!nullableType) {
         res = rewriter.create<db::NullableGetVal>(op->getLoc(), aggrFuncOp.getType(), getScalarOp);
      }
      rewriter.replaceOp(op, res);
      return mlir::success(true);
   }
};
class WrapCountRowsPattern : public mlir::RewritePattern {
   public:
   WrapCountRowsPattern(mlir::MLIRContext* context)
      : RewritePattern(relalg::CountRowsOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& attributeManager = getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

      relalg::CountRowsOp aggrFuncOp = mlir::cast<relalg::CountRowsOp>(op);
      if (mlir::isa<relalg::AggregationOp, relalg::GroupJoinOp>(op->getParentOp())) {
         return mlir::success(false);
      }
      std::string scopeName = attributeManager.getUniqueScope("aggr");
      std::string attributeName = "aggrAttr";

      auto def = attributeManager.createDef(scopeName, attributeName);
      def.getColumn().type = aggrFuncOp.getType();
      auto aggrOp = rewriter.create<relalg::AggregationOp>(op->getLoc(), tuples::TupleStreamType::get(getContext()), aggrFuncOp.getRel(), rewriter.getArrayAttr({}), rewriter.getArrayAttr({def}));
      auto* block = new mlir::Block;
      aggrOp.getAggrFunc().push_back(block);
      {
         mlir::OpBuilder::InsertionGuard insertionGuard(rewriter);
         rewriter.setInsertionPointToStart(block);
         auto tplType = tuples::TupleType::get(getContext());
         block->addArgument(tuples::TupleStreamType::get(getContext()), op->getLoc());
         block->addArgument(tplType, op->getLoc());

         auto relArgument = block->getArgument(0);
         auto val = rewriter.create<relalg::CountRowsOp>(op->getLoc(), aggrFuncOp.getType(), relArgument);
         rewriter.create<tuples::ReturnOp>(op->getLoc(), mlir::ValueRange({val}));
      }
      mlir::Type nullableType = aggrFuncOp.getType();
      if (!mlir::isa<db::NullableType>(nullableType)) {
         nullableType = db::NullableType::get(rewriter.getContext(), nullableType);
      }
      mlir::Value getScalarOp = rewriter.create<relalg::GetScalarOp>(op->getLoc(), nullableType, attributeManager.createRef(&def.getColumn()), aggrOp.asRelation());
      mlir::Value res = rewriter.create<db::AsNullableOp>(op->getLoc(), aggrFuncOp.getType(), getScalarOp);
      rewriter.replaceOp(op, res);
      return mlir::success(true);
   }
};

class RewriteComplexAggrFuncs : public mlir::RewritePattern {
   public:
   RewriteComplexAggrFuncs(mlir::MLIRContext* context)
      : RewritePattern(relalg::AggrFuncOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& attrManager = getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      auto loc = op->getLoc();
      auto* parentOp = op->getParentOp();
      if (!(mlir::isa<relalg::WindowOp>(parentOp) || mlir::isa<relalg::AggregationOp>(parentOp))) return mlir::failure();
      auto aggrFuncOp = mlir::cast<relalg::AggrFuncOp>(op);
      if (aggrFuncOp.getFn() == relalg::AggrFunc::stddev_samp) {
         mlir::Value varSamp = rewriter.create<relalg::AggrFuncOp>(loc, aggrFuncOp.getResult().getType(), relalg::AggrFunc::var_samp, aggrFuncOp.getRel(), aggrFuncOp.getAttr());
         rewriter.replaceOpWithNewOp<db::RuntimeCall>(aggrFuncOp, aggrFuncOp.getResult().getType(), "Sqrt", varSamp);
         return mlir::success();
      }
      if (aggrFuncOp.getFn() == relalg::AggrFunc::var_samp) {
         auto rel = aggrFuncOp.getRel();
         auto xType = aggrFuncOp.getResult().getType();
         auto asDoubleAttr = attrManager.createDef(attrManager.getUniqueScope("var_samp"), "asDouble");
         auto squaredAttr = attrManager.createDef(attrManager.getUniqueScope("var_samp"), "x2");
         squaredAttr.getColumn().type = mlir::isa<db::NullableType>(aggrFuncOp.getAttr().getColumn().type) ? (mlir::Type) db::NullableType::get(getContext(), rewriter.getF64Type()) : (mlir::Type) rewriter.getF64Type();
         asDoubleAttr.getColumn().type = mlir::isa<db::NullableType>(aggrFuncOp.getAttr().getColumn().type) ? (mlir::Type) db::NullableType::get(getContext(), rewriter.getF64Type()) : (mlir::Type) rewriter.getF64Type();

         auto mapOp = rewriter.create<relalg::MapOp>(op->getLoc(), tuples::TupleStreamType::get(getContext()), rel, rewriter.getArrayAttr({squaredAttr, asDoubleAttr}));
         auto* block = new mlir::Block;
         auto tuple = block->addArgument(tuples::TupleType::get(getContext()), loc);
         {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(block);
            auto x = rewriter.create<tuples::GetColumnOp>(loc, aggrFuncOp.getAttr().getColumn().type, aggrFuncOp.getAttr(), tuple);
            mlir::Value asDouble = rewriter.create<db::CastOp>(loc, mlir::isa<db::NullableType>(x.getType()) ? (mlir::Type) db::NullableType::get(getContext(), rewriter.getF64Type()) : (mlir::Type) rewriter.getF64Type(), x);
            mlir::Value squared = rewriter.create<db::MulOp>(loc, asDouble, asDouble);
            rewriter.create<tuples::ReturnOp>(loc, mlir::ValueRange{squared, asDouble});
         }
         mapOp.getPredicate().push_back(block);
         auto squaredAttrRef = attrManager.createRef(&squaredAttr.getColumn());
         auto asDoubleAttrRef = attrManager.createRef(&asDoubleAttr.getColumn());
         mlir::Value sumSquared = rewriter.create<relalg::AggrFuncOp>(loc, xType, relalg::AggrFunc::sum, mapOp.getResult(), squaredAttrRef);
         auto originalType = xType;
         mlir::Value sum = rewriter.create<relalg::AggrFuncOp>(loc, mlir::isa<db::NullableType>(originalType) ? originalType : db::NullableType::get(originalType), relalg::AggrFunc::sum, rel, asDoubleAttrRef);

         mlir::Value count = rewriter.create<relalg::AggrFuncOp>(loc, rewriter.getI64Type(), relalg::AggrFunc::count, rel, asDoubleAttrRef);
         mlir::Value squareSum = rewriter.create<db::MulOp>(loc, sum, sum);
         mlir::Value castedCount = rewriter.create<db::CastOp>(loc, getBaseType(sumSquared.getType()), count);

         mlir::Value one = rewriter.create<db::ConstantOp>(loc, castedCount.getType(), rewriter.getI64IntegerAttr(1));
         mlir::Value countM1 = rewriter.create<db::SubOp>(loc, castedCount, one);
         mlir::Value div1 = rewriter.create<db::DivOp>(loc, squareSum, castedCount);
         mlir::Value sub1 = rewriter.create<db::SubOp>(loc, sumSquared, div1);
         auto zero = rewriter.create<db::ConstantOp>(loc, castedCount.getType(), rewriter.getI64IntegerAttr(0));
         mlir::Value isZero = rewriter.create<db::CmpOp>(loc, db::DBCmpPredicate::eq, countM1, zero);
         mlir::Value result = rewriter.create<mlir::scf::IfOp>(
                                         loc, isZero, [&](mlir::OpBuilder& builder, mlir::Location loc) {
                                             mlir::Value null=builder.create<db::NullOp>(loc,sub1.getType());
                                            builder.create<mlir::scf::YieldOp>(loc,null); }, [&](mlir::OpBuilder& builder, mlir::Location loc) {
                                            mlir::Value average=builder.create<db::DivOp>(loc, sub1,countM1);
                                            builder.create<mlir::scf::YieldOp>(loc,average); })
                                 .getResult(0);
         rewriter.replaceOp(aggrFuncOp, result);
         return mlir::success();
      }
      if (aggrFuncOp.getFn() == relalg::AggrFunc::avg) {
         auto asNullable = [&](mlir::Type t) -> mlir::Type {
            if (mlir::isa<db::NullableType>(t)) {
               return t;
            }
            if (mlir::isa<db::NullableType>(aggrFuncOp.getType())) {
               return db::NullableType::get(t.getContext(), t);
            } else {
               return t;
            }
         };
         mlir::Value sum = rewriter.create<relalg::AggrFuncOp>(loc, asNullable(aggrFuncOp.getAttr().getColumn().type), relalg::AggrFunc::sum, aggrFuncOp.getRel(), aggrFuncOp.getAttr());
         mlir::Value count = rewriter.create<relalg::AggrFuncOp>(loc, rewriter.getI64Type(), relalg::AggrFunc::count, aggrFuncOp.getRel(), aggrFuncOp.getAttr());
         if (!mlir::isa<db::DecimalType>(getBaseType(sum.getType())) && mlir::isa<db::DecimalType>(getBaseType(aggrFuncOp.getResult().getType()))) {
            sum = rewriter.create<db::CastOp>(loc, asNullable(db::DecimalType::get(getContext(), 19, 0)), sum); //todo
         }
         auto baseType = getBaseType(sum.getType());

         mlir::Value casted = rewriter.create<db::CastOp>(loc, mlir::isa<db::DecimalType>(baseType) ? db::DecimalType::get(getContext(), 19, 0) : baseType, count);
         rewriter.replaceOpWithNewOp<db::DivOp>(aggrFuncOp, sum, casted);
         return mlir::success();
      }
      return mlir::failure();
   }
};

class SimplifyAggregations : public mlir::PassWrapper<SimplifyAggregations, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-simplify-aggrs"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyAggregations)
   private:
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::scf::SCFDialect>();
   }

   public:
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.insert<WrapAggrFuncPattern>(&getContext());
         patterns.insert<WrapCountRowsPattern>(&getContext());
         patterns.insert<RewriteComplexAggrFuncs>(&getContext());

         if (mlir::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
      getOperation()
         .walk([&](relalg::AggregationOp aggregationOp) {
            mlir::Value arg = aggregationOp.getAggrFunc().front().getArgument(0);
            std::vector<mlir::Operation*> users(arg.getUsers().begin(), arg.getUsers().end());
            for (auto* user : users) {
               if (auto mapOp = mlir::dyn_cast_or_null<relalg::MapOp>(user)) {
                  mapOp->moveBefore(aggregationOp);
                  mapOp.replaceAllUsesWith(aggregationOp.getAggrFunc().front().getArgument(0));
                  mapOp->setOperand(0, aggregationOp.getRel());
                  aggregationOp->setOperand(0, mapOp.getResult());
               }
            }
         });
      getOperation()
         .walk([&](relalg::WindowOp aggregationOp) {
            mlir::Value arg = aggregationOp.getAggrFunc().front().getArgument(0);
            std::vector<mlir::Operation*> users(arg.getUsers().begin(), arg.getUsers().end());
            for (auto* user : users) {
               if (auto mapOp = mlir::dyn_cast_or_null<relalg::MapOp>(user)) {
                  mapOp->moveBefore(aggregationOp);
                  mapOp.replaceAllUsesWith(aggregationOp.getAggrFunc().front().getArgument(0));
                  mapOp->setOperand(0, aggregationOp.getRel());
                  aggregationOp->setOperand(0, mapOp.getResult());
               }
            }
         });
      auto& attrManager = getOperation().getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      getOperation()
         .walk([&](relalg::AggregationOp aggregationOp) {
            auto scope = attrManager.getUniqueScope("aggr_rw");
            auto computedCols = aggregationOp.getComputedCols();
            std::vector<mlir::Value> computedValues;
            std::vector<mlir::Attribute> computedColsAfter;
            llvm::DenseMap<mlir::Value, tuples::ColumnRefAttr> aggrMapping;
            std::vector<mlir::Attribute> colsForMap;
            std::vector<mlir::Value> valsForMap;
            auto returnOp = mlir::cast<tuples::ReturnOp>(aggregationOp.getAggrFunc().front().getTerminator());
            for (size_t i = 0; i < returnOp->getNumOperands(); i++) {
               auto returnValue = returnOp.getOperand(i);
               auto isDirectAggregate = mlir::isa_and_nonnull<relalg::AggrFuncOp, relalg::CountRowsOp>(returnValue.getDefiningOp());
               if (isDirectAggregate) {
                  computedValues.push_back(returnValue);
                  computedColsAfter.push_back(computedCols[i]);
               } else {
                  colsForMap.push_back(computedCols[i]);
                  valsForMap.push_back(returnValue);
               }
            }
            size_t id = 0;
            for (auto& op : aggregationOp.getAggrFunc().front()) {
               if (mlir::isa<relalg::AggrFuncOp, relalg::CountRowsOp>(&op)) {
                  bool otherUser = false;
                  for (auto* user : op.getUsers()) {
                     otherUser |= !mlir::isa<tuples::ReturnOp>(user);
                  }
                  if (otherUser) {
                     auto attr = attrManager.createDef(scope, "rw" + std::to_string(id++));
                     attr.getColumn().type = op.getResult(0).getType();
                     computedValues.push_back(op.getResult(0));
                     computedColsAfter.push_back(attr);
                     aggrMapping.insert({op.getResult(0), attrManager.createRef(&attr.getColumn())});
                  }
               }
            }
            mlir::OpBuilder builder(aggregationOp);
            aggregationOp.setComputedColsAttr(builder.getArrayAttr(computedColsAfter));
            returnOp->setOperands(computedValues);

            if (!colsForMap.empty()) {
               auto* block = new mlir::Block;
               builder.setInsertionPointAfter(aggregationOp);
               mlir::IRMapping mapping;
               auto loc = aggregationOp->getLoc();
               auto newmap = builder.create<relalg::MapOp>(aggregationOp->getLoc(), tuples::TupleStreamType::get(builder.getContext()), aggregationOp, builder.getArrayAttr(colsForMap));
               newmap.getPredicate().push_back(block);
               auto tuple = newmap.getPredicate().addArgument(tuples::TupleType::get(builder.getContext()), loc);
               builder.setInsertionPointToStart(&newmap.getPredicate().front());
               std::vector<mlir::Operation*> getOps;
               for (auto [v, c] : aggrMapping) {
                  auto newVal = builder.create<tuples::GetColumnOp>(loc, v.getType(), c, tuple);
                  mapping.map(v, newVal);
                  getOps.push_back(newVal);
               }
               std::sort(valsForMap.begin(), valsForMap.end(), [](mlir::Value v1, mlir::Value v2) {
                  assert(v1.getDefiningOp() && v2.getDefiningOp());
                  return v1.getDefiningOp()->isBeforeInBlock(v2.getDefiningOp());
               });
               auto placeHolder = builder.create<tuples::ReturnOp>(loc);
               for (auto v : valsForMap) {
                  relalg::detail::inlineOpIntoBlock(v.getDefiningOp(), v.getDefiningOp()->getParentOp(), &newmap.getLambdaBlock(), mapping, placeHolder);
               }
               placeHolder.erase();
               for (auto* op : getOps) {
                  op->moveBefore(block, block->begin());
               }
               std::vector<mlir::Value> returnValues;
               for (auto v : valsForMap) {
                  returnValues.push_back(mapping.lookup(v));
               }
               builder.create<tuples::ReturnOp>(loc, returnValues);
               aggregationOp.getResult().replaceAllUsesExcept(newmap.getResult(), newmap);
            }
         });
      getOperation()
         .walk([&](relalg::WindowOp aggregationOp) {
            auto scope = attrManager.getUniqueScope("aggr_rw");
            auto computedCols = aggregationOp.getComputedCols();
            std::vector<mlir::Value> computedValues;
            std::vector<mlir::Attribute> computedColsAfter;
            llvm::DenseMap<mlir::Value, tuples::ColumnRefAttr> aggrMapping;
            std::vector<mlir::Attribute> colsForMap;
            std::vector<mlir::Value> valsForMap;
            auto returnOp = mlir::cast<tuples::ReturnOp>(aggregationOp.getAggrFunc().front().getTerminator());
            for (size_t i = 0; i < returnOp->getNumOperands(); i++) {
               auto returnValue = returnOp.getOperand(i);
               auto isDirectAggregate = mlir::isa_and_nonnull<relalg::AggrFuncOp, relalg::CountRowsOp, relalg::RankOp>(returnValue.getDefiningOp());
               if (isDirectAggregate) {
                  computedValues.push_back(returnValue);
                  computedColsAfter.push_back(computedCols[i]);
               } else {
                  colsForMap.push_back(computedCols[i]);
                  valsForMap.push_back(returnValue);
               }
            }
            size_t id = 0;
            for (auto& op : aggregationOp.getAggrFunc().front()) {
               if (mlir::isa<relalg::AggrFuncOp, relalg::CountRowsOp, relalg::RankOp>(&op)) {
                  bool otherUser = false;
                  for (auto* user : op.getUsers()) {
                     otherUser |= !mlir::isa<tuples::ReturnOp>(user);
                  }
                  if (otherUser) {
                     auto attr = attrManager.createDef(scope, "rw" + std::to_string(id++));
                     attr.getColumn().type = op.getResult(0).getType();
                     computedValues.push_back(op.getResult(0));
                     computedColsAfter.push_back(attr);
                     aggrMapping.insert({op.getResult(0), attrManager.createRef(&attr.getColumn())});
                  }
               }
            }
            mlir::OpBuilder builder(aggregationOp);
            aggregationOp.setComputedColsAttr(builder.getArrayAttr(computedColsAfter));
            returnOp->setOperands(computedValues);

            if (!colsForMap.empty()) {
               auto* block = new mlir::Block;
               builder.setInsertionPointAfter(aggregationOp);
               mlir::IRMapping mapping;
               auto loc = aggregationOp->getLoc();
               auto newmap = builder.create<relalg::MapOp>(aggregationOp->getLoc(), tuples::TupleStreamType::get(builder.getContext()), aggregationOp, builder.getArrayAttr(colsForMap));
               newmap.getPredicate().push_back(block);
               auto tuple = newmap.getPredicate().addArgument(tuples::TupleType::get(builder.getContext()), loc);
               builder.setInsertionPointToStart(&newmap.getPredicate().front());
               std::vector<mlir::Operation*> getOps;
               for (auto [v, c] : aggrMapping) {
                  auto newVal = builder.create<tuples::GetColumnOp>(loc, v.getType(), c, tuple);
                  mapping.map(v, newVal);
                  getOps.push_back(newVal);
               }
               for (auto v : valsForMap) {
                  relalg::detail::inlineOpIntoBlock(v.getDefiningOp(), v.getDefiningOp()->getParentOp(), &newmap.getLambdaBlock(), mapping);
               }
               for (auto* op : getOps) {
                  op->moveBefore(block, block->begin());
               }
               std::vector<mlir::Value> returnValues;
               for (auto v : valsForMap) {
                  returnValues.push_back(mapping.lookup(v));
               }
               builder.create<tuples::ReturnOp>(loc, returnValues);
               aggregationOp.getResult().replaceAllUsesExcept(newmap.getResult(), newmap);
            }
         });
   }
};
} // end anonymous namespace


std::unique_ptr<mlir::Pass> relalg::createSimplifyAggregationsPass() { return std::make_unique<SimplifyAggregations>(); }
