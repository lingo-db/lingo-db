#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <iostream>

#include "mlir-support/parsing.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <variant>
namespace {

struct Matcher {
   public:
   virtual bool matches(mlir::Value) = 0;
   virtual bool skip() { return false; }
   virtual ~Matcher() {}
};
struct AnyMatcher : public Matcher {
   bool matches(mlir::Value) override { return true; }
   virtual ~AnyMatcher() {}
};
std::optional<std::string> getConstantString(mlir::Value v) {
   if (auto* defOp = v.getDefiningOp()) {
      if (auto constOp = mlir::dyn_cast_or_null<mlir::db::ConstantOp>(defOp)) {
         if (auto strAttr = constOp.getValue().dyn_cast<mlir::StringAttr>()) {
            return strAttr.str();
         }
      }
   }
   return std::optional<std::string>();
}
struct ConstStringMatcher : public Matcher {
   ConstStringMatcher() {}
   bool matches(mlir::Value v) override {
      return getConstantString(v).has_value();
   }
   virtual ~ConstStringMatcher() {}
};
struct StringConstMatcher : public Matcher {
   std::string toMatch;
   StringConstMatcher(std::string toMatch) : toMatch(toMatch) {}
   bool matches(mlir::Value v) override {
      auto constStr = getConstantString(v);
      if (!constStr.has_value()) return false;
      return constStr.value() == toMatch;
   }
   bool skip() override {
      return true;
   }
   virtual ~StringConstMatcher() {}
};
class ReplaceFnWithFn : public mlir::RewritePattern {
   std::string funcName;
   std::string newFuncName;
   std::vector<std::shared_ptr<Matcher>> matchers;

   public:
   ReplaceFnWithFn(mlir::MLIRContext* context, std::string funcName, std::vector<std::shared_ptr<Matcher>> matchers, std::string newFuncName) : RewritePattern(mlir::db::RuntimeCall::getOperationName(), mlir::PatternBenefit(1), context), funcName(funcName), newFuncName(newFuncName), matchers(matchers) {}
   mlir::LogicalResult match(mlir::Operation* op) const override {
      auto runtimeCall = mlir::cast<mlir::db::RuntimeCall>(op);
      if (runtimeCall.fn().str() != funcName) { return mlir::failure(); }
      if (runtimeCall.args().size() != matchers.size()) { return mlir::failure(); }
      for (size_t i = 0; i < runtimeCall.args().size(); ++i) {
         if (!matchers[i]->matches(runtimeCall.args()[i])) { return mlir::failure(); }
      }
      return mlir::success();
   }

   void rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      std::vector<mlir::Value> values;
      auto runtimeCall = mlir::cast<mlir::db::RuntimeCall>(op);
      for (size_t i = 0; i < runtimeCall.args().size(); ++i) {
         if (matchers[i]->skip()) {
            continue;
         }
         values.push_back(runtimeCall.args()[i]);
      }
      rewriter.replaceOpWithNewOp<mlir::db::RuntimeCall>(op, op->getResultTypes(), newFuncName, mlir::ValueRange{values});
   }
};
//Pattern that optimizes the join order
class OptimizeRuntimeFunctions : public mlir::PassWrapper<OptimizeRuntimeFunctions, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "db-optimize-runtime-functions"; }

   public:
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.insert<ReplaceFnWithFn>(&getContext(), "ExtractFromDate", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("month"), std::make_shared<AnyMatcher>()}, "ExtractMonthFromDate");
         patterns.insert<ReplaceFnWithFn>(&getContext(), "ExtractFromDate", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("year"), std::make_shared<AnyMatcher>()}, "ExtractYearFromDate");
         patterns.insert<ReplaceFnWithFn>(&getContext(), "ExtractFromDate", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("day"), std::make_shared<AnyMatcher>()}, "ExtractDayFromDate");
         patterns.insert<ReplaceFnWithFn>(&getContext(), "Like", std::vector<std::shared_ptr<Matcher>>{std::make_shared<AnyMatcher>(),std::make_shared<ConstStringMatcher>()}, "ConstLike");
         if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
} // end anonymous namespace

namespace mlir::db {

std::unique_ptr<Pass> createOptimizeRuntimeFunctionsPass() { return std::make_unique<OptimizeRuntimeFunctions>(); }

} // end namespace mlir::db