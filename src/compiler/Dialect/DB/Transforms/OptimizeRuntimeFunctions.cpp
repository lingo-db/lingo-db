#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <iostream>

#include "lingodb/compiler/Dialect/DB/Passes.h"
#include "lingodb/compiler/helper.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>
#include <variant>
namespace {
using namespace lingodb::compiler::dialect;
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

bool isDeterministicLikePattern(std::string_view str, char escape = '\\') {
   // deterministic: patterns enclosed by '%' can contain only placeholders or only deterministic characters
   bool sawPercent = false;
   bool sawPlaceholder = false;
   bool sawCharacter = false;
   bool prevWasEscape = false;
   for (char c : str) {
      if (prevWasEscape) {
         sawCharacter = true;
         continue;
      }
      if (c == escape) {
         prevWasEscape = true;
      } else if (c == '%') {
         if (sawPercent && sawPlaceholder && sawCharacter) return false;
         sawPercent = true;
         sawPlaceholder = false;
         sawCharacter = false;
      } else if (c == '_') {
         sawPlaceholder = true;
      } else {
         sawCharacter = true;
      }
   }
   return true;
}
std::optional<std::string> getConstantString(mlir::Value v) {
   if (auto* defOp = v.getDefiningOp()) {
      if (auto constOp = mlir::dyn_cast_or_null<lingodb::compiler::dialect::db::ConstantOp>(defOp)) {
         if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(constOp.getValue())) {
            return strAttr.str();
         }
      }
   }
   return std::optional<std::string>();
}
struct ConstantDeterministicPatternMatcher : public Matcher {
   ConstantDeterministicPatternMatcher() {}
   bool matches(mlir::Value v) override {
      auto constStringOpt = getConstantString(v);
      return constStringOpt.has_value() && isDeterministicLikePattern(constStringOpt.value());
   }
   virtual ~ConstantDeterministicPatternMatcher() {}
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
   ReplaceFnWithFn(mlir::MLIRContext* context, std::string funcName, std::vector<std::shared_ptr<Matcher>> matchers, std::string newFuncName) : RewritePattern(db::RuntimeCall::getOperationName(), mlir::PatternBenefit(1), context), funcName(funcName), newFuncName(newFuncName), matchers(matchers) {}
   mlir::LogicalResult match(mlir::Operation* op) const override {
      auto runtimeCall = mlir::cast<db::RuntimeCall>(op);
      if (runtimeCall.getFn().str() != funcName) { return mlir::failure(); }
      if (runtimeCall.getArgs().size() != matchers.size()) { return mlir::failure(); }
      for (size_t i = 0; i < runtimeCall.getArgs().size(); ++i) {
         if (!matchers[i]->matches(runtimeCall.getArgs()[i])) { return mlir::failure(); }
      }
      return mlir::success();
   }

   void rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      std::vector<mlir::Value> values;
      auto runtimeCall = mlir::cast<db::RuntimeCall>(op);
      for (size_t i = 0; i < runtimeCall.getArgs().size(); ++i) {
         if (matchers[i]->skip()) {
            continue;
         }
         values.push_back(runtimeCall.getArgs()[i]);
      }
      rewriter.replaceOpWithNewOp<db::RuntimeCall>(op, op->getResultTypes(), newFuncName, mlir::ValueRange{values});
   }
};

class SimplifyRepeatedStringConcat : public mlir::RewritePattern {
   static std::vector<mlir::Value> collectConcatArgs(mlir::Value v) {
      if (auto runtimeCall2 = mlir::dyn_cast_or_null<db::RuntimeCall>(v.getDefiningOp())) {
         if (runtimeCall2.getFn().str() == "Concatenate") {
            return std::vector<mlir::Value>{runtimeCall2.getArgs()[0], runtimeCall2.getArgs()[1]};
         } else if (runtimeCall2.getFn().str() == "ConcatenateMultiple") {
            return std::vector<mlir::Value>(runtimeCall2.getArgs().begin(), runtimeCall2.getArgs().end());
         }
      }
      return {};
   }

   public:
   SimplifyRepeatedStringConcat(mlir::MLIRContext* context)
      : RewritePattern(db::RuntimeCall::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto runtimeCall = mlir::cast<db::RuntimeCall>(op);
      if (runtimeCall.getFn().str() == "Concatenate") {
         if (runtimeCall.getArgs().size() != 2) {
            return mlir::failure();
         }
         auto args = collectConcatArgs(runtimeCall.getArgs()[0]);
         if (!args.empty()) {
            args.push_back(runtimeCall.getArgs()[1]);
            rewriter.replaceOpWithNewOp<db::RuntimeCall>(op, op->getResultTypes(), "ConcatenateMultiple", mlir::ValueRange{args});
            return mlir::success();
         }

      } else if (runtimeCall.getFn().str() == "ConcatenateMultiple") {
         auto args = collectConcatArgs(runtimeCall.getArgs()[0]);
         if (!args.empty()) {
            std::vector<mlir::Value> newArgs;
            newArgs.insert(newArgs.end(), args.begin(), args.end());
            for (size_t i = 1; i < runtimeCall.getArgs().size(); ++i) {
               newArgs.push_back(runtimeCall.getArgs()[i]);
            }
            rewriter.replaceOpWithNewOp<db::RuntimeCall>(op, op->getResultTypes(), "ConcatenateMultiple", mlir::ValueRange{newArgs});
            return mlir::success();
         }
      }
      return mlir::failure();
   }
};
} // end anonymous namespace

namespace lingodb::compiler::dialect::db {

void addOptimizeRuntimeFunctionPatterns(mlir::RewritePatternSet& patterns) {
   auto* ctxt = patterns.getContext();
   patterns.insert<ReplaceFnWithFn>(ctxt, "ExtractFromDate", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("year"), std::make_shared<AnyMatcher>()}, "ExtractYearFromDate");
   patterns.insert<ReplaceFnWithFn>(ctxt, "ExtractFromDate", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("month"), std::make_shared<AnyMatcher>()}, "ExtractMonthFromDate");
   patterns.insert<ReplaceFnWithFn>(ctxt, "ExtractFromDate", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("day"), std::make_shared<AnyMatcher>()}, "ExtractDayFromDate");
   patterns.insert<ReplaceFnWithFn>(ctxt, "ExtractFromDate", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("hour"), std::make_shared<AnyMatcher>()}, "ExtractHourFromDate");
   patterns.insert<ReplaceFnWithFn>(ctxt, "ExtractFromDate", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("minute"), std::make_shared<AnyMatcher>()}, "ExtractMinuteFromDate");
   patterns.insert<ReplaceFnWithFn>(ctxt, "ExtractFromDate", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("second"), std::make_shared<AnyMatcher>()}, "ExtractSecondFromDate");

   patterns.insert<ReplaceFnWithFn>(ctxt, "DateDiff", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("second"), std::make_shared<AnyMatcher>(), std::make_shared<AnyMatcher>()}, "DateDiffSecond");
   patterns.insert<ReplaceFnWithFn>(ctxt, "DateDiff", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("minute"), std::make_shared<AnyMatcher>(), std::make_shared<AnyMatcher>()}, "DateDiffMinute");
   patterns.insert<ReplaceFnWithFn>(ctxt, "DateDiff", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("hour"), std::make_shared<AnyMatcher>(), std::make_shared<AnyMatcher>()}, "DateDiffHour");
   patterns.insert<ReplaceFnWithFn>(ctxt, "DateDiff", std::vector<std::shared_ptr<Matcher>>{std::make_shared<StringConstMatcher>("day"), std::make_shared<AnyMatcher>(), std::make_shared<AnyMatcher>()}, "DateDiffDay");

   patterns.insert<ReplaceFnWithFn>(ctxt, "Like", std::vector<std::shared_ptr<Matcher>>{std::make_shared<AnyMatcher>(), std::make_shared<ConstantDeterministicPatternMatcher>()}, "ConstLike");
   patterns.insert<SimplifyRepeatedStringConcat>(ctxt);
}

} // end namespace lingodb::compiler::dialect::db
namespace {
class OptimizeRuntimeFunctions : public mlir::PassWrapper<OptimizeRuntimeFunctions, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "db-optimize-runtime-functions"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeRuntimeFunctions)
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         lingodb::compiler::dialect::db::addOptimizeRuntimeFunctionPatterns(patterns);
         if (lingodb::compiler::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::db::createOptimizeRuntimeFunctionsPass() { return std::make_unique<OptimizeRuntimeFunctions>(); } // NOLINT(misc-use-internal-linkage)
