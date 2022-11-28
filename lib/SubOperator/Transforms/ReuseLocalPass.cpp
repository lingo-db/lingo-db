#include "llvm/Support/Debug.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_set>
namespace {
static std::vector<mlir::subop::SubOperator> findWritingOps(mlir::Block& block, std::vector<std::string>& members) {
   std::unordered_set<std::string> memberSet(members.begin(), members.end());
   std::vector<mlir::subop::SubOperator> res;
   block.walk([&memberSet, &res](mlir::subop::SubOperator subop) {
      auto writtenBySubOp = subop.getWrittenMembers();
      for (auto writtenMember : writtenBySubOp) {
         if (memberSet.contains(writtenMember)) {
            res.push_back(subop);
            break;
         }
      }
   });
   return res;
}

static std::string lookupByValue(mlir::DictionaryAttr mapping, mlir::Attribute value) {
   for (auto m : mapping) {
      if (m.getValue() == value) {
         return m.getName().str();
      }
   }
   assert(false && "should not happen");
   return "";
}

class AvoidUnnecessaryMaterialization : public mlir::RewritePattern {
   public:
   AvoidUnnecessaryMaterialization(mlir::MLIRContext* context)
      : RewritePattern(mlir::subop::ScanOp::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto& colManager = rewriter.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto scanOp = mlir::cast<mlir::subop::ScanOp>(op);
      auto readMembers = scanOp.getReadMembers();
      auto* stateCreateOp = scanOp.getState().getDefiningOp();
      if (!stateCreateOp) {
         return mlir::failure();
      }
      auto writingOps = findWritingOps(*stateCreateOp->getBlock(), readMembers);
      if (writingOps.size() != 1) {
         return mlir::failure();
      }
      if (auto materializeOp = mlir::dyn_cast_or_null<mlir::subop::MaterializeOp>(writingOps[0].getOperation())) {
         if (materializeOp->getBlock() == scanOp->getBlock() && materializeOp->isBeforeInBlock(scanOp)) {
            if (auto scanOp2 = mlir::dyn_cast_or_null<mlir::subop::ScanOp>(materializeOp.getStream().getDefiningOp())) {
               std::vector<mlir::NamedAttribute> newMapping;
               for (auto curr : scanOp.getMapping()) {
                  auto currentMember = curr.getName();
                  auto otherColumnDef = colManager.createDef(&materializeOp.getMapping().get(currentMember).cast<mlir::tuples::ColumnRefAttr>().getColumn());
                  auto otherMember = lookupByValue(scanOp2.getMapping(), otherColumnDef);
                  newMapping.push_back(rewriter.getNamedAttr(otherMember, curr.getValue()));
               }
               rewriter.updateRootInPlace(op, [&] {
                  scanOp.setOperand(scanOp2.getState());
                  scanOp.setMappingAttr(rewriter.getDictionaryAttr(newMapping));
               });
               return mlir::success();
            }
         }
      }
      return mlir::failure();
   }
};

class AvoidDeadMaterialization : public mlir::RewritePattern {
   public:
   AvoidDeadMaterialization(mlir::MLIRContext* context)
      : RewritePattern(mlir::subop::MaterializeOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto materializeOp = mlir::cast<mlir::subop::MaterializeOp>(op);
      for (auto* user : materializeOp.getState().getUsers()) {
         if (user != op) {
            //other user of state => "probably" still useful
            return mlir::failure();
         }
      }
      rewriter.eraseOp(op);
      return mlir::success();
   }
};
class ReuseLocalPass : public mlir::PassWrapper<ReuseLocalPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReuseLocalPass)
   virtual llvm::StringRef getArgument() const override { return "subop-reuse-local"; }

   void runOnOperation() override {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<AvoidUnnecessaryMaterialization>(&getContext());
      patterns.insert<AvoidDeadMaterialization>(&getContext());
      if (mlir::applyPatternsAndFoldGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::subop::createReuseLocalPass() { return std::make_unique<ReuseLocalPass>(); }