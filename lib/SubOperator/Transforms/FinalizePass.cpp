#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"

#include "mlir/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"

#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <queue>
namespace {

class FinalizePass : public mlir::PassWrapper<FinalizePass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizePass)
   virtual llvm::StringRef getArgument() const override { return "subop-finalize"; }
   void cloneRec(mlir::Operation* op, mlir::IRMapping mapping,mlir::Value val, mlir::subop::ColumnMapping columnMapping) {


      mlir::OpBuilder builder(op->getContext());
      builder.setInsertionPointAfter(mapping.lookup(val).getDefiningOp()?mapping.lookup(val).getDefiningOp():op);
      mlir::cast<mlir::subop::SubOperator>(op).cloneSubOp(builder, mapping, columnMapping);
      for (auto& use : op->getUses()) {
         cloneRec(use.getOwner(), mapping,use.get(), columnMapping);
      }
   }
   void runOnOperation() override {
      auto module = getOperation();
      std::vector<mlir::subop::UnionOp> unionOps;
      module->walk([&](mlir::subop::UnionOp unionOp) {
         unionOps.push_back(unionOp);
      });
      for (size_t i = 0; i < unionOps.size(); i++) {
         auto currentUnion = unionOps[unionOps.size() - 1 - i];
         std::vector<mlir::Value> operands(currentUnion.getOperands().begin(), currentUnion.getOperands().end());
         std::sort(operands.begin(), operands.end(), [&](mlir::Value a, mlir::Value b) {
            return a.getDefiningOp()&&b.getDefiningOp()&&a.getDefiningOp()->getBlock() == b.getDefiningOp()->getBlock() && a.getDefiningOp()->isBeforeInBlock(b.getDefiningOp());
         });
         for (size_t i=0;i+1<operands.size();i++) {
            mlir::IRMapping mapping;
            mapping.map(currentUnion.getResult(), operands[i]);
            for (auto* user : currentUnion.getResult().getUsers()) {
               cloneRec(user, mapping,currentUnion.getResult(),{});
            }
         }
         currentUnion->replaceAllUsesWith(mlir::ValueRange{operands[operands.size()-1]});
         currentUnion->erase();
      }

      std::vector<mlir::subop::GenerateOp> generateOps;
      module->walk([&](mlir::subop::GenerateOp generateOp) {
         generateOps.push_back(generateOp);
      });
      for (auto generateOp : generateOps) {
         mlir::OpBuilder builder(generateOp);
         size_t emitOps = 0;
         generateOp.getRegion().walk([&](mlir::subop::GenerateEmitOp emitOp) {
            emitOps++;
         });
         if (emitOps == 0) {
            generateOp.emitError("GenerateOp must contain at least one GenerateEmitOp");
            return signalPassFailure();
         }
         if (emitOps == 1) {
            generateOp.getRes().replaceAllUsesWith(generateOp.getStreams()[0]);
            continue;
         }
         auto& memberManager = getContext().getLoadedDialect<mlir::subop::SubOperatorDialect>()->getMemberManager();
         auto& colManager = getContext().getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
         auto loc = generateOp.getLoc();
         std::vector<mlir::Attribute> types;
         std::vector<mlir::Attribute> names;
         std::vector<mlir::NamedAttribute> defMapping;
         std::vector<mlir::NamedAttribute> refMapping;

         for (auto m : generateOp.getGeneratedColumns()) {
            auto* column = &m.cast<mlir::tuples::ColumnDefAttr>().getColumn();
            auto name = memberManager.getUniqueMember("tmp_union");
            types.push_back(mlir::TypeAttr::get(column->type));
            names.push_back(builder.getStringAttr(name));
            defMapping.push_back(builder.getNamedAttr(name, m));
            refMapping.push_back(builder.getNamedAttr(name, colManager.createRef(&m.cast<mlir::tuples::ColumnDefAttr>().getColumn())));
         }
         mlir::Value tmpBuffer;

         auto bufferType = mlir::subop::BufferType::get(builder.getContext(), mlir::subop::StateMembersAttr::get(builder.getContext(), builder.getArrayAttr(names), builder.getArrayAttr(types)));
         tmpBuffer = builder.create<mlir::subop::GenericCreateOp>(loc, bufferType);

         builder.setInsertionPointAfter(generateOp);
         for (auto stream : generateOp.getStreams()) {
            builder.create<mlir::subop::MaterializeOp>(loc, stream, tmpBuffer, builder.getDictionaryAttr(refMapping));
         }
         auto scanRefDef = colManager.createDef(colManager.getUniqueScope("tmp_union"), "scan_ref");
         scanRefDef.getColumn().type = mlir::subop::EntryRefType::get(builder.getContext(), tmpBuffer.getType().cast<mlir::subop::State>());
         auto scan = builder.create<mlir::subop::ScanRefsOp>(loc, tmpBuffer, scanRefDef);
         mlir::Value loaded = builder.create<mlir::subop::GatherOp>(loc, scan, colManager.createRef(&scanRefDef.getColumn()), builder.getDictionaryAttr(defMapping));
         generateOp.getRes().replaceAllUsesWith(loaded);
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
mlir::subop::createFinalizePass() { return std::make_unique<FinalizePass>(); }