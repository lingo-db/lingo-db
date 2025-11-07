#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "llvm/Support/Debug.h"

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"

#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <queue>
namespace {
using namespace lingodb::compiler::dialect;

class FinalizePass : public mlir::PassWrapper<FinalizePass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizePass)
   virtual llvm::StringRef getArgument() const override { return "subop-finalize"; }
   void cloneRec(mlir::Operation* op, mlir::IRMapping mapping, mlir::Value val, subop::ColumnMapping columnMapping) {
      mlir::OpBuilder builder(op->getContext());
      builder.setInsertionPointAfter(mapping.lookup(val).getDefiningOp() ? mapping.lookup(val).getDefiningOp() : op);
      mlir::cast<subop::SubOperator>(op).cloneSubOp(builder, mapping, columnMapping);
      for (auto& use : op->getUses()) {
         cloneRec(use.getOwner(), mapping, use.get(), columnMapping);
      }
   }
   void getRecursivelyUsedColumns(std::unordered_set<tuples::Column*>& usedColumns, subop::ColumnUsageAnalysis& columnUsageAnalysis, mlir::Operation* op) {
      const auto& currentUsed = columnUsageAnalysis.getUsedColumns(op);
      usedColumns.insert(currentUsed.begin(), currentUsed.end());
      for (auto res : op->getResults()) {
         if (mlir::isa<tuples::TupleStreamType>(res.getType())) {
            for (auto* user : res.getUsers()) {
               getRecursivelyUsedColumns(usedColumns, columnUsageAnalysis, user);
            }
         }
      }
   }
   void getRequired(std::unordered_set<tuples::Column*>& requiredColumns, const std::unordered_set<tuples::Column*>& usedColumns, subop::ColumnCreationAnalysis& columnCreationAnalysis, mlir::Operation* op) {
      const auto& createdCols = columnCreationAnalysis.getCreatedColumns(op);
      for (auto* c : createdCols) {
         if (usedColumns.contains(c)) {
            requiredColumns.insert(c);
         }
      }
      for (auto operand : op->getOperands()) {
         if (mlir::isa<tuples::TupleStreamType>(operand.getType())) {
            if (auto* defOp = operand.getDefiningOp()) {
               getRequired(requiredColumns, usedColumns, columnCreationAnalysis, defOp);
            }
         }
      }
   }
   void moveRecursivelyIntoMacro(mlir::Operation* op, mlir::Block*& macroBlock, std::vector<mlir::Value>& args, mlir::Value tupleStream) {
      if (auto subOp = mlir::dyn_cast_or_null<subop::SubOperator>(op)) {
         subOp->remove();
         mlir::OpBuilder builder(op->getContext());
         builder.setInsertionPointToEnd(macroBlock);
         builder.insert(subOp);
         for (auto& r : subOp->getRegions()) {
            r.walk([&](mlir::Operation* op) {
               for (auto& arg : op->getOpOperands()) {
                  bool outside = false;
                  //check if arg is defined outside of r
                  if (auto* defOp = arg.get().getDefiningOp()) {
                     if (!subOp->isAncestor(defOp)) {
                        outside = true;
                     }
                  } else if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(arg.get())) {
                     if (!subOp->isAncestor(blockArg.getOwner()->getParentOp())) {
                        outside = true;
                     }
                  }
                  if (outside) {
                     assert(!mlir::isa<tuples::TupleStreamType>(arg.get().getType()) && "tuple-streams should be handled separately");
                     auto type = arg.get().getType();
                     args.push_back(arg.get());
                     arg.set(macroBlock->addArgument(type, subOp->getLoc()));
                  }
               }
            });
         }
         for (auto& arg : subOp->getOpOperands()) {
            auto type = arg.get().getType();
            if (!mlir::isa<tuples::TupleStreamType>(type)) {
               args.push_back(arg.get());
               arg.set(macroBlock->addArgument(type, subOp->getLoc()));
            } else {
               arg.set(tupleStream);
            }
         }
         for (auto* user : subOp->getUsers()) {
            moveRecursivelyIntoMacro(user, macroBlock, args, subOp->getResult(0));
         }
      } else {
         assert(false);
      }
   }
   void runOnOperation() override {
      size_t macroOpId = 0;
      auto module = getOperation();
      auto columnUsageAnalysis = getAnalysis<subop::ColumnUsageAnalysis>();
      auto columnCreationAnalysis = getAnalysis<subop::ColumnCreationAnalysis>();
      auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      std::vector<subop::UnionOp> unionForMacros;
      module->walk([&](subop::UnionOp unionOp) {
         if (mlir::isa<subop::ExecutionGroupOp>(unionOp->getParentOp())) {
            unionForMacros.push_back(unionOp);
         }
      });
      for (size_t i = 0; i < unionForMacros.size(); i++) {
         auto unionOp = unionForMacros[unionForMacros.size() - 1 - i];
         std::unordered_set<tuples::Column*> usedColumns;
         std::unordered_set<tuples::Column*> requiredColumns;
         getRecursivelyUsedColumns(usedColumns, columnUsageAnalysis, unionOp);
         getRequired(requiredColumns, usedColumns, columnCreationAnalysis, unionOp);
         std::vector<mlir::Attribute> requiredColumnRefs;
         for (auto* c : requiredColumns) {
            requiredColumnRefs.push_back(colManager.createRef(c));
         }
         mlir::ArrayAttr requiredColumnsAttr = mlir::ArrayAttr::get(&getContext(), requiredColumnRefs);
         mlir::OpBuilder builder(unionOp);
         std::vector<mlir::Value> args;

         mlir::Block* macroBlock = new mlir::Block;
         auto tupleStreamArg = macroBlock->addArgument(tuples::TupleStreamType::get(&getContext()), unionOp.getLoc());
         for (auto* user : unionOp->getUsers()) {
            moveRecursivelyIntoMacro(user, macroBlock, args, tupleStreamArg);
         }
         {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToEnd(macroBlock);
            builder.create<subop::MacroReturnOp>(unionOp.getLoc());
         }
         auto macroName = "macro" + std::to_string(macroOpId++);
         for (auto stream : unionOp.getStreams()) {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfter(stream.getDefiningOp());
            auto newStream = builder.create<subop::MacroCallOp>(unionOp.getLoc(), macroName, stream, args, requiredColumnsAttr);
         }
         auto executionGroupOp = unionOp->getParentOfType<subop::ExecutionGroupOp>();
         builder.setInsertionPointToStart(&executionGroupOp.getSubOps().front());
         auto macroOp = builder.create<subop::Macro>(unionOp.getLoc(), macroName, requiredColumnsAttr);
         macroOp.getRegion().getBlocks().clear();
         macroOp.getRegion().getBlocks().push_back(macroBlock);
         unionOp->erase();
      }
      std::vector<subop::UnionOp> unionOps;
      module->walk([&](subop::UnionOp unionOp) {
         unionOps.push_back(unionOp);
      });
      for (size_t i = 0; i < unionOps.size(); i++) {
         auto currentUnion = unionOps[unionOps.size() - 1 - i];
         std::vector<mlir::Value> operands(currentUnion.getOperands().begin(), currentUnion.getOperands().end());
         std::sort(operands.begin(), operands.end(), [&](mlir::Value a, mlir::Value b) {
            return a.getDefiningOp() && b.getDefiningOp() && a.getDefiningOp()->getBlock() == b.getDefiningOp()->getBlock() && a.getDefiningOp()->isBeforeInBlock(b.getDefiningOp());
         });
         for (size_t i = 0; i + 1 < operands.size(); i++) {
            mlir::IRMapping mapping;
            mapping.map(currentUnion.getResult(), operands[i]);
            for (auto* user : currentUnion.getResult().getUsers()) {
               cloneRec(user, mapping, currentUnion.getResult(), {});
            }
         }
         currentUnion->replaceAllUsesWith(mlir::ValueRange{operands[operands.size() - 1]});
         currentUnion->erase();
      }


      std::vector<subop::GenerateOp> generateOps;
      module->walk([&](subop::GenerateOp generateOp) {
         generateOps.push_back(generateOp);
      });
      for (auto generateOp : generateOps) {
         mlir::OpBuilder builder(generateOp);
         size_t emitOps = 0;
         generateOp.getRegion().walk([&](subop::GenerateEmitOp emitOp) {
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
         auto& memberManager = getContext().getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
         auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
         auto loc = generateOp.getLoc();
         llvm::SmallVector<subop::Member> members;
         llvm::SmallVector<subop::DefMappingPairT> defMapping;
         llvm::SmallVector<subop::RefMappingPairT> refMapping;

         for (auto m : generateOp.getGeneratedColumns()) {
            auto* column = &mlir::cast<tuples::ColumnDefAttr>(m).getColumn();
            auto member = memberManager.createMember("tmp_union", column->type);
            members.push_back(member);
            auto colDef = mlir::cast<tuples::ColumnDefAttr>(m);
            defMapping.push_back({member, colDef});
            refMapping.push_back({member, colManager.createRef(&colDef.getColumn())});
         }
         mlir::Value tmpBuffer;

         auto bufferType = subop::BufferType::get(builder.getContext(), subop::StateMembersAttr::get(builder.getContext(), members));
         tmpBuffer = builder.create<subop::GenericCreateOp>(loc, bufferType);

         builder.setInsertionPointAfter(generateOp);
         for (auto stream : generateOp.getStreams()) {
            builder.create<subop::MaterializeOp>(loc, stream, tmpBuffer, subop::ColumnRefMemberMappingAttr::get(builder.getContext(), refMapping));
         }
         auto scanRefDef = colManager.createDef(colManager.getUniqueScope("tmp_union"), "scan_ref");
         scanRefDef.getColumn().type = subop::EntryRefType::get(builder.getContext(), mlir::cast<subop::State>(tmpBuffer.getType()));
         auto scan = builder.create<subop::ScanRefsOp>(loc, tmpBuffer, scanRefDef);
         mlir::Value loaded = builder.create<subop::GatherOp>(loc, scan, colManager.createRef(&scanRefDef.getColumn()),
                                                              subop::ColumnDefMemberMappingAttr::get(builder.getContext(), defMapping));
         generateOp.getRes().replaceAllUsesWith(loaded);
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createFinalizePass() { return std::make_unique<FinalizePass>(); }