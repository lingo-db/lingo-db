#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include <string>

namespace {
using namespace lingodb::compiler::dialect;

class IntroduceTmp : public mlir::PassWrapper<IntroduceTmp, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-introduce-tmp"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntroduceTmp)

   relalg::ColumnSet getUsed(mlir::Operation* op) {
      if (auto asOperator = mlir::dyn_cast_or_null<Operator>(op)) {
         auto cols = asOperator.getUsedColumns();
         for (auto* user : asOperator.asRelation().getUsers()) {
            cols.insert(getUsed(user));
         }
         return cols;
      } else if (auto matOp = mlir::dyn_cast_or_null<relalg::MaterializeOp>(op)) {
         return relalg::ColumnSet::fromArrayAttr(matOp.getCols());
      } else if (auto subOpMatOp = mlir::dyn_cast_or_null<subop::MaterializeOp>(op)) {
         // Special case for subop.materialize: extract columns from the mapping pairs
         relalg::ColumnSet cols;
         for (auto x : subOpMatOp.getMapping().getMapping()) {
            cols.insert(&x.second.getColumn());
         }
         return cols;
      }
      return {};
   }

   void runOnOperation() override {
      relalg::AvailabilityCache availabilityCache;
      getOperation().walk([&](Operator op) {
         if (op->getParentOfType<Operator>() || op->use_empty()) return;

         // STEP 1: Detect and handle uses inside a dynamic loop
         llvm::SmallVector<mlir::OpOperand*> loopUses;
         for (auto& use : op->getUses()) {
            auto loopParent = use.getOwner()->getParentOfType<subop::LoopOp>();
            if (loopParent && !loopParent->isAncestor(op)) {
               loopUses.push_back(&use);
            }
         }

         if (!loopUses.empty()) {
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPointAfter(op.getOperation());

            // Gather columns needed inside the loop
            relalg::ColumnSet usedCols;
            for (auto* use : loopUses) {
               usedCols.insert(getUsed(use->getOwner()));
            }
            usedCols = usedCols.intersect(op.getAvailableColumns(availabilityCache));

            auto& memberManager = getContext().getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
            auto& colManager = getContext().getOrLoadDialect<tuples::TupleStreamDialect>()->getColumnManager();

            // FIXED: Using llvm::SmallVector to satisfy SubOperatorOpsAttributes.h.inc
            llvm::SmallVector<subop::Member> members;
            llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> mappingArgs;
            llvm::SmallVector<mlir::Attribute> scanCols;
            llvm::SmallVector<mlir::Attribute> scanMapping;

            size_t loopColIdx = 0;
            for (auto* col : usedCols) {
               // FIXED: Generate a safe, unique member name instead of accessing col->name
               std::string memberName = "loop_carried_col_" + std::to_string(loopColIdx++);
               subop::Member member = memberManager.createMember(memberName, col->type);

               members.push_back(member);
               mappingArgs.push_back({member, colManager.createRef(col)});
               scanCols.push_back(colManager.createDef(col));
               scanMapping.push_back(builder.getStringAttr(memberManager.getName(member)));
            }

            // Materialize into physical memory outside the loop
            auto bufferType = subop::BufferType::get(&getContext(), subop::StateMembersAttr::get(&getContext(), members));
            auto createBufOp = builder.create<subop::GenericCreateOp>(op->getLoc(), bufferType);
            auto mappingAttr = subop::ColumnRefMemberMappingAttr::get(&getContext(), mappingArgs);
            builder.create<subop::MaterializeOp>(op->getLoc(), op.asRelation(), createBufOp.getResult(), mappingAttr);

            // Buffer scan exactly at the use site inside the loop
            for (auto* use : loopUses) {
               builder.setInsertionPoint(use->getOwner());
               auto scanOp = builder.create<relalg::BufferScanOp>(
                  op->getLoc(), tuples::TupleStreamType::get(&getContext()),
                  createBufOp.getResult(),
                  builder.getArrayAttr(scanCols),
                  builder.getArrayAttr(scanMapping));
               scanOp->setAttr("rows", builder.getF64FloatAttr(100.0));
               use->set(scanOp.getResult());
            }
         }

         // STEP 2: Handle remaining static multiple uses via standard relalg.tmp
         if (!op->hasOneUse() && !op->use_empty()) {
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPointAfter(op.getOperation());

            relalg::ColumnSet usedAttributes;
            for (auto& use : op->getUses()) {
               usedAttributes.insert(getUsed(use.getOwner()));
            }
            usedAttributes = usedAttributes.intersect(op.getAvailableColumns(availabilityCache));

            mlir::Type tupleStreamType = op.asRelation().getType();
            llvm::SmallVector<mlir::Type> resultingTypes;
            for (auto it = op->getUses().begin(); it != op->getUses().end(); it++) {
               resultingTypes.push_back(tupleStreamType);
            }

            auto tmp = builder.create<relalg::TmpOp>(op->getLoc(), resultingTypes, op.asRelation(), usedAttributes.asRefArrayAttr(&getContext()));
            size_t i = 0;
            for (auto& use : llvm::make_early_inc_range(op->getUses())) {
               if (use.getOwner() != tmp) {
                  use.set(tmp.getResult(i++));
               }
            }
         }
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createIntroduceTmpPass() { return std::make_unique<IntroduceTmp>(); }