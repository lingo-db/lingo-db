#include "llvm/Support/Debug.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnCreationAnalysis.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"

#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/SubOpDependencyAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace {
using namespace lingodb::compiler::dialect;

static std::pair<tuples::ColumnDefAttr, tuples::ColumnRefAttr> createColumn(mlir::Type type, std::string scope, std::string name) {
   auto& columnManager = type.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   std::string scopeName = columnManager.getUniqueScope(scope);
   std::string attributeName = name;
   tuples::ColumnDefAttr markAttrDef = columnManager.createDef(scopeName, attributeName);
   auto& ra = markAttrDef.getColumn();
   ra.type = type;
   return {markAttrDef, columnManager.createRef(&ra)};
}
class SplitTableScan : public mlir::RewritePattern {
   public:
   SplitTableScan(mlir::MLIRContext* context)
      : RewritePattern(subop::ScanOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto scanOp = mlir::cast<subop::ScanOp>(op);
      if (mlir::isa<subop::TableType>(scanOp.getState().getType())) {
         auto tableType = mlir::cast<subop::TableType>(scanOp.getState().getType());
         std::vector<mlir::Attribute> memberNames;
         std::vector<mlir::Attribute> memberTypes;
         auto tableMembers = tableType.getMembers();
         for (auto i = 0ul; i < tableMembers.getTypes().size(); i++) {
            auto type = mlir::cast<mlir::TypeAttr>(tableMembers.getTypes()[i]);
            auto name = mlir::cast<mlir::StringAttr>(tableMembers.getNames()[i]);
            if (scanOp.getMapping().contains(name)) {
               memberNames.push_back(name);
               memberTypes.push_back(type);
            }
         }
         auto members = subop::StateMembersAttr::get(getContext(), mlir::ArrayAttr::get(getContext(), memberNames), mlir::ArrayAttr::get(getContext(), memberTypes));

         auto [refDef, refRef] = createColumn(subop::TableEntryRefType::get(getContext(), members), "scan", "ref");
         mlir::Value scanRefsOp = rewriter.create<subop::ScanRefsOp>(op->getLoc(), scanOp.getState(), refDef);
         rewriter.replaceOpWithNewOp<subop::GatherOp>(op, scanRefsOp, refRef, scanOp.getMapping());
         return mlir::success();
      }else if(mlir::isa<subop::LocalTableType>(scanOp.getState().getType())) {
         auto tableType = mlir::cast<subop::LocalTableType>(scanOp.getState().getType());
         std::vector<mlir::Attribute> memberNames;
         std::vector<mlir::Attribute> memberTypes;
         auto tableMembers = tableType.getMembers();
         for (auto i = 0ul; i < tableMembers.getTypes().size(); i++) {
            auto type = mlir::cast<mlir::TypeAttr>(tableMembers.getTypes()[i]);
            auto name = mlir::cast<mlir::StringAttr>(tableMembers.getNames()[i]);
            if (scanOp.getMapping().contains(name)) {
               memberNames.push_back(name);
               memberTypes.push_back(type);
            }
         }
         auto members = subop::StateMembersAttr::get(getContext(), mlir::ArrayAttr::get(getContext(), memberNames), mlir::ArrayAttr::get(getContext(), memberTypes));

         auto [refDef, refRef] = createColumn(subop::TableEntryRefType::get(getContext(), members), "scan", "ref");
         mlir::Value scanRefsOp = rewriter.create<subop::ScanRefsOp>(op->getLoc(), scanOp.getState(), refDef);
         rewriter.replaceOpWithNewOp<subop::GatherOp>(op, scanRefsOp, refRef, scanOp.getMapping());
         return mlir::success();
      }
      return mlir::failure();
   }
};
class SplitGenericScan : public mlir::RewritePattern {
   public:
   SplitGenericScan(mlir::MLIRContext* context)
      : RewritePattern(subop::ScanOp::getOperationName(), 1, context) {}
   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto scanOp = mlir::cast<subop::ScanOp>(op);
      if (mlir::isa<subop::TableType>(scanOp.getState().getType())) return mlir::failure();
      //todo: check that one can obtain references
      mlir::Type refType = subop::EntryRefType::get(getContext(), mlir::cast<subop::State>(scanOp.getState().getType()));
      if (auto continuousView = mlir::dyn_cast_or_null<subop::ContinuousViewType>(scanOp.getState().getType())) {
         refType = subop::ContinuousEntryRefType::get(rewriter.getContext(), continuousView);
      }
      if (auto array = mlir::dyn_cast_or_null<subop::ArrayType>(scanOp.getState().getType())) {
         refType = subop::ContinuousEntryRefType::get(rewriter.getContext(), array);
      }
      if (auto hashMapType = mlir::dyn_cast_or_null<subop::HashMapType>(scanOp.getState().getType())) {
         refType = subop::HashMapEntryRefType::get(rewriter.getContext(), hashMapType);
      }
      if (auto hashMultiMapType = mlir::dyn_cast_or_null<subop::HashMultiMapType>(scanOp.getState().getType())) {
         refType = subop::HashMultiMapEntryRefType::get(rewriter.getContext(), hashMultiMapType);
      }

      auto [refDef, refRef] = createColumn(refType, "scan", "ref");
      mlir::Value scanRefsOp = rewriter.create<subop::ScanRefsOp>(op->getLoc(), scanOp.getState(), refDef);
      if (scanOp->hasAttr("sequential")) {
         scanRefsOp.getDefiningOp()->setAttr("sequential", rewriter.getUnitAttr());
      }
      rewriter.replaceOpWithNewOp<subop::GatherOp>(op, scanRefsOp, refRef, scanOp.getMapping());

      return mlir::success();
   }
};
class NormalizeSubOpPass : public mlir::PassWrapper<NormalizeSubOpPass, mlir::OperationPass<mlir::ModuleOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeSubOpPass)
   virtual llvm::StringRef getArgument() const override { return "subop-normalize"; }
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
   void runOnOperation() override {
      std::unordered_map<mlir::Operation*, size_t> unionStreamCount;
      auto columnUsageAnalysis = getAnalysis<subop::ColumnUsageAnalysis>();
      auto columnCreationAnalysis = getAnalysis<subop::ColumnCreationAnalysis>();

      getOperation()->walk([&](mlir::Operation* op) {
         if (auto unionOp = mlir::dyn_cast_or_null<subop::UnionOp>(op)) {
            size_t streams = 0;
            for (auto stream : unionOp.getStreams()) {
               if (auto* def = stream.getDefiningOp()) {
                  if (unionStreamCount.contains(def)) {
                     streams += unionStreamCount[def];
                  } else {
                     streams += 1;
                  }
               }
            }
            size_t numUsers = 0;
            size_t numEndUsers = 0;
            for (auto* user : unionOp.getRes().getUsers()) {
               numUsers++;
               bool tupleStreamContinues = false;
               for (auto userResultType : user->getResultTypes()) {
                  tupleStreamContinues |= mlir::isa<tuples::TupleStreamType>(userResultType);
               }
               if (!tupleStreamContinues) {
                  numEndUsers++;
               }
            }
            //return false;
            if (numUsers == 1 && numEndUsers == 1) {
               //do not materialize
               unionStreamCount[op] = streams;
            } else if (streams <= 3) {
               //do not materialize
               unionStreamCount[op] = streams;
            } else {
               std::unordered_set<tuples::Column*> usedColumns;
               std::unordered_set<tuples::Column*> requiredColumns;
               getRecursivelyUsedColumns(usedColumns, columnUsageAnalysis, unionOp);
               getRequired(requiredColumns, usedColumns, columnCreationAnalysis, unionOp);
               mlir::OpBuilder builder(&getContext());

               auto& memberManager = builder.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
               auto& colManager = builder.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

               std::vector<mlir::Attribute> types;
               std::vector<mlir::Attribute> names;
               std::vector<mlir::NamedAttribute> defMapping;
               std::vector<mlir::NamedAttribute> refMapping;
               for (auto* column : requiredColumns) {
                  auto name = memberManager.getUniqueMember("tmp_union");
                  types.push_back(mlir::TypeAttr::get(column->type));
                  names.push_back(builder.getStringAttr(name));
                  defMapping.push_back(builder.getNamedAttr(name, colManager.createDef(column)));
                  refMapping.push_back(builder.getNamedAttr(name, colManager.createRef(column)));
               }
               builder.setInsertionPointToStart(unionOp->getBlock());
               auto bufferType = subop::BufferType::get(builder.getContext(), subop::StateMembersAttr::get(builder.getContext(), builder.getArrayAttr(names), builder.getArrayAttr(types)));
               mlir::Value tmpBuffer = builder.create<subop::GenericCreateOp>(unionOp->getLoc(), bufferType);
               builder.setInsertionPoint(unionOp);
               for (auto stream : unionOp.getStreams()) {
                  builder.create<subop::MaterializeOp>(unionOp->getLoc(), stream, tmpBuffer, builder.getDictionaryAttr(refMapping));
               }
               auto scanRefDef = colManager.createDef(colManager.getUniqueScope("tmp_union"), "scan_ref");
               scanRefDef.getColumn().type = subop::EntryRefType::get(builder.getContext(), mlir::cast<subop::State>(tmpBuffer.getType()));
               auto scan = builder.create<subop::ScanRefsOp>(unionOp->getLoc(), tmpBuffer, scanRefDef);
               mlir::Value loaded = builder.create<subop::GatherOp>(unionOp->getLoc(), scan, colManager.createRef(&scanRefDef.getColumn()), builder.getDictionaryAttr(defMapping));
               unionOp.getRes().replaceAllUsesWith(loaded);
               unionOp.erase();
            }
         } else {
            size_t sum = 0;
            for (auto operand : op->getOperands()) {
               if (mlir::isa<tuples::TupleStreamType>(operand.getType())) {
                  if (auto* def = operand.getDefiningOp()) {
                     if (unionStreamCount.contains(def)) {
                        sum += unionStreamCount[def];
                     }
                  }
               }
            }
            if (auto nestedMapOp = mlir::dyn_cast_or_null<subop::NestedMapOp>(op)) {
               if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(nestedMapOp.getRegion().front().getTerminator())) {
                  for (auto operand : returnOp.getOperands()) {
                     if (auto* def = operand.getDefiningOp()) {
                        if (unionStreamCount.contains(def)) {
                           sum += unionStreamCount[def];
                        }
                     }
                  }
               }
            }
            if (sum > 0) {
               if (sum > 3) {
                  size_t numUsers = 0;
                  size_t numEndUsers = 0;
                  for (auto* user : op->getUsers()) {
                     numUsers++;
                     bool tupleStreamContinues = false;
                     for (auto userResultType : user->getResultTypes()) {
                        tupleStreamContinues |= mlir::isa<tuples::TupleStreamType>(userResultType);
                     }
                     if (!tupleStreamContinues) {
                        numEndUsers++;
                     }
                  }
                  if (numUsers == 0) {
                  } else if (numUsers == 1 && numEndUsers == 1) {
                  } else {
                     assert(op->getNumResults() == 1);
                     //materialize
                     std::unordered_set<tuples::Column*> usedColumns;
                     std::unordered_set<tuples::Column*> requiredColumns;
                     for (auto* user : op->getUsers()) {
                        getRecursivelyUsedColumns(usedColumns, columnUsageAnalysis, user);
                     }
                     getRequired(requiredColumns, usedColumns, columnCreationAnalysis, op);
                     mlir::OpBuilder builder(&getContext());

                     auto& memberManager = builder.getContext()->getLoadedDialect<subop::SubOperatorDialect>()->getMemberManager();
                     auto& colManager = builder.getContext()->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();

                     std::vector<mlir::Attribute> types;
                     std::vector<mlir::Attribute> names;
                     std::vector<mlir::NamedAttribute> defMapping;
                     std::vector<mlir::NamedAttribute> refMapping;
                     for (auto* column : requiredColumns) {
                        auto name = memberManager.getUniqueMember("tmp_union");
                        types.push_back(mlir::TypeAttr::get(column->type));
                        names.push_back(builder.getStringAttr(name));
                        defMapping.push_back(builder.getNamedAttr(name, colManager.createDef(column)));
                        refMapping.push_back(builder.getNamedAttr(name, colManager.createRef(column)));
                     }
                     builder.setInsertionPointToStart(op->getBlock());
                     auto bufferType = subop::BufferType::get(builder.getContext(), subop::StateMembersAttr::get(builder.getContext(), builder.getArrayAttr(names), builder.getArrayAttr(types)));
                     mlir::Value tmpBuffer = builder.create<subop::GenericCreateOp>(op->getLoc(), bufferType);
                     builder.setInsertionPointAfter(op);
                     auto materializeOp = builder.create<subop::MaterializeOp>(op->getLoc(), op->getResult(0), tmpBuffer, builder.getDictionaryAttr(refMapping));
                     auto scanRefDef = colManager.createDef(colManager.getUniqueScope("tmp_union"), "scan_ref");
                     scanRefDef.getColumn().type = subop::EntryRefType::get(builder.getContext(), mlir::cast<subop::State>(tmpBuffer.getType()));
                     auto scan = builder.create<subop::ScanRefsOp>(op->getLoc(), tmpBuffer, scanRefDef);
                     mlir::Value loaded = builder.create<subop::GatherOp>(op->getLoc(), scan, colManager.createRef(&scanRefDef.getColumn()), builder.getDictionaryAttr(defMapping));
                     op->getResult(0).replaceAllUsesExcept(loaded, materializeOp);
                  }
               } else {
                  unionStreamCount[op] = sum;
               }
            }
         }
      });
      //transform "standalone" aggregation functions

      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<SplitTableScan>(&getContext());
      patterns.insert<SplitGenericScan>(&getContext());

      if (mlir::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         assert(false && "should not happen");
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
subop::createNormalizeSubOpPass() { return std::make_unique<NormalizeSubOpPass>(); }