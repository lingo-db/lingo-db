#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_set>
namespace {
using namespace lingodb::compiler::dialect;

class EliminateNullableTypes : public mlir::PassWrapper<EliminateNullableTypes, mlir::OperationPass<mlir::func::FuncOp>> {
   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateNullableTypes)
   virtual llvm::StringRef getArgument() const override { return "relalg-eliminate-nullable-types"; }
   static relalg::ColumnSet getRequired(Operator op, llvm::DenseMap<Operator, relalg::ColumnSet>& cache) {
      if (cache.contains(op)) {
         return cache[op];
      }
      relalg::ColumnSet required;
      for (auto* user : op->getUsers()) {
         if (auto consumingOp = mlir::dyn_cast_or_null<Operator>(user)) {
            required.insert(getRequired(consumingOp, cache));
            required.insert(consumingOp.getUsedColumns());
         }
         if (auto materializeOp = mlir::dyn_cast_or_null<relalg::MaterializeOp>(user)) {
            required.insert(relalg::ColumnSet::fromArrayAttr(materializeOp.getCols()));
         }
      }
      cache[op] = required;
      return required;
   }
   void materialize(mlir::Operation* op, size_t idx, lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& info) {
      llvm::DenseMap<Operator, relalg::ColumnSet> cache;
      auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      mlir::OpBuilder builder(op->getContext());
      builder.setInsertionPoint(op);
      relalg::ColumnSet required;
      if (auto materializeOp = mlir::dyn_cast_or_null<relalg::MaterializeOp>(op)) {
         required = relalg::ColumnSet::fromArrayAttr(materializeOp.getCols());
      } else {
         required = getRequired(mlir::cast<Operator>(op), cache);
         required.insert(mlir::cast<Operator>(op).getUsedColumns());
      }

      std::vector<mlir::Attribute>
         newColDefs;
      auto* block = new mlir::Block;
      auto loc = builder.getUnknownLoc();
      auto tuple = block->addArgument(tuples::TupleType::get(&getContext()), loc);
      {
         std::vector<mlir::Value> toReturn;
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(block);
         for (auto [nullable, nonnullable] : info.directMappings) {
            auto colRef = colManager.createRef(nonnullable);
            if (required.contains(nullable)) {
               mlir::Value val = builder.create<tuples::GetColumnOp>(loc, nonnullable->type, colRef, tuple);
               toReturn.push_back(builder.create<db::AsNullableOp>(loc, nullable->type, val));
               auto colDef = colManager.createDef(nullable);
               newColDefs.push_back(colDef);
            }
         }
         if (!newColDefs.empty()) {
            builder.create<tuples::ReturnOp>(loc, toReturn);
         }
      }
      if (newColDefs.empty()) {
         delete block;
         return;
      }
      auto mapOp = builder.create<relalg::MapOp>(op->getLoc(), tuples::TupleStreamType::get(&getContext()), op->getOperand(idx), builder.getArrayAttr(newColDefs));
      mapOp.getPredicate().push_back(block);
      op->setOperand(idx, mapOp.asRelation());
   }
   tuples::Column* handleNullCheck(lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo& info, relalg::SelectionOp selectionOp) {
      if (selectionOp.getPredicate().empty()) return nullptr;
      if (selectionOp.getPredicate().front().empty()) return nullptr;
      if (auto returnOp = mlir::dyn_cast_or_null<tuples::ReturnOp>(selectionOp.getPredicate().front().getTerminator())) {
         if (returnOp.getResults().size() != 1) return nullptr;
         if (auto notOp = mlir::dyn_cast_or_null<db::NotOp>(returnOp.getResults()[0].getDefiningOp())) {
            if (auto isNullOp = mlir::dyn_cast_or_null<db::IsNullOp>(notOp.getVal().getDefiningOp())) {
               if (auto getColOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(isNullOp.getVal().getDefiningOp())) {
                  if (!info.directMappings.contains(&getColOp.getAttr().getColumn())) {
                     return &getColOp.getAttr().getColumn();
                  }
               }
            }
         }
      }
      return nullptr;
   }
   void handleRec(mlir::Value val, lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo info) {
      if (!mlir::isa<tuples::TupleStreamType>(val.getType())) return;
      std::vector<std::pair<mlir::Operation*, uint32_t>> uses;
      for (auto& use : val.getUses()) {
         uses.push_back({use.getOwner(), use.getOperandNumber()});
      }
      for (auto [useOwner, useIdx] : uses) {
         if (auto nullCollumnTypeChangeable = mlir::dyn_cast<NullColumnTypeChangeable>(useOwner)) {
            if (nullCollumnTypeChangeable.changeForColumns(info).succeeded()) {
               if (auto op = mlir::dyn_cast<Operator>(useOwner)) {
                  handleRec(op.asRelation(), info);
               }
            } else {
               materialize(useOwner, useIdx, info);
            }
         } else {
            materialize(useOwner, useIdx, info);
         }
      }
   }
   void runOnOperation() override {
      llvm::DenseMap<Operator, relalg::ColumnSet> cache;
      auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      mlir::OpBuilder builder(&getContext());
      getOperation().walk([&](relalg::BaseTableOp baseTableOp) {
         auto required = getRequired(baseTableOp, cache);
         if (!baseTableOp.getRestriction().filterDescription.empty()) {
            auto usedColumns = baseTableOp.getUsedColumns();
            auto restrictions = baseTableOp.getRestriction().filterDescription;
            lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo info;
            std::unordered_set<std::string> notNullColumns;
            for (auto& r : restrictions) {
               if (r.op == lingodb::runtime::FilterOp::NOTNULL) {
                  notNullColumns.insert(r.columnName);
               }
            }
            if (notNullColumns.empty()) return;
            std::vector<mlir::NamedAttribute> mapping;
            for (auto x : baseTableOp.getColumnsAttr()) {
               if (notNullColumns.contains(x.getName().str())) {
                  auto colDef = mlir::cast<tuples::ColumnDefAttr>(x.getValue());
                  if (required.contains(&colDef.getColumn())) {
                     auto [scope, name] = colManager.getName(&colDef.getColumn());
                     auto newColDef = colManager.createDef(scope, name + "__notnull");
                     newColDef.getColumn().type = getBaseType(colDef.getColumn().type);
                     info.directMappings[&colDef.getColumn()] = &newColDef.getColumn();
                     mapping.push_back(builder.getNamedAttr(x.getName(), newColDef));
                  }
               } else {
                  mapping.push_back(x);
               }
            }
            baseTableOp.setColumnsAttr(builder.getDictionaryAttr(mapping));
            handleRec(baseTableOp.asRelation(), info);
         }
      });
      getOperation().walk([&](relalg::SelectionOp selection) {
         lingodb::compiler::dialect::relalg::ColumnNullableChangeInfo info;
         if (auto* notNullCheckedCol = handleNullCheck(info, selection)) {
            auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
            auto [scope, name] = colManager.getName(notNullCheckedCol);
            auto newColDef = colManager.createDef(scope, name + "__notnullsel");
            newColDef.getColumn().type = getBaseType(notNullCheckedCol->type);
            info.directMappings[notNullCheckedCol] = &newColDef.getColumn();
            handleRec(selection.asRelation(), info);
            mlir::OpBuilder builder(selection->getContext());
            builder.setInsertionPointAfter(selection);
            auto* block = new mlir::Block;
            auto loc = builder.getUnknownLoc();
            auto tuple = block->addArgument(tuples::TupleType::get(&getContext()), loc);
            {
               mlir::OpBuilder::InsertionGuard guard(builder);
               builder.setInsertionPointToStart(block);
               auto colRef = colManager.createRef(notNullCheckedCol);
               mlir::Value val = builder.create<tuples::GetColumnOp>(loc, notNullCheckedCol->type, colRef, tuple);
               mlir::Value nonNullableVal = builder.create<db::NullableGetVal>(loc, getBaseType(notNullCheckedCol->type), val);
               builder.create<tuples::ReturnOp>(loc, nonNullableVal);
            }
            auto mapOp = builder.create<relalg::MapOp>(selection->getLoc(), tuples::TupleStreamType::get(&getContext()), selection.asRelation(), builder.getArrayAttr({newColDef}));
            mapOp.getPredicate().push_back(block);
            selection.asRelation().replaceAllUsesExcept(mapOp.asRelation(), mapOp);
            handleRec(mapOp.asRelation(), info);
         }
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createEliminateNullableTypesPass() { return std::make_unique<EliminateNullableTypes>(); }
