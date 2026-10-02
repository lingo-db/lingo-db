#include "lingodb/catalog/Catalog.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/frontend/driver.h"
#include "lingodb/compiler/frontend/sql_analyzer.h"
#include "lingodb/compiler/frontend/sql_context.h"
#include "lingodb/compiler/frontend/sql_mlir_translator.h"

#include "mlir/IR/BuiltinOps.h"

#include <lingodb/catalog/TableCatalogEntry.h>

namespace {
using namespace lingodb::compiler::dialect;

// Replace every `relalg.sql_query` op with the relalg ops produced by parsing
// + analyzing the embedded SQL text. The SQL string can reference the op's
// variadic parameter Values via `PARAM(n)`; SQLContext carries them through
// to the analyzer/translator. The translator emits a `relalg.materialize`
// for the parsed query — we convert that to a `relalg.getscalar` since the
// surrounding hipy UDF expects a single scalar value (typically wrapped in
// a nullable<T>).
class ParseNestedSQL : public mlir::PassWrapper<ParseNestedSQL, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-parse-nested-sql"; }
   lingodb::catalog::Catalog& catalog;
   size_t cnt = 0;

   // Parse, analyze and translate the SQL text of `op` in front of `builder`'s
   // insertion point. Returns the translator's relalg.materialize (or null
   // after emitting an error).
   relalg::MaterializeOp translate(relalg::SQLQueryOp op, mlir::OpBuilder& builder) {
      std::string scopePrefix = "nested_sql_" + std::to_string(cnt++) + "_";
      ::Driver drv;
      if (drv.parse(op.getSql().str(), /*isFile=*/false)) {
         op.emitError("Could not parse nested SQL");
         return {};
      }
      std::vector<mlir::Value> params(op.getParameters().begin(), op.getParameters().end());
      auto sqlContext = std::make_shared<lingodb::analyzer::SQLContext>(params, scopePrefix);
      sqlContext->catalog = &catalog;
      lingodb::analyzer::SQLQueryAnalyzer analyzer{&catalog};
      drv.result[0] = analyzer.canonicalizeAndAnalyze(drv.result[0], sqlContext);
      lingodb::translator::SQLMlirTranslator translator{getOperation(), &catalog};
      auto translated = translator.translateStart(builder, drv.result[0], sqlContext);
      if (!translated.has_value()) {
         op.emitError("Could not translate nested SQL");
         return {};
      }
      auto materializeOp = mlir::dyn_cast_or_null<relalg::MaterializeOp>(translated.value().getDefiningOp());
      if (!materializeOp) {
         op.emitError("Nested SQL did not lower to a relalg.materialize");
      }
      return materializeOp;
   }

   // Cast `value` to the base type of `targetType`, keeping its nullability
   // (e.g. an int4 column read as hipy's sql.nullable(int)). With `exact`,
   // also adapt the nullability to `targetType`.
   static mlir::Value coerce(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, mlir::Type targetType, bool exact) {
      auto* ctxt = builder.getContext();
      mlir::Type valueType = value.getType();
      bool valueNullable = mlir::isa<db::NullableType>(valueType);
      if (getBaseType(valueType) != getBaseType(targetType)) {
         mlir::Type castedType = getBaseType(targetType);
         if (valueNullable) castedType = db::NullableType::get(ctxt, castedType);
         value = builder.create<db::CastOp>(loc, castedType, value);
      }
      if (exact && value.getType() != targetType) {
         if (mlir::isa<db::NullableType>(targetType)) {
            value = builder.create<db::AsNullableOp>(loc, targetType, value);
         } else {
            value = builder.create<db::NullableGetVal>(loc, targetType, value);
         }
      }
      return value;
   }

   // Append a relalg.map to `rel` computing new columns of types
   // `resultTypes` from the columns `cols` (via `compute`).
   std::pair<mlir::Value, llvm::SmallVector<tuples::ColumnRefAttr>> addMap(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value rel, llvm::ArrayRef<mlir::Type> resultTypes, llvm::ArrayRef<tuples::ColumnRefAttr> cols, const std::function<llvm::SmallVector<mlir::Value>(mlir::OpBuilder&, llvm::ArrayRef<mlir::Value>)>& compute) {
      auto& colManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      std::string scope = colManager.getUniqueScope("nested_sql_map");
      llvm::SmallVector<mlir::Attribute> defs;
      llvm::SmallVector<tuples::ColumnRefAttr> refs;
      for (auto [i, type] : llvm::enumerate(resultTypes)) {
         auto def = colManager.createDef(scope, "res" + std::to_string(i));
         def.getColumn().type = type;
         defs.push_back(def);
         refs.push_back(colManager.createRef(&def.getColumn()));
      }
      auto* mapBlock = new mlir::Block;
      auto tuple = mapBlock->addArgument(tuples::TupleType::get(&getContext()), loc);
      {
         mlir::OpBuilder mapBuilder(&getContext());
         mapBuilder.setInsertionPointToStart(mapBlock);
         llvm::SmallVector<mlir::Value> values;
         for (auto col : cols) {
            values.push_back(mapBuilder.create<tuples::GetColumnOp>(loc, col.getColumn().type, col, tuple));
         }
         mapBuilder.create<tuples::ReturnOp>(loc, compute(mapBuilder, values));
      }
      auto mapOp = builder.create<relalg::MapOp>(loc, tuples::TupleStreamType::get(&getContext()), rel, builder.getArrayAttr(defs));
      mapOp.getPredicate().push_back(mapBlock);
      return {mapOp.asRelation(), refs};
   }

   // getscalar of column `idx` of a translated query, cast to `type`'s base
   // type. Consumes `materializeOp`.
   mlir::Value scalarFromColumn(mlir::OpBuilder& builder, mlir::Location loc, relalg::MaterializeOp materializeOp, size_t idx, mlir::Type type) {
      auto colRef = mlir::cast<tuples::ColumnRefAttr>(materializeOp.getCols()[idx]);
      mlir::Value rel = materializeOp.getRel();
      if (getBaseType(colRef.getColumn().type) != getBaseType(type)) {
         mlir::Type castedType = getBaseType(type);
         if (mlir::isa<db::NullableType>(colRef.getColumn().type)) castedType = db::NullableType::get(&getContext(), castedType);
         auto [mappedRel, mappedCols] = addMap(builder, loc, rel, {castedType}, {colRef}, [&](mlir::OpBuilder& b, llvm::ArrayRef<mlir::Value> values) -> llvm::SmallVector<mlir::Value> {
            return {coerce(b, loc, values[0], type, /*exact=*/false)};
         });
         rel = mappedRel;
         colRef = mappedCols[0];
      }
      mlir::Value res = builder.create<relalg::GetScalarOp>(loc, type, colRef, rel);
      materializeOp->dropAllUses();
      materializeOp->erase();
      return res;
   }

   // Lower one relalg.sql_query; returns the replacement value or null.
   mlir::Value lower(relalg::SQLQueryOp op, mlir::OpBuilder& builder) {
      auto materializeOp = translate(op, builder);
      if (!materializeOp) return {};
      return scalarFromColumn(builder, op.getLoc(), materializeOp, 0, op.getResult().getType());
   }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParseNestedSQL)
   ParseNestedSQL(lingodb::catalog::Catalog& catalog) : catalog(catalog) {}

   void runOnOperation() override {
      auto moduleOp = getOperation();
      std::vector<relalg::SQLQueryOp> nestedQueries;
      moduleOp.walk([&](relalg::SQLQueryOp op) {
         nestedQueries.push_back(op);
      });
      for (auto op : nestedQueries) {
         // Snapshot the set of relalg ops present before translation, so we
         // can identify which ops translateStart inserted and tag them.
         llvm::DenseSet<mlir::Operation*> preexisting;
         auto parentFunc = op->getParentOfType<mlir::func::FuncOp>();
         if (parentFunc) {
            parentFunc.walk([&](mlir::Operation* o) {
               if (o->getDialect() && o->getDialect()->getNamespace() == "relalg")
                  preexisting.insert(o);
            });
         }

         mlir::OpBuilder builder(op);
         mlir::Value replacement = lower(op, builder);
         if (!replacement) {
            signalPassFailure();
            return;
         }
         op.getResult().replaceAllUsesWith(replacement);
         op->erase();

         // Tag every relalg op the translator just inserted so CSE doesn't
         // merge them with structurally-identical outer ops. The inlined
         // subtree's BaseTables carry scope-prefixed columns AND capture
         // outer ColumnRefs via PARAM(n) — merging with the outer subtree
         // silently equates the two sets of column references and produces
         // nested_map captures the SubOp→ControlFlow lowering can't resolve.
         if (parentFunc) {
            parentFunc.walk([&](mlir::Operation* o) {
               if (o->getDialect() && o->getDialect()->getNamespace() == "relalg" && !preexisting.contains(o)) {
                  o->setAttr("nested_sql_emitted", mlir::UnitAttr::get(o->getContext()));
               }
            });
         }
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::relalg::createParseNestedSQLPass(lingodb::catalog::Catalog& catalog) {
   return std::make_unique<ParseNestedSQL>(catalog);
}
