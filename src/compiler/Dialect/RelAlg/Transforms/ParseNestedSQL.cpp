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

#include "lingodb/compiler/Dialect/util/UtilOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
// a nullable<T>) or, for a row-valued query (result type tuple<...>, hipy's
// sql.row(...)), one value per result column.
class ParseNestedSQL : public mlir::PassWrapper<ParseNestedSQL, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-parse-nested-sql"; }
   lingodb::catalog::Catalog& catalog;
   size_t cnt = 0;

   // Parse, analyze and translate the SQL text of `op` in front of `builder`'s
   // insertion point. Returns the translator's relalg.materialize (or null
   // after emitting an error). `zeroInsteadOfNull` receives, per result
   // column, whether a missing value means 0 (an ungrouped COUNT: when the
   // query is decorrelated, an empty input yields no group, i.e. NULL; the SQL
   // translator compensates the same way for scalar subqueries).
   relalg::MaterializeOp translate(relalg::SQLQueryOp op, mlir::OpBuilder& builder, std::vector<bool>* zeroInsteadOfNull = nullptr) {
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
      if (zeroInsteadOfNull) {
         for (auto& column : sqlContext->currentScope->targetInfo.getTargetColumns()) {
            zeroInsteadOfNull->push_back(column->resultType.useZeroInsteadOfNull);
         }
      }
      return materializeOp;
   }

   // Cast `value` to the base type of `targetType`, keeping its nullability
   // (e.g. an int4 column read as hipy's sql.nullable(int)). With `exact`,
   // also make a non-nullable value nullable if `targetType` is.
   static mlir::Value coerce(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, mlir::Type targetType, bool exact) {
      auto* ctxt = builder.getContext();
      mlir::Type valueType = value.getType();
      bool valueNullable = mlir::isa<db::NullableType>(valueType);
      if (getBaseType(valueType) != getBaseType(targetType)) {
         mlir::Type castedType = getBaseType(targetType);
         if (valueNullable) castedType = db::NullableType::get(ctxt, castedType);
         value = builder.create<db::CastOp>(loc, castedType, value);
      }
      if (exact && !valueNullable && mlir::isa<db::NullableType>(targetType)) {
         value = builder.create<db::AsNullableOp>(loc, targetType, value);
      }
      return value;
   }

   // The value of a nullable `value`; a runtime error if it is NULL (a nested
   // query result declared non-nullable, e.g. sql.execute(int, ...)).
   static mlir::Value checkedGetValue(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, llvm::StringRef message) {
      auto nullableType = mlir::cast<db::NullableType>(value.getType());
      mlir::Value isNull = builder.create<db::IsNullOp>(loc, value);
      builder.create<mlir::scf::IfOp>(loc, isNull, [&](mlir::OpBuilder& b, mlir::Location loc) {
         mlir::Value messageValue = b.create<db::ConstantOp>(loc, db::StringType::get(b.getContext()), b.getStringAttr(message));
         b.create<db::RuntimeCall>(loc, mlir::TypeRange{}, "RaiseError", mlir::ValueRange{messageValue});
         b.create<mlir::scf::YieldOp>(loc);
      });
      return builder.create<db::NullableGetVal>(loc, nullableType.getType(), value);
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

   bool checkColumnCount(relalg::SQLQueryOp op, relalg::MaterializeOp materializeOp, size_t expected) {
      if (materializeOp.getCols().size() != expected) {
         op.emitError("Nested SQL returns ") << materializeOp.getCols().size() << " columns, but the declared row type has " << expected;
         return false;
      }
      return true;
   }

   // Lower one relalg.sql_query; returns the replacement value or null.
   mlir::Value lower(relalg::SQLQueryOp op, mlir::OpBuilder& builder) {
      auto loc = op.getLoc();
      mlir::Type resultType = op.getResult().getType();
      auto tupleType = mlir::dyn_cast<mlir::TupleType>(resultType);
      // sql.nullable(sql.row(...)): NULL instead of a runtime error if there is no row
      bool optionalRow = false;
      if (auto nullableType = mlir::dyn_cast<db::NullableType>(resultType)) {
         if ((tupleType = mlir::dyn_cast<mlir::TupleType>(nullableType.getType()))) {
            optionalRow = true;
         }
      }
      if (!tupleType) {
         // getscalar yields NULL for "no row" as well, so it is always
         // nullable; a non-nullable result type gets a runtime error instead
         std::vector<bool> zeroInsteadOfNull;
         auto materializeOp = translate(op, builder, &zeroInsteadOfNull);
         if (!materializeOp) return {};
         mlir::Type baseType = getBaseType(resultType);
         mlir::Type scalarType = db::NullableType::get(&getContext(), baseType);
         mlir::Value scalar = scalarFromColumn(builder, loc, materializeOp, 0, scalarType);
         if (!zeroInsteadOfNull.empty() && zeroInsteadOfNull[0] && (mlir::isa<mlir::IntegerType>(baseType) || mlir::isa<mlir::FloatType>(baseType))) {
            mlir::Value isNull = builder.create<db::IsNullOp>(loc, scalar);
            mlir::Value value = builder.create<db::NullableGetVal>(loc, baseType, scalar);
            mlir::Attribute zeroAttr = mlir::isa<mlir::FloatType>(baseType) ? mlir::Attribute(builder.getFloatAttr(baseType, 0.0)) : mlir::Attribute(builder.getIntegerAttr(baseType, 0));
            mlir::Value zero = builder.create<db::ConstantOp>(loc, baseType, zeroAttr);
            mlir::Value count = builder.create<mlir::arith::SelectOp>(loc, isNull, zero, value);
            if (mlir::isa<db::NullableType>(resultType)) return builder.create<db::AsNullableOp>(loc, resultType, count);
            return count;
         }
         if (mlir::isa<db::NullableType>(resultType)) return scalar;
         return checkedGetValue(builder, loc, scalar, "nested SQL query returned NULL or no row for a non-nullable result type");
      }
      // Row-valued query: the values of its first row, one per declared tuple
      // element (cast to the declared element types); a runtime error if it
      // yields no row (unless optionalRow), or NULL for a non-nullable element.
      auto materializeOp = translate(op, builder);
      if (!materializeOp) return {};
      if (!checkColumnCount(op, materializeOp, tupleType.size())) return {};
      mlir::Value rel = materializeOp.getRel();
      llvm::SmallVector<tuples::ColumnRefAttr> cols;
      // the declared element types, but nullable where a non-nullable element
      // is read from a nullable column (checked after getfirstrow)
      llvm::SmallVector<mlir::Type> rowTypes;
      bool needsCast = false;
      bool needsCheck = false;
      for (auto [col, type] : llvm::zip(materializeOp.getCols(), tupleType.getTypes())) {
         cols.push_back(mlir::cast<tuples::ColumnRefAttr>(col));
         mlir::Type rowType = type;
         if (mlir::isa<db::NullableType>(cols.back().getColumn().type) && !mlir::isa<db::NullableType>(type)) {
            rowType = db::NullableType::get(&getContext(), type);
            needsCheck = true;
         }
         rowTypes.push_back(rowType);
         needsCast |= cols.back().getColumn().type != rowType;
      }
      if (needsCast) {
         std::tie(rel, cols) = addMap(builder, loc, rel, rowTypes, cols, [&](mlir::OpBuilder& b, llvm::ArrayRef<mlir::Value> values) {
            llvm::SmallVector<mlir::Value> casted;
            for (auto [value, type] : llvm::zip(values, rowTypes)) {
               casted.push_back(coerce(b, loc, value, type, /*exact=*/true));
            }
            return casted;
         });
      }
      materializeOp->dropAllUses();
      materializeOp->erase();
      llvm::SmallVector<mlir::Attribute> colAttrs(cols.begin(), cols.end());
      mlir::Type rowTupleType = mlir::TupleType::get(&getContext(), rowTypes);
      if (optionalRow) rowTupleType = db::NullableType::get(&getContext(), rowTupleType);
      mlir::Value row = builder.create<relalg::GetFirstRowOp>(loc, rowTupleType, rel, builder.getArrayAttr(colAttrs));
      if (!needsCheck) return row;
      auto checkRow = [&](mlir::OpBuilder& b, mlir::Value row) -> mlir::Value {
         auto values = b.create<util::UnPackOp>(loc, row).getResults();
         llvm::SmallVector<mlir::Value> checked;
         for (auto [value, type] : llvm::zip(values, tupleType.getTypes())) {
            checked.push_back(value.getType() == type ? mlir::Value(value) : checkedGetValue(b, loc, value, "nested SQL query returned NULL for a non-nullable result type"));
         }
         return b.create<util::PackOp>(loc, tupleType, checked);
      };
      if (!optionalRow) return checkRow(builder, row);
      mlir::Value isNull = builder.create<db::IsNullOp>(loc, row);
      auto ifOp = builder.create<mlir::scf::IfOp>(
         loc, isNull, [&](mlir::OpBuilder& b, mlir::Location loc) {
            mlir::Value null = b.create<db::NullOp>(loc, resultType);
            b.create<mlir::scf::YieldOp>(loc, null); }, [&](mlir::OpBuilder& b, mlir::Location loc) {
            mlir::Value value = b.create<db::NullableGetVal>(loc, getBaseType(row.getType()), row);
            mlir::Value checked = b.create<db::AsNullableOp>(loc, resultType, checkRow(b, value));
            b.create<mlir::scf::YieldOp>(loc, checked); });
      return ifOp.getResult(0);
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
