#include "lingodb/catalog/Catalog.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
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

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParseNestedSQL)
   ParseNestedSQL(lingodb::catalog::Catalog& catalog) : catalog(catalog) {}

   void runOnOperation() override {
      size_t cnt = 0;
      auto moduleOp = getOperation();
      std::vector<relalg::SQLQueryOp> nestedQueries;
      moduleOp.walk([&](relalg::SQLQueryOp op) {
         nestedQueries.push_back(op);
      });
      for (auto op : nestedQueries) {
         auto sqlString = op.getSql().str();
         std::string scopePrefix = "nested_sql_" + std::to_string(cnt++) + "_";
         ::Driver drv;
         if (drv.parse(sqlString, /*isFile=*/false)) {
            op.emitError("Could not parse nested SQL");
            signalPassFailure();
            return;
         }
         std::vector<mlir::Value> params(op.getParameters().begin(), op.getParameters().end());
         auto sqlContext = std::make_shared<lingodb::analyzer::SQLContext>(params, scopePrefix);
         sqlContext->catalog = &catalog;
         lingodb::analyzer::SQLQueryAnalyzer analyzer{&catalog};
         drv.result[0] = analyzer.canonicalizeAndAnalyze(drv.result[0], sqlContext);

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

         lingodb::translator::SQLMlirTranslator translator{moduleOp, &catalog};
         mlir::OpBuilder builder(op);
         auto translated = translator.translateStart(builder, drv.result[0], sqlContext);
         if (!translated.has_value()) {
            op.emitError("Could not translate nested SQL");
            signalPassFailure();
            return;
         }
         auto materializeOp = mlir::dyn_cast_or_null<relalg::MaterializeOp>(translated.value().getDefiningOp());
         if (!materializeOp) {
            op.emitError("Nested SQL did not lower to a relalg.materialize");
            signalPassFailure();
            return;
         }
         auto colRef = mlir::cast<tuples::ColumnRefAttr>(materializeOp.getCols()[0]);
         auto scalarOp = builder.create<relalg::GetScalarOp>(op.getLoc(), op.getResult().getType(), colRef, materializeOp.getRel());
         op.getResult().replaceAllUsesWith(scalarOp.getResult());
         materializeOp->dropAllUses();
         materializeOp->erase();
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
