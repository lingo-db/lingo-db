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

class ParseNestedSQL : public mlir::PassWrapper<ParseNestedSQL, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-parse-nested-sql"; }
   lingodb::catalog::Catalog& catalog;

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParseNestedSQL)
   ParseNestedSQL(lingodb::catalog::Catalog& catalog) : catalog(catalog) {}
   void runOnOperation() override {
      size_t cnt = 0;
      auto moduleOp = mlir::OperationPass<mlir::ModuleOp>::getOperation();
      std::vector<relalg::SQLQueryOp> nestedQueries;
      getOperation().walk([&](relalg::SQLQueryOp op) {
         nestedQueries.push_back(op);
      });
      for (auto op : nestedQueries) {
         auto sqlString = op.getSql().str();
         std::string scopePrefix = "nested_sql_" + std::to_string(cnt++) + "_";
         Driver drv;
         if (!drv.parse(sqlString, false)) {
            auto sqlContext = std::make_shared<lingodb::analyzer::SQLContext>(std::vector<mlir::Value>(op.getParameters().begin(), op.getParameters().end()), scopePrefix);
            sqlContext->catalog = &catalog;
            lingodb::analyzer::SQLQueryAnalyzer analyzer{&catalog};
            drv.result[0] = analyzer.canonicalizeAndAnalyze(drv.result[0], sqlContext);
            lingodb::translator::SQLMlirTranslator translator{moduleOp, &catalog};
            mlir::OpBuilder builder(op);
            auto val = translator.translateStart(builder, drv.result[0], sqlContext);
            if (!val.has_value()) {
               op.emitError("Could not translate query");
               signalPassFailure();
               return;
            }
            if (auto queryOp = mlir::dyn_cast<relalg::QueryOp>(val.value().getDefiningOp())) {
               auto terminator = queryOp.getQueryOps().front().getTerminator();
               std::vector<mlir::Operation*> toMove;
               for (auto& o : queryOp.getQueryOps().front().getOperations()) {
                  if (&o == terminator) {
                     continue;
                  }
                  toMove.push_back(&o);
               }
               for (auto o : toMove) {
                  o->remove();
                  builder.insert(o);
               }
               auto materializeOp = mlir::cast<relalg::MaterializeOp>(terminator->getOperand(0).getDefiningOp());
               auto scalarOp = builder.create<relalg::GetScalarOp>(op.getLoc(), op.getResult().getType(), mlir::cast<tuples::ColumnRefAttr>(materializeOp.getCols()[0]), materializeOp.getRel());
               materializeOp->dropAllUses();
               materializeOp->erase();
               op.getResult().replaceAllUsesWith(scalarOp.getResult());
               queryOp.erase();
               op->erase();
            }
         } else {
            op.emitError("Could not parse query");
            signalPassFailure();
         }
      }
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createParseNestedSQLPass(catalog::Catalog& catalog) { return std::make_unique<ParseNestedSQL>(catalog); }
