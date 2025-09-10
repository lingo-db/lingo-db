#include "lingodb/runtime/Session.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"

#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/frontend/driver.h"
#include "lingodb/compiler/frontend/sql_analyzer.h"
#include "lingodb/compiler/frontend/sql_mlir_translator.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <fstream>

namespace {
using namespace lingodb::compiler::dialect;
void printMLIR(std::string sql, std::shared_ptr<lingodb::catalog::Catalog> catalog) {
   mlir::MLIRContext context;
   mlir::DialectRegistry registry;
   registry.insert<mlir::BuiltinDialect>();
   registry.insert<relalg::RelAlgDialect>();
   registry.insert<subop::SubOperatorDialect>();
   registry.insert<tuples::TupleStreamDialect>();
   registry.insert<db::DBDialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   context.appendDialectRegistry(registry);
   context.loadAllAvailableDialects();
   context.loadDialect<relalg::RelAlgDialect>();
   mlir::OpBuilder builder(&context);
   mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

   builder.setInsertionPointToStart(moduleOp.getBody());
   auto* queryBlock = new mlir::Block;
   {
      driver drv;
      lingodb::analyzer::SQLQueryAnalyzer analyzer{catalog.get()};
      lingodb::translator::SQLMlirTranslator translator{moduleOp, catalog.get()};
      auto sqlContext = std::make_shared<lingodb::analyzer::SQLContext>();
      sqlContext->catalog = catalog.get();
      if (!drv.parse(":" + sql)) {
         auto results = drv.result;
         if (results.size() > 1) {
            throw std::runtime_error("Only one statement allowed");
         }
         drv.result[0] = analyzer.canonicalizeAndAnalyze(drv.result[0], sqlContext);
         auto val = translator.translateStart(builder, drv.result[0], sqlContext);
      } else {
         throw std::runtime_error("Something went wrong");
      }


      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(queryBlock);



      auto val = translator.translateStart(builder, drv.result[0], sqlContext);
      if (val.has_value()) {
         builder.create<subop::SetResultOp>(builder.getUnknownLoc(), 0, val.value());
      }
      builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
   }
   mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {}));
   funcOp.getBody().push_back(queryBlock);

   mlir::OpPrintingFlags flags;
   flags.assumeVerified();
   moduleOp->print(llvm::outs(), flags);
   moduleOp.erase();
}
} // end namespace
int main(int argc, char** argv) {
   std::string filename = std::string(argv[1]);
   auto catalog = lingodb::catalog::Catalog::createEmpty();
   if (argc >= 3) {
      std::string dbDir = std::string(argv[2]);
      catalog = lingodb::catalog::Catalog::create(dbDir, false);
   }
   std::ifstream istream{filename};
   std::stringstream buffer;
   buffer << istream.rdbuf();
   while (true) {
      std::stringstream query;
      std::string line;
      std::getline(buffer, line);
      while (true) {
         if (!buffer.good()) {
            if (buffer.eof()) {
               query << line << std::endl;
            }
            break;
         }
         query << line << std::endl;
         if (!line.empty() && line.find(';') == line.size() - 1) {
            break;
         }
         std::getline(buffer, line);
      }
      printMLIR(query.str(), catalog);
      if (buffer.eof()) {
         //exit from repl loop
         break;
      }
   }
   return 0;
}