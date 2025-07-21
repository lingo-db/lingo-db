#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
//#include "lingodb/compiler/frontend/SQL/Parser.h"
#include "lingodb/compiler/frontend/driver.h"
#include "lingodb/compiler/frontend/sql_analyzer.h"
#include "lingodb/compiler/frontend/sql_mlir_translator.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/runtime/Session.h"

#include "mlir/IR/BuiltinDialect.h"
#include <iostream>
int main(int argc, char* argv[]) {
   if (argc <= 2) {
      std::cerr << "USAGE: <sql database> <sql statement> [-e]" << std::endl;
      return 1;
   }

   auto session = lingodb::runtime::Session::createSession(std::string(argv[1]), false);
   driver drv;
   if (!drv.parse(argv[2])) {
      auto results = drv.result;
      if (results.empty()) {
         std::cerr << "Error parsing query" << std::endl;
         return 1;
      }
      std::vector<std::shared_ptr<lingodb::analyzer::SQLContext>> contexts;
      for (size_t i = 0; i < results.size(); ++i) {
         std::cout << "------------------" << i << "----------------------" << std::endl;

         std::cout << "digraph ast {" << std::endl;
         lingodb::ast::NodeIdGenerator idGen{};
         std::cout << drv.result[i]->toDotGraph(1, idGen) << std::endl;
         std::cout << "}" << std::endl;

         lingodb::analyzer::SQLQueryAnalyzer sqlAnalyzer{session->getCatalog()};
         auto sqlContext = std::make_shared<lingodb::analyzer::SQLContext>();
         contexts.push_back(sqlContext);
         sqlContext->catalog = session->getCatalog();
         lingodb::analyzer::SQLQueryAnalyzer analyzer{session->getCatalog()};
         drv.result[i] = analyzer.canonicalizeAndAnalyze(drv.result[i], sqlContext);
         std::cout << std::endl
                   << std::endl;
         std::cout << "After" << std::endl;
         std::cout << "digraph ast {" << std::endl;
         std::cout << drv.result[i]->toDotGraph(1, idGen) << std::endl;
         std::cout << "}" << std::endl;
      }
      for (size_t i = 0; i < results.size(); ++i) {
         std::cout << "------------------" << i << "----------------------" << std::endl;
         /*
          * Translator
          */
         mlir::MLIRContext context;
         mlir::DialectRegistry registry;
         registry.insert<mlir::BuiltinDialect>();
         registry.insert<lingodb::compiler::dialect::relalg::RelAlgDialect>();
         registry.insert<lingodb::compiler::dialect::subop::SubOperatorDialect>();
         registry.insert<lingodb::compiler::dialect::tuples::TupleStreamDialect>();
         registry.insert<lingodb::compiler::dialect::db::DBDialect>();
         registry.insert<mlir::func::FuncDialect>();
         registry.insert<mlir::arith::ArithDialect>();

         registry.insert<mlir::memref::MemRefDialect>();
         registry.insert<lingodb::compiler::dialect::util::UtilDialect>();
         registry.insert<mlir::scf::SCFDialect>();
         registry.insert<mlir::LLVM::LLVMDialect>();
         context.appendDialectRegistry(registry);
         context.loadAllAvailableDialects();
         context.loadDialect<lingodb::compiler::dialect::relalg::RelAlgDialect>();
         mlir::OpBuilder builder(&context);
         mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
         builder.setInsertionPointToStart(moduleOp.getBody());
         lingodb::translator::SQLMlirTranslator translator{moduleOp, session->getCatalog()};
         auto* queryBlock = new mlir::Block;
         {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(queryBlock);
            auto val = translator.translateStart(builder, drv.result[i], contexts[i]);
            if (val.has_value()) {
               builder.create<lingodb::compiler::dialect::subop::SetResultOp>(builder.getUnknownLoc(), 0, val.value());
            }
            builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
         }
         mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {}));
         funcOp.getBody().push_back(queryBlock);

         mlir::OpPrintingFlags flags;
         flags.assumeVerified();
         std::string output;
         llvm::raw_string_ostream os(output);
         moduleOp->print(os, flags);
         os.flush();
         moduleOp.erase();
         std::cout << output << std::endl;
      }

   } else {
      return 1;
   }

   return 0;
}