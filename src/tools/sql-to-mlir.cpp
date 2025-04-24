#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/old-frontend/SQL/Parser.h"
#include "lingodb/runtime/Session.h"

#include "mlir/IR/BuiltinDialect.h"

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
   lingodb::compiler::frontend::sql::Parser translator(sql, *catalog, moduleOp);

   builder.setInsertionPointToStart(moduleOp.getBody());
   auto* queryBlock = new mlir::Block;
   {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(queryBlock);
      auto val = translator.translate(builder);
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