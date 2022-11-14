#include "frontend/SQL/Parser.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "runtime/MetaDataOnlyDatabase.h"
int main(int argc, char** argv) {
   mlir::MLIRContext context;
   mlir::DialectRegistry registry;
   registry.insert<mlir::BuiltinDialect>();
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::subop::SubOperatorDialect>();
   registry.insert<mlir::tuples::TupleStreamDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   context.appendDialectRegistry(registry);
   context.loadAllAvailableDialects();
   context.loadDialect<mlir::relalg::RelAlgDialect>();
   mlir::OpBuilder builder(&context);
   std::string filename = std::string(argv[1]);
   auto metadataDB = runtime::MetaDataOnlyDatabase::emptyMetaData();
   if (argc >= 3) {
      std::string metadataFile = std::string(argv[2]);
      metadataDB = runtime::MetaDataOnlyDatabase::loadMetaData(metadataFile);
   }
   std::ifstream istream{filename};
   std::stringstream buffer;
   buffer << istream.rdbuf();
   mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
   frontend::sql::Parser translator(buffer.str(), *metadataDB, moduleOp);

   builder.setInsertionPointToStart(moduleOp.getBody());
   auto* queryBlock = new mlir::Block;
   {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(queryBlock);
      auto val = translator.translate(builder);
      if (val.has_value()) {
         builder.create<mlir::subop::SetResultOp>(builder.getUnknownLoc(), 0, val.value());
      }
      builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
   }
   mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {}));
   funcOp.getBody().push_back(queryBlock);

   mlir::OpPrintingFlags flags;
   flags.assumeVerified();
   moduleOp->print(llvm::outs(), flags);
   return 0;
}