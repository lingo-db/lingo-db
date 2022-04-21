#include "frontend/SQL/Parser.h"
#include "runtime/MetaDataOnlyDatabase.h"
int main(int argc, char** argv) {
   mlir::MLIRContext context;
   mlir::DialectRegistry registry;
   registry.insert<mlir::BuiltinDialect>();
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithmeticDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   context.appendDialectRegistry(registry);
   context.loadAllAvailableDialects();
   context.loadDialect<mlir::relalg::RelAlgDialect>();
   mlir::OpBuilder builder(&context);
   std::string filename = std::string(argv[1]);
   std::string metadataFile = std::string(argv[2]);
   auto metadataDB = runtime::MetaDataOnlyDatabase::loadMetaData(metadataFile);
   std::ifstream istream{filename};
   std::stringstream buffer;
   buffer << istream.rdbuf();
   frontend::sql::Parser translator(buffer.str(), *metadataDB, &context);
   mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

   builder.setInsertionPointToStart(moduleOp.getBody());
   mlir::FuncOp funcOp = builder.create<mlir::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {mlir::dsa::TableType::get(builder.getContext())}));
   funcOp.body().push_back(new mlir::Block);
   builder.setInsertionPointToStart(&funcOp.body().front());
   mlir::Value val = translator.translate(builder);

   builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), val);
   mlir::OpPrintingFlags flags;
   flags.assumeVerified();
   moduleOp->print(llvm::outs(), flags);
   return 0;
}