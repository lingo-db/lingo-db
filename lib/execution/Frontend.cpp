#include "execution/Frontend.h"
#include "frontend/SQL/Parser.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/Passes.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <iostream>

namespace {

void initializeContext(mlir::MLIRContext& context) {
   mlir::DialectRegistry registry;
   registry.insert<mlir::BuiltinDialect>();
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::tuples::TupleStreamDialect>();
   registry.insert<mlir::subop::SubOperatorDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::dsa::DSADialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithDialect>();
   registry.insert<mlir::cf::ControlFlowDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   context.appendDialectRegistry(registry);
   context.loadAllAvailableDialects();
   context.loadDialect<mlir::relalg::RelAlgDialect>();
   context.disableMultithreading();
}
class MLIRFrontend : public execution::Frontend {
   mlir::MLIRContext context;
   mlir::OwningOpRef<mlir::ModuleOp> module;
   void loadFromFile(std::string fileName) override {
      initializeContext(context);
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
         llvm::MemoryBuffer::getFileOrSTDIN(fileName);
      if (std::error_code ec = fileOrErr.getError()) {
         error.emit() << "Could not open input file: " << ec.message();
         return;
      }
      llvm::SourceMgr sourceMgr;
      sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
      module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
      if (!module) {
         error.emit() << "Error can't load file " << fileName << "\n";
         return;
      }
   }
   void loadFromString(std::string data) override {
      initializeContext(context);
      module = mlir::parseSourceString<mlir::ModuleOp>(data, &context);
      if (!module) {
         error.emit() << "Error can't load module\n";
      }
   }
   mlir::ModuleOp* getModule() override {
      assert(module);
      return module.operator->();
   }
};
class SQLFrontend : public execution::Frontend {
   mlir::MLIRContext context;
   mlir::OwningOpRef<mlir::ModuleOp> module;
   bool parallismAllowed;
   void loadFromString(std::string sql) override {
      initializeContext(context);

      mlir::OpBuilder builder(&context);

      mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
      if (!database) {
         error.emit() << "Database must be attached for parsing SQL";
      }
      frontend::sql::Parser translator(sql, *database, moduleOp);
      builder.setInsertionPointToStart(moduleOp.getBody());
      auto* queryBlock = new mlir::Block;
      std::vector<mlir::Type> returnTypes;
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
      module = moduleOp;
      parallismAllowed=translator.isParallelismAllowed();
   }
   void loadFromFile(std::string fileName) override {
      std::ifstream istream{fileName};
      if (!istream) {
         error.emit() << "Error can't load file " << fileName;
      }
      std::stringstream buffer;
      buffer << istream.rdbuf();
      std::string sqlQuery = buffer.str();
      loadFromString(sqlQuery);
   }
   mlir::ModuleOp* getModule() override {
      assert(module);
      return module.operator->();
   }
   bool isParallelismAllowed() override{
      return parallismAllowed;
   }
};
class BatchedSQLFrontend : public execution::Frontend {
   mlir::MLIRContext context;
   mlir::OwningOpRef<mlir::ModuleOp> module;
   void loadFromString(std::string sql) override {
      initializeContext(context);

      mlir::OpBuilder builder(&context);

      mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
      if (!database) {
         error.emit() << "Database must be attached for parsing SQL";
      }
      std::stringstream batched(sql);
      builder.setInsertionPointToStart(moduleOp.getBody());
      auto* queryBlock = new mlir::Block;
      std::vector<mlir::Type> returnTypes;
      size_t i = 0;
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(queryBlock);
         while (true) {
            std::stringstream query;
            std::string line;
            std::getline(batched, line);
            if (batched.eof()) {
               //exit from repl loop
               break;
            }
            while (true) {
               if (!batched.good()) {
                  if (batched.eof()) {
                     query << line << std::endl;
                  }
                  break;
               }
               query << line << std::endl;
               if (!line.empty() && line.find(';') == line.size() - 1) {
                  break;
               }
               std::getline(batched, line);
            }
            frontend::sql::Parser translator(query.str(), *database, moduleOp);
            auto val = translator.translate(builder);
            if (val.has_value()) {
               builder.create<mlir::subop::SetResultOp>(builder.getUnknownLoc(), i++, val.value());
            }
         }
         builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
      }
      mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {}));
      funcOp.getBody().push_back(queryBlock);
      module = moduleOp;
   }
   void loadFromFile(std::string fileName) override {
      std::ifstream istream{fileName};
      if (!istream) {
         error.emit() << "Error can't load file " << fileName;
      }
      std::stringstream buffer;
      buffer << istream.rdbuf();
      std::string sqlQuery = buffer.str();
      loadFromString(sqlQuery);
   }
   mlir::ModuleOp* getModule() override {
      assert(module);
      return module.operator->();
   }
};
} // namespace
std::unique_ptr<execution::Frontend> execution::createMLIRFrontend() {
   return std::make_unique<MLIRFrontend>();
}
std::unique_ptr<execution::Frontend> execution::createSQLFrontend() {
   return std::make_unique<SQLFrontend>();
}
std::unique_ptr<execution::Frontend> execution::createBatchedSQLFrontend() {
   return std::make_unique<BatchedSQLFrontend>();
}