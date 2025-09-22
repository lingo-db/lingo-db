#include "lingodb/execution/Frontend.h"

#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/Passes.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/frontend/driver.h"
#include "lingodb/compiler/frontend/sql_analyzer.h"
#include "lingodb/compiler/frontend/sql_mlir_translator.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <iostream>
void lingodb::execution::initializeContext(mlir::MLIRContext& context) {
   using namespace lingodb::compiler::dialect;
   mlir::DialectRegistry registry;
   registry.insert<mlir::BuiltinDialect>();
   registry.insert<relalg::RelAlgDialect>();
   registry.insert<tuples::TupleStreamDialect>();
   registry.insert<subop::SubOperatorDialect>();
   registry.insert<db::DBDialect>();
   registry.insert<lingodb::compiler::dialect::arrow::ArrowDialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithDialect>();
   registry.insert<mlir::cf::ControlFlowDialect>();
   registry.insert<mlir::DLTIDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   // TODO: should we make this optional? Would require a cmake flag that deactivates the LLVM backend.
   registry.insert<mlir::LLVM::LLVMDialect>();

#if GPU_ENABLED == 1
   registry.insert<mlir::async::AsyncDialect>();
   registry.insert<mlir::gpu::GPUDialect>();
   mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
#endif
   mlir::registerAllExtensions(registry);
   // TODO: same as above, should we make this optional?
   mlir::registerAllToLLVMIRTranslations(registry);
   context.appendDialectRegistry(registry);
   context.loadAllAvailableDialects();
   context.loadDialect<relalg::RelAlgDialect>();
   context.disableMultithreading();
}
namespace {

class MLIRFrontend : public lingodb::execution::Frontend {
   mlir::MLIRContext context;
   mlir::OwningOpRef<mlir::ModuleOp> module;
   void loadFromFile(std::string fileName) override {
      lingodb::execution::initializeContext(context);
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
      lingodb::execution::initializeContext(context);
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

class SQLFrontend : public lingodb::execution::Frontend {
   mlir::MLIRContext context;
   mlir::OwningOpRef<mlir::ModuleOp> module;
   bool parallismAllowed;
   void load(std::string fileOrDirect) {
      lingodb::execution::initializeContext(context);
      driver drv;
      try {
         if (!drv.parse(fileOrDirect)) {
            auto results = drv.result;

            if (results.empty() || results.size() > 1) {
               error.emit() << "Error during parsing: Only one statement allowed";
               return;
            }
            auto sqlContext = std::make_shared<lingodb::analyzer::SQLContext>();
            sqlContext->catalog = catalog;
            lingodb::analyzer::SQLQueryAnalyzer analyzer{catalog};
            drv.result[0] = analyzer.canonicalizeAndAnalyze(drv.result[0], sqlContext);

            mlir::OpBuilder builder(&context);

            mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
            lingodb::translator::SQLMlirTranslator translator{moduleOp, catalog};
            builder.setInsertionPointToStart(moduleOp.getBody());
            auto* queryBlock = new mlir::Block;
            std::vector<mlir::Type> returnTypes;
            {
               mlir::OpBuilder::InsertionGuard guard(builder);
               builder.setInsertionPointToStart(queryBlock);
               auto val = translator.translateStart(builder, drv.result[0], sqlContext);
               if (val.has_value()) {
                  builder.create<lingodb::compiler::dialect::subop::SetResultOp>(builder.getUnknownLoc(), 0, val.value());
               }
               builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
            }
            mlir::func::FuncOp funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", builder.getFunctionType({}, {}));
            funcOp.getBody().push_back(queryBlock);
            module = moduleOp;
            parallismAllowed = analyzer.parallelismAllowed;
            timing.emplace("frontEnd", analyzer.getTiming() + translator.getTiming());

            } else {
               error.emit() << "Error during parsing";
            }
      } catch (lingodb::frontend_error& e) {
         error.emit() << e.what();

      }
   }

   void loadFromString(std::string sql) override {
      sql = ":" + sql;
      load(sql);
   }
   void loadFromFile(std::string fileName) override {
      load(fileName);
   }


   mlir::ModuleOp* getModule() override {
      assert(module);
      return module.operator->();
   }
   bool isParallelismAllowed() override {
      return parallismAllowed;
   }
};


} // namespace
std::unique_ptr<lingodb::execution::Frontend> lingodb::execution::createMLIRFrontend() {
   return std::make_unique<MLIRFrontend>();
}
std::unique_ptr<lingodb::execution::Frontend> lingodb::execution::createSQLFrontend() {
   return std::make_unique<SQLFrontend>();
}