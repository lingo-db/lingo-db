#include "llvm/Support/CommandLine.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/util/UtilDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Parser/AsmParserState.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <iostream>
#include <list>
#include <queue>

#include "../vendored/json.h"
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

mlir::Location dropNames(mlir::Location l) {
   if (auto namedLoc = l.dyn_cast<mlir::NameLoc>()) {
      return dropNames(namedLoc.getChildLoc());
   } else if (auto namedResultsLoc = l.dyn_cast<mlir::NamedResultsLoc>()) {
      return dropNames(namedResultsLoc.getChildLoc());
   }
   return l;
}
int main(int argc, char** argv) {
   cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::StandardOpsDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::arith::ArithmeticDialect>();

   registry.insert<mlir::scf::SCFDialect>();

   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::memref::MemRefDialect>();

   mlir::MLIRContext context;
   context.appendDialectRegistry(registry);

   llvm::SourceMgr sourceMgr;
   mlir::Block block;
   mlir::LocationAttr fileLoc;
   mlir::AsmParserState state;
   if (mlir::parseSourceFile(inputFilename, sourceMgr, &block, &context, &fileLoc, &state).failed()) {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      return 3;
   }
   nlohmann::json j;

   block.walk([&](mlir::Operation* op) {
      auto opDef = state.getOpDef(op);
      if (opDef) {

         if (auto fileLineLoc = dropNames(op->getLoc()).dyn_cast<mlir::FileLineColLoc>()) {
            auto loc1 = sourceMgr.getLineAndColumn(opDef->scopeLoc.Start);
            auto loc2 = sourceMgr.getLineAndColumn(opDef->scopeLoc.End);
            nlohmann::json mapping;
            nlohmann::json opStart;
            nlohmann::json opEnd;
            opStart["line"] = loc1.first;
            opStart["col"] = loc1.second;
            opEnd["line"] = loc2.first;
            opEnd["col"] = loc2.second;
            mapping["start"] = opStart;
            mapping["end"] = opEnd;
            mapping["created_from"] = nlohmann::json();
            mapping["created_from"]["line"] = fileLineLoc.getLine();
            mapping["created_from"]["col"] = fileLineLoc.getColumn();
            mapping["created_from"]["file"] = fileLineLoc.getFilename().str();
            mapping["operation"]=std::string(opDef->scopeLoc.Start.getPointer(),opDef->scopeLoc.End.getPointer());
            j.push_back(mapping);
         }
      }
   });
   std::cout << j.dump() << std::endl;
   return 0;
}