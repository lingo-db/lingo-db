#include "arrow/util/decimal.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/util/Passes.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/Support/ErrorOr.h>
#include <iomanip>
#include <iostream>
#include <time.h>

namespace cl = llvm::cl;
/*
 namespace {

Status RunMain(int argc, char** argv) {
   const char* csv_filename = "/home/michael/master-thesis/related-code/tpl_tables/tables/lineitem.data";
   const char* arrow_filename = "test.arrow";

   std::cerr << "* Reading CSV file '" << csv_filename << "' into table" << std::endl;
   ARROW_ASSIGN_OR_RAISE(auto input_file,
                         arrow::io::ReadableFile::Open(csv_filename));
   auto convert_options=arrow::csv::ConvertOptions();
   convert_options.column_types["5"]=arrow::decimal128(15,2);
   ARROW_ASSIGN_OR_RAISE(
      auto csv_reader,
      arrow::csv::TableReader::Make(arrow::default_memory_pool(),
                                    input_file,
                                    arrow::csv::ReadOptions::Defaults(),
                                    arrow::csv::ParseOptions::Defaults(),
                                    convert_options));
   ARROW_ASSIGN_OR_RAISE(auto table, csv_reader->Read());

   std::cerr << "* Read table:" << std::endl;
   ARROW_RETURN_NOT_OK(arrow::PrettyPrint(*table, {}, &std::cerr));

   std::cerr << "* Writing table into Arrow IPC file '" << arrow_filename << "'" << std::endl;
   ARROW_ASSIGN_OR_RAISE(auto output_file,
                         arrow::io::FileOutputStream::Open(arrow_filename));
   ARROW_ASSIGN_OR_RAISE(auto batch_writer,
                         arrow::ipc::MakeStreamWriter(output_file,
                                                      table->schema()));
   ARROW_RETURN_NOT_OK(batch_writer->WriteTable(*table));
   ARROW_RETURN_NOT_OK(batch_writer->Close());
   std::shared_ptr<arrow::Array> arr = table->column(5)->chunk(0);
   auto dec_array = std::static_pointer_cast<arrow::Decimal128Array>(arr);
   arrow::StringType::offset_type len;
   auto ptr = dec_array->GetValue(0);
   arrow::BasicDecimal128 basicDecimal128(ptr);
   arrow::Decimal128 dec128(basicDecimal128);
   std::cout<<dec128.ToString(2)<<std::endl;

   return Status::OK();
}
Status RunMain2(int argc, char** argv) {
   const char* arrow_filename = "test.arrow";

   ARROW_ASSIGN_OR_RAISE(auto input_file, arrow::io::ReadableFile::Open(arrow_filename))
   ARROW_ASSIGN_OR_RAISE(auto batch_reader, arrow::ipc::RecordBatchStreamReader::Open(input_file));
   ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatchReader(batch_reader.get()));
   ARROW_RETURN_NOT_OK(arrow::PrettyPrint(*table, {}, &std::cerr));
   std::shared_ptr<arrow::Array> arr = table->column(8)->chunk(0);
   auto str_arr = std::static_pointer_cast<arrow::StringArray>(arr);
   arrow::StringType::offset_type len;
   auto ptr = str_arr->GetValue(0, &len);
   std::string str((char*)ptr,len);
   std::cout<<str<<std::endl;
   return Status::OK();
}

} // namespace
 */

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int loadMLIR(mlir::MLIRContext& context, mlir::OwningModuleRef& module) {
   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
   if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
      return -1;
   }

   // Parse the input mlir.
   llvm::SourceMgr sourceMgr;
   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
   module = mlir::parseSourceFile(sourceMgr, &context);
   if (!module) {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      return 3;
   }
   return 0;
}

int runJit(mlir::ModuleOp module) {
   // Initialize LLVM targets.
   llvm::InitializeNativeTarget();
   llvm::InitializeNativeTargetAsmPrinter();

   // An optimization pipeline to use within the execution engine.
   auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/false ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

   // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
   // the module.
   auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline);
   assert(maybeEngine && "failed to construct an execution engine");
   auto& engine = maybeEngine.get();

   //int32_t res=0;
   std::vector<void*> args = {}; // {&res};
   // Invoke the JIT-compiled function.
   auto invocationResult = engine->invokePacked("main", args);
   if (invocationResult) {
      llvm::errs() << "JIT invocation failed\n";
      return -1;
   }

   return 0;
}
int dumpLLVMIR(mlir::ModuleOp module) {
   // Convert the module to LLVM IR in a new LLVM IR context.
   llvm::LLVMContext llvmContext;
   auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
   if (!llvmModule) {
      llvm::errs() << "Failed to emit LLVM IR\n";
      return -1;
   }

   // Initialize LLVM targets.
   llvm::InitializeNativeTarget();
   llvm::InitializeNativeTargetAsmPrinter();
   mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

   /// Optionally run an optimization pipeline over the llvm module.
   auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/false ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
   if (auto err = optPipeline(llvmModule.get())) {
      llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
      return -1;
   }
   llvm::errs() << *llvmModule << "\n";
   return 0;
}
namespace {
struct ToLLVMLoweringPass
   : public mlir::PassWrapper<ToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void ToLLVMLoweringPass::runOnOperation() {
   // The first thing to define is the conversion target. This will define the
   // final target for this lowering. For this lowering, we are only targeting
   // the LLVM dialect.
   mlir::LLVMConversionTarget target(getContext());
   target.addLegalOp<mlir::ModuleOp>();

   // During this lowering, we will also be lowering the MemRef types, that are
   // currently being operated on, to a representation in LLVM. To perform this
   // conversion we use a TypeConverter as part of the lowering. This converter
   // details how one type maps to another. This is necessary now that we will be
   // doing more complicated lowerings, involving loop region arguments.
   mlir::LowerToLLVMOptions options(&getContext());
   options.useBarePtrCallConv = true;
   mlir::LLVMTypeConverter typeConverter(&getContext(), options);

   // Now that the conversion target has been defined, we need to provide the
   // patterns used for lowering. At this point of the compilation process, we
   // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
   // are already exists a set of patterns to transform `affine` and `std`
   // dialects. These patterns lowering in multiple stages, relying on transitive
   // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
   // patterns must be applied to fully transform an illegal operation into a
   // set of legal ones.
   mlir::RewritePatternSet patterns(&getContext());
   populateAffineToStdConversionPatterns(patterns);
   populateLoopToStdConversionPatterns(patterns);
   mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
   populateStdToLLVMConversionPatterns(typeConverter, patterns);
   // We want to completely lower to LLVM, so we use a `FullConversion`. This
   // ensures that only legal operations will remain after the conversion.
   auto module = getOperation();
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
   return std::make_unique<ToLLVMLoweringPass>();
}
extern "C" __attribute__((visibility("default"))) void dumpInt(bool null, int64_t val) {
   if (null) {
      std::cout << "int(NULL)" << std::endl;
   } else {
      std::cout << "int(" << val << ")" << std::endl;
   }
}
extern "C" __attribute__((visibility("default"))) void dumpBool(bool null, bool val) {
   if (null) {
      std::cout << "bool(NULL)" << std::endl;
   } else {
      std::cout << "bool(" << std::boolalpha << val << ")" << std::endl;
   }
}
extern "C" __attribute__((visibility("default"))) void dumpDecimal(bool null, uint64_t low, uint64_t high, int32_t scale) {
   if (null) {
      std::cout << "decimal(NULL)" << std::endl;
   } else {
      arrow::Decimal128 decimalrep(arrow::BasicDecimal128(high, low));
      std::cout << "decimal(" << decimalrep.ToString(scale) << ")" << std::endl;
   }
}
extern "C" __attribute__((visibility("default"))) void dumpDate(bool null, uint32_t date) {
   if (null) {
      std::cout << "date(NULL)" << std::endl;
   } else {
      time_t time = date;
      tm tmStruct;
      time *= 24 * 60 * 60;
      auto* x = gmtime_r(&time, &tmStruct);

      std::cout << "date(" << (x->tm_year + 1900) << "-" << std::setw(2) << std::setfill('0') << (x->tm_mon + 1) << "-" << std::setw(2) << std::setfill('0') << x->tm_mday << ")" << std::endl;
   }
}
extern "C" __attribute__((visibility("default"))) void dumpTimestamp(bool null, uint64_t date) {
   if (null) {
      std::cout << "timestamp(NULL)" << std::endl;
   } else {
      time_t time = date;
      tm tmStruct;
      auto* x = gmtime_r(&time, &tmStruct);
      std::cout << "timestamp(" << (x->tm_year + 1900) << "-" << std::setw(2) << std::setfill('0') << (x->tm_mon + 1) << "-" << std::setw(2) << std::setfill('0') << x->tm_mday << " " << std::setw(2) << std::setfill('0') << x->tm_hour << ":" << std::setw(2) << std::setfill('0') << x->tm_min << ":" << std::setw(2) << std::setfill('0') << x->tm_sec << ")" << std::endl;
   }
}
extern "C" __attribute__((visibility("default"))) void dumpInterval(bool null, uint64_t interval) {
   if (null) {
      std::cout << "interval(NULL)" << std::endl;
   } else {
      std::cout << "interval(" << interval << ")" << std::endl;
   }
}
extern "C" __attribute__((visibility("default"))) void dumpFloat(bool null, double val) {
   if (null) {
      std::cout << "float(NULL)" << std::endl;
   } else {
      std::cout << "float(" << val << ")" << std::endl;
   }
}
extern "C" __attribute__((visibility("default"))) void dumpString(bool null, char* ptr, size_t len) {
   if (null) {
      std::cout << "string(NULL)" << std::endl;
   } else {
      std::cout << "string(\"" << std::string(ptr, len) << "\")" << std::endl;
   }
}

int main(int argc, char** argv) {
   cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::StandardOpsDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::memref::MemRefDialect>();

   mlir::MLIRContext context;
   context.appendDialectRegistry(registry);
   mlir::registerLLVMDialectTranslation(context);
   mlir::OwningModuleRef module;
   llvm::SourceMgr sourceMgr;
   llvm::DebugFlag = false;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
   if (int error = loadMLIR(context, module))
      return error;
   mlir::PassManager pm(&context);
   pm.addPass(mlir::db::createLowerToStdPass());
   if (mlir::failed(pm.run(module.get()))) {
      return 1;
   }
   module->dump();
   mlir::PassManager pm2(&context);
   pm2.addPass(mlir::createLowerToCFGPass());
   pm2.addPass(createLowerToLLVMPass());
   if (mlir::failed(pm2.run(module.get()))) {
      return 1;
   }
   module->dump();
   dumpLLVMIR(module.get());
   runJit(module.get());
   return 0;
}