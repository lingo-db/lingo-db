#include "lingodb/compiler/frontend/UDFImplementer.h"

#include "lingodb/catalog/FunctionCatalogEntry.h"
#include "lingodb/catalog/MLIRTypes.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/utility/Serialization.h"
#include "lingodb/utility/Setting.h"
#include "lingodb/compiler/runtime/PythonRuntime.h"

#include <lingodb/execution/Backend.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include <filesystem>

#include <llvm/Support/SourceMgr.h>
#include <dlfcn.h>

#include "json.h"

#include <lingodb/compiler/Dialect/util/UtilOps.h.inc>
#include <lingodb/compiler/Dialect/util/UtilOps.h.inc>
#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/Bytecode/BytecodeWriter.h>
namespace {
lingodb::utility::GlobalSetting<std::string> pythonBinary("system.hipy.python_binary", ".venv/bin/python3");
lingodb::utility::GlobalSetting<std::string> hipyDir("system.hipy.hipy_dir", "vendored/hipy");
lingodb::utility::GlobalSetting<std::string> cUDFCompilerDriver("system.compilation.c_udf_compiler_driver", "cc");


class PythonUDFImplementer : public lingodb::catalog::MLIRUDFImplementor {
   std::string functionName;
   std::string code;
   std::vector<lingodb::catalog::Type> argumentTypes;
   lingodb::catalog::Type returnType;

   public:
   PythonUDFImplementer(std::string functionName, std::string code, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType) : functionName(std::move(functionName)), code(std::move(code)), argumentTypes(std::move(argumentTypes)), returnType(std::move(returnType)) {}

   mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args, lingodb::catalog::Catalog* catalog) override {
      using namespace lingodb::compiler::dialect;
      mlir::Value moduleVal = builder.create<py_interp::CreateModule>(loc,py_interp::PyObjectType::get(builder.getContext()),builder.getStringAttr("udf_"+functionName), builder.getStringAttr(code));
      mlir::Value functionVal = builder.create<py_interp::GetAttr>(loc, py_interp::PyObjectType::get(builder.getContext()), moduleVal, builder.getStringAttr(functionName));

      std::vector<mlir::Value> values;
      std::vector<mlir::Value> isNull;
      for (auto arg : args) {
         values.push_back(arg);
         if (mlir::isa<db::NullableType>(arg.getType())) {
            isNull.push_back(builder.create<db::IsNullOp>(loc, arg));
         }
      }
      if (isNull.size() > 0) {
         auto allNotNull = builder.create<db::OrOp>(loc, isNull);
         auto* elseBlock = new mlir::Block;
         mlir::Type resType;
         {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(elseBlock);
            std::vector<mlir::Value> notNullValues;
            for (auto v : values) {
               notNullValues.push_back(mlir::isa<db::NullableType>(v.getType()) ? builder.create<db::NullableGetVal>(loc, mlir::cast<db::NullableType>(v.getType()).getType(), v) : v);
            }
            std::vector<mlir::Value> castedArgs;
            for (auto arg : notNullValues) {
               castedArgs.push_back(builder.create<py_interp::CastToPyObject>(loc, py_interp::PyObjectType::get(builder.getContext()), arg));
            }
            mlir::Value res = builder.create<py_interp::Call>(loc, py_interp::PyObjectType::get(builder.getContext()), functionVal, mlir::ValueRange(castedArgs),builder.getArrayAttr({}));
            mlir::Value nativeRes = builder.create<py_interp::CastFromPyObject>(loc, returnType.getMLIRTypeCreator()->createType(builder.getContext()), res);

            mlir::Value resNullable = builder.create<db::AsNullableOp>(loc, db::NullableType::get(nativeRes.getType()), nativeRes);
            resType = resNullable.getType();
            builder.create<mlir::scf::YieldOp>(loc, resNullable);
         }
         auto* thenBlock = new mlir::Block;

         {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(thenBlock);
            mlir::Value res = builder.create<db::NullOp>(loc, resType);
            builder.create<mlir::scf::YieldOp>(loc, res);
         }
         auto ifOp = builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{resType}, allNotNull, false);
         ifOp.getThenRegion().getBlocks().clear();
         ifOp.getThenRegion().push_back(thenBlock);
         ifOp.getElseRegion().getBlocks().clear();
         ifOp.getElseRegion().push_back(elseBlock);
         return ifOp.getResult(0);
      }

      std::vector<mlir::Value> castedArgs;
      for (auto arg : args) {
         castedArgs.push_back(builder.create<py_interp::CastToPyObject>(loc, py_interp::PyObjectType::get(builder.getContext()), arg));
      }
      mlir::Value res = builder.create<py_interp::Call>(loc, py_interp::PyObjectType::get(builder.getContext()), functionVal, mlir::ValueRange(castedArgs),builder.getArrayAttr({}));
      mlir::Value nativeRes = builder.create<py_interp::CastFromPyObject>(loc, returnType.getMLIRTypeCreator()->createType(builder.getContext()), res);

      return nativeRes;
   }
};
class CUDFImplementer : public lingodb::catalog::MLIRUDFImplementor {
   std::string functionName;
   std::string code;
   std::vector<lingodb::catalog::Type> argumentTypes;
   lingodb::catalog::Type returnType;

   public:
   CUDFImplementer(std::string functionName, std::string code, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType) : functionName(std::move(functionName)), code(std::move(code)), argumentTypes(std::move(argumentTypes)), returnType(std::move(returnType)) {}

   mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args, lingodb::catalog::Catalog* catalog) override {
      //Check if function has already been added before
      bool functionExists = lingodb::catalog::FunctionCatalogEntry::getUdfFunctions().contains(functionName);
      if (!functionExists) {
         auto currPath = std::filesystem::current_path().string();
         std::string pathToCFile = "";
         std::string pathToSOFile = "";
         try {
            bool soFileAlreadyExists = false;
            //Determine path to so file and check if so file already exists in db directory
            if (catalog->getDbDir().empty()) {
               char tempSoFileTemplate[] = "/tmp/c_udf_XXXXXX";
               int soFd = mkstemp(tempSoFileTemplate);
               if (soFd == -1) {
                  throw std::runtime_error("Failed to create temporary file.");
               }
               pathToSOFile = tempSoFileTemplate;
            } else {
               std::filesystem::create_directories(catalog->getDbDir() + "/udf");
               pathToSOFile = catalog->getDbDir() + "/udf/" + functionName + ".so";
               soFileAlreadyExists = std::filesystem::exists(pathToSOFile);
            }
            if (!soFileAlreadyExists) {
               char tempCFileTemplate[] = "/tmp/c_udf_XXXXXX";
               int fd = mkstemp(tempCFileTemplate);
               if (fd == -1) {
                  throw std::runtime_error("Failed to create temporary file.");
               }
               pathToCFile = std::string(tempCFileTemplate) + ".c";
               std::filesystem::rename(tempCFileTemplate, pathToCFile);

               std::ofstream tempFile(pathToCFile, std::ios::out | std::ios::trunc);

               tempFile << "#include <stdlib.h>\n"
                           "#include <string.h>\n"
                           "#include <stdint.h>\n"
                           "#include <stdbool.h>\n";
               tempFile << code;
               tempFile.close();

#ifdef __APPLE__
               std::string cmd = cUDFCompilerDriver.getValue() + std::string(" -march=native -shared -O3 -g -gdwarf-4 -fPIC -Wl, -I ") + std::string(SOURCE_DIR) + "/include " + pathToCFile + " -o " + pathToSOFile;
#else
               std::string cmd = cUDFCompilerDriver.getValue() + std::string(" -march=native -shared -O3 -g -gdwarf-4 -fPIC -Wl,--export-dynamic -I ") + std::string(SOURCE_DIR) + "/include " + pathToCFile + " -o " + pathToSOFile;
#endif
               auto* pPipe = ::popen(cmd.c_str(), "r");
               if (pPipe == nullptr) {
                  throw std::runtime_error("Could not compile query module statically (Pipe could not be opened)");
               }
               std::array<char, 256> buffer;
               std::string result;
               while (not std::feof(pPipe)) {
                  auto bytes = std::fread(buffer.data(), 1, buffer.size(), pPipe);
                  result.append(buffer.data(), bytes);
               }
               auto rc = ::pclose(pPipe);
               if (WEXITSTATUS(rc)) {
                  throw std::runtime_error("Could not compile query module statically (Pipe could not be closed)");
               }
            }
         } catch (std::exception& e) {
            throw std::runtime_error(std::string("Error during compilation of c udf: ") + e.what());
         }

         void* handle = dlopen(std::string(pathToSOFile).c_str(), RTLD_LAZY | RTLD_GLOBAL);
         const char* dlsymError = dlerror();
         if (dlsymError) {
            throw std::runtime_error(dlsymError);
         }
         assert(reinterpret_cast<lingodb::execution::mainFnType>(dlsym(handle, functionName.c_str())));
         lingodb::catalog::FunctionCatalogEntry::UDFHandle udfHandle{handle, dlsym(handle, functionName.c_str())};
         lingodb::catalog::FunctionCatalogEntry::getUdfFunctions().insert(std::pair(functionName, udfHandle));
      }

      mlir::func::FuncOp func = moduleOp.lookupSymbol<mlir::func::FuncOp>(functionName);
      if (!func || !functionExists) {
         std::vector<mlir::Type> argMLIRTypes;
         for (auto argType : argumentTypes) {
            argMLIRTypes.push_back(argType.getMLIRTypeCreator()->createType(builder.getContext()));
         }
         auto funcType = builder.getFunctionType(mlir::TypeRange(argMLIRTypes), {returnType.getMLIRTypeCreator()->createType(builder.getContext())});
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(moduleOp.getBody());
         func = builder.create<mlir::func::FuncOp>(loc, functionName, funcType);
         func.setPrivate();
      }
      return builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), func, args).getResult(0);
   }
};
class HiPyFunctionImplementer : public lingodb::catalog::MLIRUDFImplementor {
   std::string functionName;
   std::string byteCode;
   std::vector<lingodb::catalog::Type> argumentTypes;
   lingodb::catalog::Type returnType;

   mlir::Value handlePython(mlir::Value val, lingodb::catalog::Type type, mlir::OpBuilder& builder, mlir::Location loc) {
      using namespace lingodb::compiler::dialect;
      if (mlir::isa<py_interp::PyObjectType>(val.getType())) {
         val = builder.create<py_interp::CastFromPyObject>(loc, type.getMLIRTypeCreator()->createType(builder.getContext()), val);
      }
      return val;
   }
   public:
   HiPyFunctionImplementer(std::string functionName, std::string byteCode, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType) : functionName(std::move(functionName)), byteCode(std::move(byteCode)), argumentTypes(std::move(argumentTypes)), returnType(std::move(returnType)) {}
   mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args, lingodb::catalog::Catalog* catalog) override {
      using namespace lingodb::compiler::dialect;
      if (!moduleOp.lookupSymbol(functionName+"_"+functionName)) {
         auto parseBlock = new mlir::Block;
         std::unique_ptr<llvm::MemoryBuffer> buf =
            llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef(byteCode.data(), byteCode.size()));
         llvm::MemoryBufferRef bufRef = buf->getMemBufferRef();

         // ParserConfig constructed with the context.
         mlir::ParserConfig config(builder.getContext());

         // Read the bytecode into the module's top block. `buf` must remain alive
         // until readBytecodeFile returns (it does while we hold `buf`).
         if (mlir::readBytecodeFile(bufRef, parseBlock, config).failed()) {
            throw std::runtime_error("Failed to parse HiPy UDF bytecode");
         }
         //parse bytecode
         llvm::SourceMgr sourceMgr;
         std::vector<mlir::Operation*> toMove;
         for (auto& op : mlir::cast<mlir::ModuleOp>(parseBlock->front()).getOps()) {
            toMove.push_back(&op);
         }
         for (auto* op : toMove) {
            op->remove();
            if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
               funcOp.setSymVisibility("private");
            }
            moduleOp.getBody()->push_back(op);
         }
         delete parseBlock;
      }
      std::vector<mlir::Value> values;
      std::vector<mlir::Value> isNull;
      for (auto arg : args) {
         values.push_back(arg);
         if (mlir::isa<db::NullableType>(arg.getType())) {
            isNull.push_back(builder.create<db::IsNullOp>(loc, arg));
         }
      }
      if (isNull.size() > 0) {
         auto allNotNull = builder.create<db::OrOp>(loc, isNull);
         auto* elseBlock = new mlir::Block;
         mlir::Type resType;
         {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(elseBlock);
            std::vector<mlir::Value> notNullValues;
            for (auto v : values) {
               notNullValues.push_back(mlir::isa<db::NullableType>(v.getType()) ? builder.create<db::NullableGetVal>(loc, mlir::cast<db::NullableType>(v.getType()).getType(), v) : v);
            }
            auto func = mlir::cast<mlir::func::FuncOp>(moduleOp.lookupSymbol(functionName+"_"+functionName));
            auto res = handlePython(builder.create<mlir::func::CallOp>(loc, func, notNullValues).getResult(0), returnType, builder, loc);
            mlir::Value resNullable = builder.create<db::AsNullableOp>(loc, db::NullableType::get(res.getType()), res);

            resType = resNullable.getType();
            builder.create<mlir::scf::YieldOp>(loc, resNullable);
         }
         auto* thenBlock = new mlir::Block;

         {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(thenBlock);
            mlir::Value res = builder.create<db::NullOp>(loc, resType);
            builder.create<mlir::scf::YieldOp>(loc, res);
         }
         auto ifOp = builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{resType}, allNotNull, false);
         ifOp.getThenRegion().getBlocks().clear();
         ifOp.getThenRegion().push_back(thenBlock);
         ifOp.getElseRegion().getBlocks().clear();
         ifOp.getElseRegion().push_back(elseBlock);
         return ifOp.getResult(0);
      }
      auto func = mlir::cast<mlir::func::FuncOp>(moduleOp.lookupSymbol(functionName+"_"+functionName));

      return handlePython(builder.create<mlir::func::CallOp>(loc, func, values).getResult(0), returnType, builder, loc);
   }
};
} //namespace

namespace lingodb::compiler::frontend {

std::string compileHiPyUDF(std::string functionName, std::string code, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType, bool fallback) {
   mlir::MLIRContext context;
   lingodb::execution::initializeContext(context);
   mlir::OwningOpRef<mlir::ModuleOp> module;

   std::string tempFilePath, outputFilePath;
   try {
      // Create a temporary file path for the Python code
      char tempFileTemplate[] = "/tmp/python_udf_XXXXXX";
      int fd = mkstemp(tempFileTemplate);
      if (fd == -1) {
         throw std::runtime_error("Failed to create temporary file.");
      }
      tempFilePath = tempFileTemplate;

      // Write Python code to the temporary file
      std::ofstream tempFile(tempFilePath, std::ios::out | std::ios::trunc);
      if (!tempFile.is_open()) {
         throw std::runtime_error("Failed to open temporary file for writing Python code.");
      }
      tempFile << code;
      tempFile.close();

      // Create a temporary file path for the output
      char outputFileTemplate[] = "/tmp/python_udf_out_XXXXXX";
      int fdOut = mkstemp(outputFileTemplate);
      if (fdOut == -1) {
         throw std::runtime_error("Failed to create temporary output file.");
      }
      outputFilePath = outputFileTemplate;

      // Prepare JSON argument types
      nlohmann::json jsonArgs = nlohmann::json::array();
      for (const auto& argType : argumentTypes) {
         switch (argType.getTypeId()) {
            case lingodb::catalog::LogicalTypeId::BOOLEAN:
               jsonArgs.push_back("bool");
               break;
            case lingodb::catalog::LogicalTypeId::DOUBLE:
               jsonArgs.push_back("float");
               break;
            case lingodb::catalog::LogicalTypeId::STRING:
               jsonArgs.push_back("str");
               break;
            case lingodb::catalog::LogicalTypeId::INT:
               jsonArgs.push_back("int");
               break;
            case lingodb::catalog::LogicalTypeId::DATE:
              jsonArgs.push_back("date");
              break;
            default:
               throw std::runtime_error("Unsupported argument type for Python UDF: " + argType.toString());
         }
      }
      std::string jsonArgsStr = jsonArgs.dump();

      // Step 2: Invoke the external script
      std::ostringstream command;
      command << pythonBinary.getValue() << " "<<hipyDir.getValue()<<"/compile.py " << tempFilePath << " " << functionName << " '" << jsonArgsStr << "' "<< functionName <<" " << (fallback ? "fallback" : "nofallback") <<" "<< outputFilePath << " 2>&1";
      std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.str().c_str(), "r"), pclose);
      if (!pipe) {
         throw std::runtime_error("Failed to execute compile.py script.");
      }

      std::string output;
      char buffer[128];
      while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
         output += buffer;
      }

      int returnCode = pclose(pipe.release());
      if (returnCode != 0) {
         throw std::runtime_error("compile.py script failed with return code: " + std::to_string(returnCode) + "\nOutput:\n" + output);
      }
      // Step 3: Read back the output file
      llvm::SourceMgr sourceMgr;
      mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
         llvm::MemoryBuffer::getFileOrSTDIN(outputFilePath);
      if (std::error_code ec = fileOrErr.getError()) {
         throw std::runtime_error("Could not open input file: " + ec.message() + "\n");
      }

      // Parse the input mlir.
      sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
      module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
      if (!module) {
         throw std::runtime_error("Error can't load file " + outputFilePath + "\n");
      }
   } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Error during compilation: ") + e.what());
   }
   std::string bytecode;

   llvm::raw_string_ostream os(bytecode);
   if (mlir::writeBytecodeToFile(module->getOperation(), os).failed()) {
      throw std::runtime_error("Failed to write MLIR bytecode to string.");
   }
   return bytecode;
}
std::shared_ptr<catalog::MLIRUDFImplementor> getUDFImplementer(std::shared_ptr<catalog::FunctionCatalogEntry> entry) {
   switch (entry->getEntryType()) {
      case catalog::CatalogEntry::CatalogEntryType::C_FUNCTION_ENTRY: {
         return createCUDFImplementer(entry->getName(), entry->getCode(), entry->getArgumentTypes(), entry->getReturnType());
      }
      case catalog::CatalogEntry::CatalogEntryType::HIPY_FUNCTION_ENTRY: {
         return std::make_shared<HiPyFunctionImplementer>(entry->getName(), std::dynamic_pointer_cast<catalog::HiPyFunctionCatalogEntry>(entry)->getByteCode(), entry->getArgumentTypes(), entry->getReturnType());
      }
      case catalog::CatalogEntry::CatalogEntryType::PYTHON_FUNCTION_ENTRY: {
         return std::make_shared<PythonUDFImplementer>(entry->getName(), entry->getCode(), entry->getArgumentTypes(), entry->getReturnType());

      }
      default: throw std::runtime_error("getUDFImplementer: unknown catalog entry type");
   }
}
std::shared_ptr<catalog::MLIRUDFImplementor> createCUDFImplementer(std::string funcName, std::string cCode, std::vector<catalog::Type> argumentTypes, catalog::Type returnType) {
   return std::make_shared<CUDFImplementer>(funcName, cCode, argumentTypes, returnType);
}

} // namespace lingodb::compiler::frontend