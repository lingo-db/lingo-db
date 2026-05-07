#include "lingodb/compiler/frontend/UDFImplementer.h"

#include "lingodb/catalog/FunctionCatalogEntry.h"
#include "lingodb/catalog/MLIRTypes.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#ifndef MLIR_DISABLED
// Local TableGen-generated dialect headers — only present in the full build.
// The standalone-query build defines MLIR_DISABLED and only ships the
// catalog/runtime/scheduler subset, where the Python UDF path isn't reachable
// (queries are pre-compiled at build time).
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowTypes.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h"
#endif
#include "lingodb/execution/Execution.h"
#include "lingodb/utility/Serialization.h"
#include "lingodb/utility/Setting.h"

#include <lingodb/execution/Backend.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include <filesystem>
#include <fstream>

#include <dlfcn.h>
namespace {
lingodb::utility::GlobalSetting<std::string> cUDFCompilerDriver("system.compilation.c_udf_compiler_driver", "cc");

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

#ifndef MLIR_DISABLED
class PythonUDFImplementer : public lingodb::catalog::MLIRUDFImplementor {
   std::string functionName;
   std::string code;
   std::vector<lingodb::catalog::Type> argumentTypes;
   lingodb::catalog::Type returnType;

   static std::string getPythonType(lingodb::catalog::Type type) {
      using namespace lingodb::catalog;
      switch (type.getTypeId()) {
         case LogicalTypeId::BOOLEAN: return "builtins.bool";
         case LogicalTypeId::INT: return "builtins.int";
         case LogicalTypeId::FLOAT: return "builtins.float";
         case LogicalTypeId::DOUBLE: return "builtins.float";
         case LogicalTypeId::STRING: return "builtins.str";
         case LogicalTypeId::DATE: return "datetime.date";
         default:
            throw std::runtime_error("Unsupported type for Python UDF: " + type.toString());
      }
   }

   public:
   PythonUDFImplementer(std::string functionName, std::string code, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType)
      : functionName(std::move(functionName)), code(std::move(code)), argumentTypes(std::move(argumentTypes)), returnType(std::move(returnType)) {}

   // Emit the body of a Python UDF call. Reference counting is handled
   // automatically by the SubOperator memory-management pass: every PyObject
   // value produced here is cleaned up via py_interp.dec_ref at end of scope.
   // The cached module returned by create_module is excluded from refcount
   // management (seedNotCounted in MemoryMgmtPass) so the interpreter cache
   // retains ownership.
   mlir::Value emitCall(mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange rawArgs) {
      using namespace lingodb::compiler::dialect;
      auto pyObjType = py_interp::PyObjectType::get(builder.getContext());

      mlir::Value moduleVal = builder.create<py_interp::CreateModule>(
         loc, pyObjType, builder.getStringAttr("udf_" + functionName), builder.getStringAttr(code));
      mlir::Value functionVal = builder.create<py_interp::GetAttr>(
         loc, pyObjType, moduleVal, builder.getStringAttr(functionName));

      std::vector<mlir::Value> castedArgs;
      for (auto [arg, argType] : llvm::zip(rawArgs, argumentTypes)) {
         castedArgs.push_back(builder.create<py_interp::CastToPyObject>(
            loc, pyObjType, arg, getPythonType(argType)));
      }
      mlir::Value res = builder.create<py_interp::Call>(
         loc, pyObjType, functionVal, mlir::ValueRange(castedArgs), builder.getArrayAttr({}));
      return builder.create<py_interp::CastFromPyObject>(
         loc, returnType.getMLIRTypeCreator()->createType(builder.getContext()), res, getPythonType(returnType));
   }

   mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args, lingodb::catalog::Catalog* catalog) override {
      using namespace lingodb::compiler::dialect;

      // Detect nullable args; if any are nullable, wrap the call in an SCF if so
      // null inputs propagate to a NULL result without ever entering the interpreter.
      std::vector<mlir::Value> isNull;
      for (auto arg : args) {
         if (mlir::isa<db::NullableType>(arg.getType())) {
            isNull.push_back(builder.create<db::IsNullOp>(loc, arg));
         }
      }
      if (isNull.empty()) {
         return emitCall(builder, loc, args);
      }

      auto anyNull = builder.create<db::OrOp>(loc, isNull);
      auto* elseBlock = new mlir::Block;
      mlir::Type resType;
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(elseBlock);
         std::vector<mlir::Value> notNullValues;
         for (auto v : args) {
            notNullValues.push_back(mlir::isa<db::NullableType>(v.getType()) ? builder.create<db::NullableGetVal>(loc, mlir::cast<db::NullableType>(v.getType()).getType(), v).getResult() : v);
         }
         mlir::Value nativeRes = emitCall(builder, loc, notNullValues);
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
      auto ifOp = builder.create<mlir::scf::IfOp>(loc, mlir::TypeRange{resType}, anyNull, false);
      ifOp.getThenRegion().getBlocks().clear();
      ifOp.getThenRegion().push_back(thenBlock);
      ifOp.getElseRegion().getBlocks().clear();
      ifOp.getElseRegion().push_back(elseBlock);
      return ifOp.getResult(0);
   }
};

// Tabular Python UDF: take an arrow.table input + already-cast scalar args,
// resolve the python function (cached module + getattr), call it, and cast
// the returned PyObject back to an arrow.table. The translator wraps this
// in the relalg.nested + materialize/scan scaffolding around it.
class PythonTableUDFImplementer : public lingodb::catalog::MLIRTableUDFImplementor {
   std::string functionName;
   std::string code;
   std::vector<lingodb::catalog::Type> scalarArgumentTypes;

   static std::string getPythonScalarType(lingodb::catalog::Type type) {
      using namespace lingodb::catalog;
      switch (type.getTypeId()) {
         case LogicalTypeId::BOOLEAN: return "builtins.bool";
         case LogicalTypeId::INT: return "builtins.int";
         case LogicalTypeId::FLOAT: return "builtins.float";
         case LogicalTypeId::DOUBLE: return "builtins.float";
         case LogicalTypeId::STRING: return "builtins.str";
         case LogicalTypeId::DATE: return "datetime.date";
         default:
            throw std::runtime_error("Unsupported scalar type for tabular Python UDF: " + type.toString());
      }
   }

   public:
   PythonTableUDFImplementer(std::string functionName, std::string code, std::vector<lingodb::catalog::Type> scalarArgumentTypes)
      : functionName(std::move(functionName)), code(std::move(code)), scalarArgumentTypes(std::move(scalarArgumentTypes)) {}

   mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::ValueRange inputArrowTables, mlir::ValueRange scalarArgs,
                            lingodb::catalog::Catalog* catalog) override {
      namespace py_interp = lingodb::compiler::dialect::py_interp;
      namespace arrow_dialect = lingodb::compiler::dialect::arrow;
      auto* mlirContext = builder.getContext();
      auto pyObjType = py_interp::PyObjectType::get(mlirContext);
      auto arrowTableType = arrow_dialect::TableType::get(mlirContext);

      // Resolve the python function (cached module + getattr).
      mlir::Value moduleVal = builder.create<py_interp::CreateModule>(
         loc, pyObjType,
         builder.getStringAttr("udf_" + functionName),
         builder.getStringAttr(code));
      mlir::Value functionVal = builder.create<py_interp::GetAttr>(
         loc, pyObjType, moduleVal, builder.getStringAttr(functionName));

      // Cast inputs to PyObjects: input tables first (in declaration order),
      // then scalar args.
      std::vector<mlir::Value> pyArgs;
      pyArgs.reserve(inputArrowTables.size() + scalarArgs.size());
      for (auto inputTable : inputArrowTables) {
         pyArgs.push_back(builder.create<py_interp::CastToPyObject>(loc, pyObjType, inputTable, "pyarrow.Table"));
      }
      for (auto [scalarArg, declaredType] : llvm::zip(scalarArgs, scalarArgumentTypes)) {
         pyArgs.push_back(builder.create<py_interp::CastToPyObject>(
            loc, pyObjType, scalarArg, getPythonScalarType(declaredType)));
      }

      mlir::Value pyResult = builder.create<py_interp::Call>(
         loc, pyObjType, functionVal, mlir::ValueRange(pyArgs), builder.getArrayAttr({}));

      mlir::Value arrowTableOut = builder.create<py_interp::CastFromPyObject>(
         loc, arrowTableType, pyResult, "pyarrow.Table");

      // TODO: emit dec_ref ops for pyResult / pyArgs / functionVal once the
      // SubOp execution-step splitter properly orders side-effect-only ops
      // after the last use of their operand. Skipping for now leaks
      // refcounts per query.
      return arrowTableOut;
   }
};
#endif // MLIR_DISABLED

} //namespace

namespace lingodb::compiler::frontend {
std::shared_ptr<catalog::MLIRUDFImplementor> getUDFImplementer(std::shared_ptr<catalog::FunctionCatalogEntry> entry) {
   switch (entry->getEntryType()) {
      case catalog::CatalogEntry::CatalogEntryType::C_FUNCTION_ENTRY: {
         return createCUDFImplementer(entry->getName(), entry->getCode(), entry->getArgumentTypes(), entry->getReturnType());
      }
      case catalog::CatalogEntry::CatalogEntryType::PYTHON_FUNCTION_ENTRY: {
#ifdef MLIR_DISABLED
         throw std::runtime_error("Python UDFs are not available in standalone-query builds (MLIR_DISABLED)");
#else
         return createPythonUDFImplementer(entry->getName(), entry->getCode(), entry->getArgumentTypes(), entry->getReturnType());
#endif
      }
      default: throw std::runtime_error("getUDFImplementer: unknown catalog entry type");
   }
}
std::shared_ptr<catalog::MLIRUDFImplementor> createCUDFImplementer(std::string funcName, std::string cCode, std::vector<catalog::Type> argumentTypes, catalog::Type returnType) {
   return std::make_shared<CUDFImplementer>(funcName, cCode, argumentTypes, returnType);
}
std::shared_ptr<catalog::MLIRUDFImplementor> createPythonUDFImplementer(std::string funcName, std::string pyCode, std::vector<catalog::Type> argumentTypes, catalog::Type returnType) {
#ifdef MLIR_DISABLED
   throw std::runtime_error("Python UDFs are not available in standalone-query builds (MLIR_DISABLED)");
#else
   return std::make_shared<PythonUDFImplementer>(funcName, pyCode, argumentTypes, returnType);
#endif
}

std::shared_ptr<catalog::MLIRTableUDFImplementor> getTableUDFImplementer(std::shared_ptr<catalog::TableFunctionCatalogEntry> entry) {
   const auto& language = entry->getLanguage();
   if (language == "python") {
      return createPythonTableUDFImplementer(entry->getName(), entry->getCode(), entry->getArgumentTypes());
   }
   throw std::runtime_error("getTableUDFImplementer: unsupported language '" + language + "'");
}
std::shared_ptr<catalog::MLIRTableUDFImplementor> createPythonTableUDFImplementer(std::string funcName, std::string pyCode, std::vector<catalog::Type> scalarArgumentTypes) {
#ifdef MLIR_DISABLED
   throw std::runtime_error("Tabular Python UDFs are not available in standalone-query builds (MLIR_DISABLED)");
#else
   return std::make_shared<PythonTableUDFImplementer>(funcName, pyCode, scalarArgumentTypes);
#endif
}

} // namespace lingodb::compiler::frontend