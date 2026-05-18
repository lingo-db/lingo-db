#include "lingodb/compiler/frontend/UDFImplementer.h"

#include "lingodb/catalog/FunctionCatalogEntry.h"
#include "lingodb/catalog/MLIRTypes.h"
#include "lingodb/catalog/TableCatalogEntry.h"
#ifndef MLIR_DISABLED
// Local TableGen-generated dialect headers — only present in the full build.
// The standalone-query build defines MLIR_DISABLED and only ships the
// catalog/runtime/scheduler subset, where the Python UDF path isn't reachable
// (queries are pre-compiled at build time).
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowTypes.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpDialect.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#endif
#include "lingodb/execution/Execution.h"
#include "lingodb/utility/Serialization.h"
#include "lingodb/utility/Setting.h"

#include <lingodb/execution/Backend.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>
#ifndef MLIR_DISABLED
#include <llvm/Support/raw_ostream.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#endif

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>

#include <dlfcn.h>
#include <unistd.h>
namespace {
lingodb::utility::GlobalSetting<std::string> cUDFCompilerDriver("system.compilation.c_udf_compiler_driver", "cc");
// Python interpreter used to run vendored/hipy/compile.py at CREATE FUNCTION
// time. Must be an environment with hipy's dependencies installed; overridable
// via LINGODB_HIPY_PYTHON_BINARY.
lingodb::utility::GlobalSetting<std::string> hipyPythonBinary("system.hipy.python_binary", ".venv/bin/python3");
// Directory holding the hipy checkout (with compile.py at its root).
lingodb::utility::GlobalSetting<std::string> hipyDir("system.hipy.hipy_dir", "vendored/hipy");

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
// Fully-qualified Python type name for a catalog type, as understood by the
// py_interp cast ops. Shared by the runtime-Python and hipy implementers.
std::string catalogTypeToPythonType(lingodb::catalog::Type type) {
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

// Wrap a scalar UDF call in a null-propagation guard. If no argument is
// nullable, `emitCall` runs directly. Otherwise the call is placed in the
// else-branch of an scf.if: when any input is null the result is NULL, and the
// UDF only ever sees unwrapped non-null values. `emitCall` receives the
// non-null values and must return a non-nullable native result.
mlir::Value emitScalarUDFCallWithNullGuard(
   mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args,
   const std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>& emitCall) {
   using namespace lingodb::compiler::dialect;
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

class PythonUDFImplementer : public lingodb::catalog::MLIRUDFImplementor {
   std::string functionName;
   std::string code;
   std::vector<lingodb::catalog::Type> argumentTypes;
   lingodb::catalog::Type returnType;

   static std::string getPythonType(lingodb::catalog::Type type) { return catalogTypeToPythonType(type); }

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
      // The null guard keeps null inputs from ever entering the interpreter.
      return emitScalarUDFCallWithNullGuard(
         builder, loc, args,
         [this](mlir::OpBuilder& b, mlir::Location l, mlir::ValueRange a) { return emitCall(b, l, a); });
   }
};

// hipy UDF: the Python source was compiled to a LingoDB MLIR module ahead of
// time (see compileHiPyUDF) and stored as MLIR bytecode. At query translation
// the bytecode is parsed once and its functions spliced into the query module
// as private symbols; the UDF is then invoked with a plain func.call. In
// `hipy_fallback` mode the compiled function may return a py_interp PyObject
// for parts hipy could not lower natively; that is cast back here.
class HiPyFunctionImplementer : public lingodb::catalog::MLIRUDFImplementor {
   std::string functionName;
   std::string byteCode;
   std::vector<lingodb::catalog::Type> argumentTypes;
   lingodb::catalog::Type returnType;

   // hipy emits the entry point suffixed with the UDF name (compile.py is
   // invoked with function_suffix == functionName) so several hipy UDFs can
   // coexist in one query module without symbol clashes.
   std::string mangledName() const { return functionName + "_" + functionName; }

   // Parse the stored bytecode once and move its top-level ops into the query
   // module, making functions private. Guarded on the entry symbol so a UDF
   // used multiple times in one query is spliced only once. parseSourceString
   // transparently accepts MLIR bytecode as well as textual MLIR.
   void ensureSpliced(mlir::ModuleOp& moduleOp) {
      if (moduleOp.lookupSymbol(mangledName())) {
         return;
      }
      mlir::OwningOpRef<mlir::ModuleOp> parsed =
         mlir::parseSourceString<mlir::ModuleOp>(byteCode, moduleOp.getContext());
      if (!parsed) {
         throw std::runtime_error("Failed to parse hipy UDF bytecode for '" + functionName + "'");
      }
      std::vector<mlir::Operation*> toMove;
      for (auto& op : parsed->getOps()) {
         toMove.push_back(&op);
      }
      for (auto* op : toMove) {
         op->remove();
         if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
            funcOp.setSymVisibility("private");
         }
         moduleOp.getBody()->push_back(op);
      }
   }

   public:
   HiPyFunctionImplementer(std::string functionName, std::string byteCode, std::vector<lingodb::catalog::Type> argumentTypes, lingodb::catalog::Type returnType)
      : functionName(std::move(functionName)), byteCode(std::move(byteCode)), argumentTypes(std::move(argumentTypes)), returnType(std::move(returnType)) {}

   mlir::Value callFunction(mlir::ModuleOp& moduleOp, mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args, lingodb::catalog::Catalog* catalog) override {
      using namespace lingodb::compiler::dialect;
      ensureSpliced(moduleOp);
      auto func = mlir::cast<mlir::func::FuncOp>(moduleOp.lookupSymbol(mangledName()));
      return emitScalarUDFCallWithNullGuard(
         builder, loc, args,
         [&](mlir::OpBuilder& b, mlir::Location l, mlir::ValueRange a) -> mlir::Value {
            mlir::Value res = b.create<mlir::func::CallOp>(l, func, a).getResult(0);
            // hipy_fallback may yield a PyObject for un-lowered parts; bring
            // it back to the declared native return type.
            if (mlir::isa<py_interp::PyObjectType>(res.getType())) {
               res = b.create<py_interp::CastFromPyObject>(
                  l, returnType.getMLIRTypeCreator()->createType(b.getContext()), res, catalogTypeToPythonType(returnType));
            }
            return res;
         });
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

      // Manual cleanup: result first, then arg conversions, then function attr.
      // The cached module returned by create_module is owned by the
      // interpreter cache, so we must NOT decref it.
      builder.create<py_interp::DecRef>(loc, pyResult);
      for (auto a : pyArgs) {
         builder.create<py_interp::DecRef>(loc, a);
      }
      builder.create<py_interp::DecRef>(loc, functionVal);
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
      case catalog::CatalogEntry::CatalogEntryType::HIPY_FUNCTION_ENTRY: {
#ifdef MLIR_DISABLED
         throw std::runtime_error("hipy UDFs are not available in standalone-query builds (MLIR_DISABLED)");
#else
         auto hipyEntry = std::static_pointer_cast<catalog::HiPyFunctionCatalogEntry>(entry);
         return std::make_shared<HiPyFunctionImplementer>(hipyEntry->getName(), hipyEntry->getByteCode(), hipyEntry->getArgumentTypes(), hipyEntry->getReturnType());
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

#ifdef MLIR_DISABLED
std::string compileHiPyUDF(std::string, std::string, std::vector<catalog::Type>, catalog::Type, bool) {
   throw std::runtime_error("hipy UDFs are not available in standalone-query builds (MLIR_DISABLED)");
}
#else
namespace {
// hipy's compile.py argument-type vocabulary (see vendored/hipy/compile.py).
std::string catalogTypeToHiPyArgType(const catalog::Type& argType) {
   switch (argType.getTypeId()) {
      case catalog::LogicalTypeId::INT: return "int";
      case catalog::LogicalTypeId::FLOAT: return "float";
      case catalog::LogicalTypeId::DOUBLE: return "float";
      case catalog::LogicalTypeId::STRING: return "str";
      case catalog::LogicalTypeId::DATE: return "date";
      default:
         throw std::runtime_error("Unsupported argument type for hipy UDF: " + argType.toString());
   }
}
// Best-effort temp-file removal; never throws.
void removeIfExists(const std::string& path) {
   if (!path.empty()) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
   }
}
// Dialect set hipy's compile.py can emit, mirroring the non-LLVM dialects of
// execution::initializeContext. compileHiPyUDF parses against its own context
// (it runs outside query compilation); the LLVM/GPU dialects are unneeded
// because hipy emits only high-level LingoDB + standard ops.
void registerHiPyDialects(mlir::DialectRegistry& registry) {
   using namespace lingodb::compiler::dialect;
   registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect, mlir::arith::ArithDialect,
                   mlir::cf::ControlFlowDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                   mlir::DLTIDialect>();
   registry.insert<relalg::RelAlgDialect, tuples::TupleStreamDialect, subop::SubOperatorDialect,
                   db::DBDialect, arrow::ArrowDialect, util::UtilDialect, py_interp::PyInterpDialect>();
}
} // namespace

std::string compileHiPyUDF(std::string functionName, std::string code,
                           std::vector<catalog::Type> argumentTypes,
                           catalog::Type /*returnType*/, bool fallback) {
   std::string pythonFilePath, outputFilePath;
   try {
      // Write the UDF source to a temp file with a .py suffix (compile.py
      // imports it as a module and would otherwise copy it to add the suffix).
      char pythonTemplate[] = "/tmp/hipy_udf_XXXXXX";
      int pythonFd = mkstemp(pythonTemplate);
      if (pythonFd == -1) {
         throw std::runtime_error("Failed to create temporary file for hipy UDF source.");
      }
      close(pythonFd);
      pythonFilePath = std::string(pythonTemplate) + ".py";
      std::filesystem::rename(pythonTemplate, pythonFilePath);
      {
         std::ofstream pythonFile(pythonFilePath, std::ios::out | std::ios::trunc);
         if (!pythonFile.is_open()) {
            throw std::runtime_error("Failed to open temporary file for hipy UDF source.");
         }
         pythonFile << code;
      }

      char outputTemplate[] = "/tmp/hipy_udf_out_XXXXXX";
      int outputFd = mkstemp(outputTemplate);
      if (outputFd == -1) {
         throw std::runtime_error("Failed to create temporary output file for hipy UDF.");
      }
      close(outputFd);
      outputFilePath = outputTemplate;

      // hipy's compile.py expects the argument types as a JSON array.
      std::string jsonArgs = "[";
      for (size_t i = 0; i < argumentTypes.size(); i++) {
         jsonArgs += (i ? ",\"" : "\"") + catalogTypeToHiPyArgType(argumentTypes[i]) + "\"";
      }
      jsonArgs += "]";

      // compile.py <src> <function> <arg_types_json> <function_suffix> <fallback> <output>
      // function_suffix == functionName so several hipy UDFs can coexist in one
      // query module (see HiPyFunctionImplementer::mangledName).
      std::ostringstream command;
      command << hipyPythonBinary.getValue() << " "
              << hipyDir.getValue() << "/compile.py "
              << pythonFilePath << " " << functionName << " '" << jsonArgs << "' "
              << functionName << " " << (fallback ? "fallback" : "nofallback") << " "
              << outputFilePath << " 2>&1";

      std::string output;
      {
         std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.str().c_str(), "r"), pclose);
         if (!pipe) {
            throw std::runtime_error("Failed to execute hipy compile.py.");
         }
         std::array<char, 256> buffer;
         while (std::fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            output += buffer.data();
         }
         int returnCode = pclose(pipe.release());
         if (returnCode != 0) {
            throw std::runtime_error("hipy compile.py failed (exit " + std::to_string(returnCode) + "):\n" + output);
         }
      }

      // compile.py writes the textual MLIR module to the output file. Parse it
      // and re-emit as MLIR bytecode — that compact form is what gets stored in
      // the catalog and spliced back in at query translation.
      std::ifstream resultFile(outputFilePath, std::ios::in | std::ios::binary);
      if (!resultFile.is_open()) {
         throw std::runtime_error("Could not read hipy compile.py output.");
      }
      std::ostringstream resultStream;
      resultStream << resultFile.rdbuf();
      std::string compiledMLIR = resultStream.str();
      if (compiledMLIR.empty()) {
         throw std::runtime_error("hipy compile.py produced an empty module.\nOutput:\n" + output);
      }

      mlir::DialectRegistry registry;
      registerHiPyDialects(registry);
      mlir::MLIRContext context(registry);
      context.loadAllAvailableDialects();
      mlir::OwningOpRef<mlir::ModuleOp> module =
         mlir::parseSourceString<mlir::ModuleOp>(compiledMLIR, &context);
      if (!module) {
         throw std::runtime_error("Could not parse MLIR produced by hipy compile.py.");
      }
      std::string byteCode;
      llvm::raw_string_ostream os(byteCode);
      if (mlir::writeBytecodeToFile(module->getOperation(), os).failed()) {
         throw std::runtime_error("Failed to serialize hipy UDF module to bytecode.");
      }
      removeIfExists(pythonFilePath);
      removeIfExists(outputFilePath);
      return byteCode;
   } catch (...) {
      removeIfExists(pythonFilePath);
      removeIfExists(outputFilePath);
      throw;
   }
}
#endif // MLIR_DISABLED

} // namespace lingodb::compiler::frontend