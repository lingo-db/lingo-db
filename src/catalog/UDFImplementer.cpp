#include "lingodb/compiler/frontend/UDFImplementer.h"

#include "lingodb/catalog/FunctionCatalogEntry.h"
#include "lingodb/catalog/MLIRTypes.h"
#include "lingodb/catalog/TableCatalogEntry.h"
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

} //namespace

namespace lingodb::compiler::frontend {
std::shared_ptr<catalog::MLIRUDFImplementor> getUDFImplementer(std::shared_ptr<catalog::FunctionCatalogEntry> entry) {
   switch (entry->getEntryType()) {
      case catalog::CatalogEntry::CatalogEntryType::C_FUNCTION_ENTRY: {
         return createCUDFImplementer(entry->getName(), entry->getCode(), entry->getArgumentTypes(), entry->getReturnType());
      }
      default: throw std::runtime_error("getUDFImplementer: unknown catalog entry type");
   }
}
std::shared_ptr<catalog::MLIRUDFImplementor> createCUDFImplementer(std::string funcName, std::string cCode, std::vector<catalog::Type> argumentTypes, catalog::Type returnType) {
   return std::make_shared<CUDFImplementer>(funcName, cCode, argumentTypes, returnType);
}

} // namespace lingodb::compiler::frontend