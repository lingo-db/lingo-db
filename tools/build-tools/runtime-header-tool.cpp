// Declares clang::SyntaxOnlyAction.
#include "clang/AST/Mangle.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include <cstdio>
#include <iostream>

#include <optional>
using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;
namespace {
cl::OptionCategory mycat("myname", "mydescription");

cl::opt<std::string> headerOutputFile("oh", cl::desc("output path for header"), cl::cat(mycat));
cl::opt<std::string> cppOutputFile("ocpp", cl::desc("output path for cpp file"), cl::cat(mycat));
cl::opt<std::string> libPrefix("lib-prefix", cl::desc("output path for cpp file"), cl::cat(mycat));
cl::opt<std::string> resultNamespace("result-namespace", cl::desc("lib prefix"), cl::cat(mycat));


DeclarationMatcher methodMatcher = cxxRecordDecl(isDefinition(), hasParent(namespaceDecl(matchesName("::runtime"))), isExpansionInMainFile()).bind("class");

class MethodPrinter : public MatchFinder::MatchCallback {
   llvm::raw_ostream& hStream;
   llvm::raw_ostream& cppStream;
   std::string translateIntegerType(size_t width) {
      return "mlir::IntegerType::get(context," + std::to_string(width) + ")";
   }
   std::string translateFloatType(size_t width) {
      return "mlir::FloatType::get(context," + std::to_string(width) + ")";
   }
   std::string translatePointer() {
      return "lingodb::compiler::dialect::util::RefType::get(context,mlir::IntegerType::get(context,8))";
   }
   std::optional<std::string> translateType(QualType type) {
      if (const auto* tdType = type->getAs<TypedefType>()) {
         return translateType(tdType->desugar());
      }
      if (const auto* pointerType = type->getAs<clang::PointerType>()) {
         auto pointeeType = pointerType->desugar()->getPointeeType();
         if (const auto* parenType = pointeeType->getAs<ParenType>()) {
            if (const auto* funcProtoType = parenType->getInnerType()->getAs<FunctionProtoType>()) {
               std::string funcType = "mlir::FunctionType::get(context, {";
               bool first = true;
               for (auto paramType : funcProtoType->param_types()) {
                  if (first) {
                     first = false;
                  } else {
                     funcType += ",";
                  }
                  auto translated = translateType(paramType);
                  if (!translated.has_value()) return {};
                  funcType += translated.value();
               }
               if (funcProtoType->getReturnType()->isVoidType()) {
                  funcType += "},{})";
               } else {
                  auto translated = translateType(funcProtoType->getReturnType());
                  if (!translated.has_value()) return {};
                  funcType += "}, {" + translated.value() + "})";
               }
               return funcType;
            }
         }
         auto asString = pointeeType.getAsString();
         if (asString == "std::shared_ptr<arrow::Table>") {
            return "lingodb::compiler::dialect::dsa::TableType::get(context)";
         }
         return translatePointer();
      }
      auto canonicalType = type.getCanonicalType();
      if (const auto* bt = dyn_cast<BuiltinType>(canonicalType)) {
         switch (bt->getKind()) {
            case clang::BuiltinType::Bool: return translateIntegerType(1);
            case clang::BuiltinType::SChar: return translateIntegerType(8);
            case clang::BuiltinType::UChar: return translateIntegerType(8);
            case clang::BuiltinType::UShort: return translateIntegerType(16);
            case clang::BuiltinType::Short: return translateIntegerType(16);
            case clang::BuiltinType::UInt: return translateIntegerType(32);
            case clang::BuiltinType::Int: return translateIntegerType(32);
            case clang::BuiltinType::ULong: return translateIntegerType(64);
            case clang::BuiltinType::Long: return translateIntegerType(64);
            case clang::BuiltinType::Float: return "mlir::Float32Type::get(context)";
            case clang::BuiltinType::Double: return "mlir::Float64Type::get(context)";
            case clang::BuiltinType::Int128: return translateIntegerType(128);
            default: break;
         }
      }
      std::string asString = type.getAsString();
      if (asString.ends_with("VarLen32")) {
         return "lingodb::compiler::dialect::util::VarLen32Type::get(context)";
      }
      if (asString.ends_with("Buffer")) {
         return "lingodb::compiler::dialect::util::BufferType::get(context," + translateIntegerType(8) + ")";
      }
      return std::optional<std::string>();
   }
   void emitTypeCreateFn(llvm::raw_ostream& os, std::vector<std::string> types) {
      os << " [](mlir::MLIRContext* context) { return std::vector<mlir::Type>{";
      bool first = true;
      for (auto t : types) {
         if (first) {
            first = false;
         } else {
            os << ",";
         }
         os << t;
      }
      os << "};}";
   }

   public:
   MethodPrinter(raw_ostream& hStream, raw_ostream& cppStream) : hStream(hStream), cppStream(cppStream) {}
   virtual void run(const MatchFinder::MatchResult& result) override {
      //std::cout << "match" << std::endl;
      if (const CXXRecordDecl* md = result.Nodes.getNodeAs<clang::CXXRecordDecl>("class")) {
         auto* mangleContext = md->getASTContext().createMangleContext();

         std::string className = md->getNameAsString();
         hStream << "struct " << className << " {\n";
         for (const auto& method : md->getDefinition()->methods()) {
            if (method->isVirtual()) continue;
            if (method->isImplicit()) continue;
            if (isa<CXXConstructorDecl>(method)) continue;
            if (isa<CXXDestructorDecl>(method)) continue;
            if (method->getAccess() == clang::AS_protected || method->getAccess() == clang::AS_private) continue;
            std::string methodName = method->getNameAsString();
            std::vector<std::string> types;
            std::vector<std::string> resTypes;
            if (!method->isStatic()) {
               types.push_back(translatePointer());
            }
            std::string mangled;
            llvm::raw_string_ostream mangleStream(mangled);
            mangleContext->mangleName(method, mangleStream);
            mangleStream.flush();
            bool unknownTypes = false;
            for (const auto& p : method->parameters()) {
               auto translatedType = translateType(p->getType());
               if (!translatedType.has_value()) {
                  unknownTypes = true;
                  break;
               }
               types.push_back(translatedType.value());
            }
            const auto& resType = method->getReturnType();
            if (!resType->isVoidType()) {
               auto translatedType = translateType(resType);
               if (!translatedType.has_value()) {
                  unknownTypes = true;
                  continue;
               }
               resTypes.push_back(translatedType.value());
            }
            if (unknownTypes) {
               std::cerr << "ignoring func " << methodName << std::endl;
               continue;
            }
            auto getPtrFuncName = "getPtrOfMethod" + methodName;
            hStream << "static void* " << getPtrFuncName << "();\n";
            hStream << "#ifndef RUNTIME_PTR_LIB\n";
            hStream << " inline static lingodb::compiler::dialect::util::FunctionSpec " << methodName << " = ";
            std::string fullName = libPrefix + className + "::" + methodName;
            hStream << " lingodb::compiler::dialect::util::FunctionSpec(\"" << fullName << "\", \"" << mangled << "\", ";
            emitTypeCreateFn(hStream, types);
            hStream << ",";
            emitTypeCreateFn(hStream, resTypes);
            hStream << "," << getPtrFuncName << ");\n";
            hStream << "#endif \n";
            cppStream << "void* "<<resultNamespace<<"::" << className << "::" << getPtrFuncName << "(){  auto x= &" << fullName << ";  return *reinterpret_cast<void**>(&x);}";
         };
         hStream << "};\n";
      }
   }
};
} // namespace

int main(int argc, const char** argv) {
   auto expectedParser = CommonOptionsParser::create(argc, argv, mycat);
   if (!expectedParser) {
      // Fail gracefully for unsupported options.
      llvm::errs() << expectedParser.takeError();
      return 1;
   }
   CommonOptionsParser& optionsParser = expectedParser.get();
   std::string hContent;
   std::string cppContent;

   llvm::raw_string_ostream hStream(hContent);
   llvm::raw_string_ostream cppStream(cppContent);

   ClangTool tool(optionsParser.getCompilations(), optionsParser.getSourcePathList());
   auto currentFile = optionsParser.getSourcePathList().at(0);
   std::string delimiter = "include/";
   currentFile.erase(0, currentFile.find(delimiter) + delimiter.length());

   MethodPrinter printer(hStream, cppStream);
   MatchFinder finder;
   finder.addMatcher(methodMatcher, &printer);
   hStream << "#include \"lingodb/compiler/Dialect/util/FunctionHelper.h\"\n";
   hStream << "namespace "<<resultNamespace<<" {\n";
   cppStream << "#define RUNTIME_PTR_LIB\n";
   cppStream << "#include \"" << headerOutputFile << "\"\n";
   cppStream << "#include \"" << currentFile << "\"\n";

   tool.run(newFrontendActionFactory(&finder).get());

   hStream << "}\n";
   hStream.flush();
   cppStream.flush();
   std::cout << "------ .h --------" << std::endl;
   std::cout << hContent << std::endl;
   std::error_code errorCode;
   llvm::raw_fd_ostream hOStream(headerOutputFile, errorCode);
   hOStream << hContent;
   llvm::raw_fd_ostream cppOStream(cppOutputFile, errorCode);
   cppOStream << cppContent;
   return 0;
}