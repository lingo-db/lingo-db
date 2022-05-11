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

DeclarationMatcher methodMatcher = cxxRecordDecl(isDefinition(), hasParent(namespaceDecl(hasName("runtime"))), isExpansionInMainFile()).bind("class");

class MethodPrinter : public MatchFinder::MatchCallback {
   llvm::raw_ostream& hStream;
   std::string translateIntegerType(size_t width) {
      return "mlir::IntegerType::get(context," + std::to_string(width) + ")";
   }
   std::string translateFloatType(size_t width) {
      return "mlir::FloatType::get(context," + std::to_string(width) + ")";
   }
   std::string translatePointer() {
      return "mlir::util::RefType::get(context,mlir::IntegerType::get(context,8))";
   }
   std::optional<std::string> translateType(QualType type) {
      //type.dump();
      if (const auto* tdType = type->getAs<TypedefType>()) {
         return translateType(tdType->desugar());
      }
      if (const auto* pointerType = type->getAs<PointerType>()) {
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
               auto translated = translateType(funcProtoType->getReturnType());
               if (!translated.has_value()) return {};
               funcType += "}, {" + translated.value() + "})";
               return funcType;
            }
         }
         auto asString = pointeeType.getAsString();
         if (asString == "struct runtime::ArrowTable") {
            return "mlir::dsa::TableType::get(context)";
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
            case clang::BuiltinType::Float: return "mlir::FloatType::getF32(context)";
            case clang::BuiltinType::Double: return "mlir::FloatType::getF64(context)";
            case clang::BuiltinType::Int128: return translateIntegerType(128);
            default: break;
         }
      }
      std::string asString = type.getAsString();
      if (asString.ends_with("runtime::VarLen32")) {
         return "mlir::util::VarLen32Type::get(context)";
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
   MethodPrinter(raw_ostream& hStream) : hStream(hStream) {}
   virtual void run(const MatchFinder::MatchResult& result) override {
      //std::cout << "match" << std::endl;
      if (const CXXRecordDecl* md = result.Nodes.getNodeAs<clang::CXXRecordDecl>("class")) {
         std::string className = md->getNameAsString();
         hStream << "struct " << className << " {\n";
         auto* mangleContext = md->getASTContext().createMangleContext();
         for (const auto& method : md->methods()) {
            if (method->isVirtual()) continue;
            if (method->isImplicit()) continue;
            if (isa<CXXConstructorDecl>(method)) continue;
            if (isa<CXXDestructorDecl>(method)) continue;
            if (method->getAccess() == clang::AS_protected || method->getAccess() == clang::AS_private) continue;
            bool noSideEffects=false;
            for(auto *attr:method->attrs()){
               if(attr->getKind()==clang::attr::Annotate){
                  if(auto *annotateAttr=llvm::dyn_cast<clang::AnnotateAttr>(attr)){
                     if(annotateAttr->getAnnotation()=="rt-no-sideffect"){
                        noSideEffects=true;
                     }
                  }
               }
            }
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
            auto resType = method->getReturnType();
            if (!resType->isVoidType()) {
               auto translatedType = translateType(resType);
               if (!translatedType.has_value()) {
                  unknownTypes = true;
                  break;
               }
               resTypes.push_back(translatedType.value());
            }
            if (unknownTypes) {
               std::cerr << "ignoring func " << methodName << std::endl;
               continue;
            }
            hStream << " inline static mlir::util::FunctionSpec " << methodName << " = ";

            std::string fullName = "runtime::" + className + "::" + methodName;
            hStream << " mlir::util::FunctionSpec(\"" << fullName << "\", \"" << mangled << "\", ";
            emitTypeCreateFn(hStream, types);
            hStream << ",";
            emitTypeCreateFn(hStream, resTypes);
            hStream << "," << (noSideEffects ? "true" : "false") << ");\n";
         }
         hStream << "};\n";
      }
   }
};

cl::OptionCategory mycat("myname", "mydescription");

static cl::opt<std::string> headerOutputFile("oh", cl::desc("output path for header"), cl::cat(mycat));

int main(int argc, const char** argv) {
   auto expectedParser = CommonOptionsParser::create(argc, argv, mycat);
   if (!expectedParser) {
      // Fail gracefully for unsupported options.
      llvm::errs() << expectedParser.takeError();
      return 1;
   }
   CommonOptionsParser& optionsParser = expectedParser.get();
   std::string hContent;
   llvm::raw_string_ostream hStream(hContent);
   ClangTool tool(optionsParser.getCompilations(), optionsParser.getSourcePathList());
   MethodPrinter printer(hStream);
   MatchFinder finder;
   finder.addMatcher(methodMatcher, &printer);
   hStream << "#include <mlir/Dialect/util/FunctionHelper.h>\n";
   hStream << "namespace rt {\n";
   tool.run(newFrontendActionFactory(&finder).get());

   hStream << "}\n";
   hStream.flush();
   std::cout << "------ .h --------" << std::endl;
   std::cout << hContent << std::endl;
   std::error_code errorCode;
   llvm::raw_fd_ostream hOStream(headerOutputFile, errorCode);
   hOStream << hContent;
   return 0;
}