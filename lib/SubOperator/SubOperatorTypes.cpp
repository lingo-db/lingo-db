#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

static mlir::LogicalResult parseStateMembers(mlir::AsmParser& parser, mlir::FailureOr<mlir::subop::StateMembersAttr>& stateMembersAttr) {
   if (parser.parseLSquare()) return mlir::failure();
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      llvm::StringRef colName;
      mlir::Type t;
      if (parser.parseKeyword(&colName) || parser.parseColon() || parser.parseType(t)) { return mlir::failure(); }
      names.push_back(parser.getBuilder().getStringAttr(colName));
      types.push_back(mlir::TypeAttr::get(t));
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRSquare()) { return mlir::failure(); }
      break;
   }
   *stateMembersAttr=mlir::subop::StateMembersAttr::get(parser.getContext(),parser.getBuilder().getArrayAttr(names),parser.getBuilder().getArrayAttr(types));
   return mlir::success();
}
static void printStateMembers(mlir::AsmPrinter& p, mlir::subop::StateMembersAttr stateMembersAttr) {
   p << "[";
   auto first = true;
   for (size_t i=0;i<stateMembersAttr.getNames().size();i++) {
      auto name = stateMembersAttr.getNames()[i].cast<mlir::StringAttr>().str();
      auto type = stateMembersAttr.getTypes()[i].cast<mlir::TypeAttr>().getValue();
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      p << name << " : " << type ;
   }
   p << "]";
}
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SubOperator/SubOperatorOpsTypes.cpp.inc"
namespace mlir::subop {
void SubOperatorDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/SubOperator/SubOperatorOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::subop