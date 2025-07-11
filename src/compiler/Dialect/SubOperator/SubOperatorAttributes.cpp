#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include <llvm/ADT/TypeSwitch.h>

namespace lingodb::compiler::dialect::subop {

::mlir::Attribute StateMembersAttr::parse(::mlir::AsmParser& odsParser, ::mlir::Type odsType) {
   auto& memberManager = odsParser.getContext()->getOrLoadDialect<SubOperatorDialect>()->getMemberManager();
   if (odsParser.parseLSquare()) return {};
   std::vector<Member> members;
   while (true) {
      if (!odsParser.parseOptionalRSquare()) { break; }
      llvm::StringRef colName;
      ::mlir::Type t;
      if (odsParser.parseKeyword(&colName) || odsParser.parseColon() || odsParser.parseType(t)) { return {}; }
      members.push_back(memberManager.createMemberDirect(colName.str(), t));
      if (!odsParser.parseOptionalComma()) { continue; }
      if (odsParser.parseRSquare()) { return {}; }
      break;
   }
   return StateMembersAttr::get(odsParser.getContext(), std::make_shared<Members>(members));
}
void StateMembersAttr::print(::mlir::AsmPrinter& odsPrinter) const {
   auto& memberManager = getContext()->getOrLoadDialect<SubOperatorDialect>()->getMemberManager();
   odsPrinter << "[";
   auto first = true;
   for (Member m : getMembers()) {
      if (first) {
         first = false;
      } else {
         odsPrinter << ", ";
      }
      odsPrinter << memberManager.getName(m) << " : " << memberManager.getType(m);
   }
   odsPrinter << "]";
}

::mlir::Attribute ColumnRefMemberMappingAttr::parse(::mlir::AsmParser& odsParser, ::mlir::Type odsType) {
   assert(false);
   //todo
}
void ColumnRefMemberMappingAttr::print(::mlir::AsmPrinter& odsPrinter) const {
   auto& memberManager = getContext()->getOrLoadDialect<SubOperatorDialect>()->getMemberManager();
   odsPrinter << "[";
   auto first = true;
   for (auto& [member, col] : getMapping()->getMapping()) {
      if (first) {
         first = false;
      } else {
         odsPrinter << ", ";
      }
      odsPrinter << memberManager.getName(member) << " : " << col;
   }
}

void ColumnDefMemberMappingAttr::print(::mlir::AsmPrinter& odsPrinter) const {
   assert(false);
   //todo
}
::mlir::Attribute ColumnDefMemberMappingAttr::parse(::mlir::AsmParser& odsParser, ::mlir::Type odsType) {
   assert(false);
   //todo
}

void MemberAttr::print(::mlir::AsmPrinter& odsPrinter) const {
   auto& memberManager = getContext()->getOrLoadDialect<SubOperatorDialect>()->getMemberManager();
   odsPrinter.printString(memberManager.getName(getMember()));
}
::mlir::Attribute MemberAttr::parse(::mlir::AsmParser& odsParser, ::mlir::Type odsType) {
   assert(false);
   //todo
}

} // namespace lingodb::compiler::dialect::subop

#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.cpp.inc"

void lingodb::compiler::dialect::subop::SubOperatorDialect::registerAttrs() {
   addAttributes<
#define GET_ATTRDEF_LIST
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.cpp.inc"

      >();
}