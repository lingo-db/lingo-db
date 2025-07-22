#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>
namespace {
using namespace lingodb::compiler::dialect;
static mlir::LogicalResult parseStateMembers(mlir::AsmParser& parser, subop::StateMembersAttr& stateMembersAttr) {
   auto& memberManager = parser.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   if (parser.parseLSquare()) return mlir::failure();
   llvm::SmallVector<subop::Member> members;
   while (true) {
      if (!parser.parseOptionalRSquare()) { break; }
      llvm::StringRef colName;
      mlir::Type t;
      if (parser.parseKeyword(&colName) || parser.parseColon() || parser.parseType(t)) { return mlir::failure(); }
      members.push_back(memberManager.createMemberDirect(colName.str(), t));
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRSquare()) { return mlir::failure(); }
      break;
   }
   stateMembersAttr = subop::StateMembersAttr::get(parser.getContext(), members);
   return mlir::success();
}
static void printStateMembers(mlir::AsmPrinter& p, subop::StateMembersAttr stateMembersAttr) {
   auto& memberManager = stateMembersAttr.getContext()->getOrLoadDialect<subop::SubOperatorDialect>()->getMemberManager();
   p << "[";
   auto first = true;
   for (subop::Member m : stateMembersAttr.getMembers()) {
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      p << memberManager.getName(m) << " : " << memberManager.getType(m);
   }
   p << "]";
}

static mlir::LogicalResult parseWithLock(mlir::AsmParser& parser, bool& withLock) {
   llvm::StringRef boolAsRef;
   if (parser.parseOptionalComma().succeeded() && parser.parseKeyword(&boolAsRef).succeeded()) {
      withLock = (boolAsRef == "lockable");
   } else {
      withLock = false;
   }
   return mlir::success();
}
static void printWithLock(mlir::AsmPrinter& p, bool withLock) {
   if (withLock) {
      p << ", lockable";
   }
}
llvm::SmallVector<subop::Member> combineMembers(
   const subop::StateMembersAttr& keyMembers,
   const subop::StateMembersAttr& valueMembers) {
   llvm::SmallVector<subop::Member> combined;
   combined.reserve(keyMembers.getMembers().size() + valueMembers.getMembers().size());
   for (const auto& member : keyMembers.getMembers()) {
      combined.push_back(member);
   }
   for (const auto& member : valueMembers.getMembers()) {
      combined.push_back(member);
   }
   return combined;
}
} // namespace
subop::StateMembersAttr subop::HashMapType::getMembers() {
   return subop::StateMembersAttr::get(this->getContext(), combineMembers(getKeyMembers(), getValueMembers()));
}
subop::StateMembersAttr subop::PreAggrHtFragmentType::getMembers() {
   return subop::StateMembersAttr::get(this->getContext(), combineMembers(getKeyMembers(), getValueMembers()));
}
subop::StateMembersAttr subop::PreAggrHtType::getMembers() {
   return subop::StateMembersAttr::get(this->getContext(), combineMembers(getKeyMembers(), getValueMembers()));
}
subop::StateMembersAttr subop::HashMultiMapType::getMembers() {
   return subop::StateMembersAttr::get(this->getContext(), combineMembers(getKeyMembers(), getValueMembers()));
}
subop::StateMembersAttr subop::MultiMapType::getMembers() {
   return subop::StateMembersAttr::get(this->getContext(), combineMembers(getKeyMembers(), getValueMembers()));
}
subop::StateMembersAttr subop::ExternalHashIndexType::getMembers() {
   return subop::StateMembersAttr::get(this->getContext(), combineMembers(getKeyMembers(), getValueMembers()));
}
subop::StateMembersAttr subop::MapType::getMembers() {
   return subop::StateMembersAttr::get(this->getContext(), combineMembers(getKeyMembers(), getValueMembers()));
}
subop::StateMembersAttr subop::HashIndexedViewType::getMembers() {
   return subop::StateMembersAttr::get(this->getContext(), combineMembers(getKeyMembers(), getValueMembers()));
}
subop::StateMembersAttr subop::SegmentTreeViewType::getMembers() {
   return subop::StateMembersAttr::get(this->getContext(), combineMembers(getKeyMembers(), getValueMembers()));
}
subop::StateMembersAttr subop::SimpleStateType::getValueMembers() {
   return getMembers();
}
#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsTypes.cpp.inc"
void lingodb::compiler::dialect::subop::SubOperatorDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsTypes.cpp.inc"

      >();
}
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsTypeInterfaces.cpp.inc"
