#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>
namespace {
using namespace lingodb::compiler::dialect;
static mlir::LogicalResult parseStateMembers(mlir::AsmParser& parser, subop::StateMembersAttr& stateMembersAttr) {
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
   stateMembersAttr = subop::StateMembersAttr::get(parser.getContext(), parser.getBuilder().getArrayAttr(names), parser.getBuilder().getArrayAttr(types));
   return mlir::success();
}
static void printStateMembers(mlir::AsmPrinter& p, subop::StateMembersAttr stateMembersAttr) {
   p << "[";
   auto first = true;
   for (size_t i = 0; i < stateMembersAttr.getNames().size(); i++) {
      auto name = mlir::cast<mlir::StringAttr>(stateMembersAttr.getNames()[i]).str();
      auto type = mlir::cast<mlir::TypeAttr>(stateMembersAttr.getTypes()[i]).getValue();
      if (first) {
         first = false;
      } else {
         p << ", ";
      }
      p << name << " : " << type;
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
} // namespace
subop::StateMembersAttr subop::HashMapType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getKeyMembers().getNames().begin(), getKeyMembers().getNames().end());
   names.insert(names.end(), getValueMembers().getNames().begin(), getValueMembers().getNames().end());
   types.insert(types.end(), getKeyMembers().getTypes().begin(), getKeyMembers().getTypes().end());
   types.insert(types.end(), getValueMembers().getTypes().begin(), getValueMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}
subop::StateMembersAttr subop::PreAggrHtFragmentType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getKeyMembers().getNames().begin(), getKeyMembers().getNames().end());
   names.insert(names.end(), getValueMembers().getNames().begin(), getValueMembers().getNames().end());
   types.insert(types.end(), getKeyMembers().getTypes().begin(), getKeyMembers().getTypes().end());
   types.insert(types.end(), getValueMembers().getTypes().begin(), getValueMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}
subop::StateMembersAttr subop::PreAggrHtType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getKeyMembers().getNames().begin(), getKeyMembers().getNames().end());
   names.insert(names.end(), getValueMembers().getNames().begin(), getValueMembers().getNames().end());
   types.insert(types.end(), getKeyMembers().getTypes().begin(), getKeyMembers().getTypes().end());
   types.insert(types.end(), getValueMembers().getTypes().begin(), getValueMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}
subop::StateMembersAttr subop::HashMultiMapType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getKeyMembers().getNames().begin(), getKeyMembers().getNames().end());
   names.insert(names.end(), getValueMembers().getNames().begin(), getValueMembers().getNames().end());
   types.insert(types.end(), getKeyMembers().getTypes().begin(), getKeyMembers().getTypes().end());
   types.insert(types.end(), getValueMembers().getTypes().begin(), getValueMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}
subop::StateMembersAttr subop::MultiMapType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getKeyMembers().getNames().begin(), getKeyMembers().getNames().end());
   names.insert(names.end(), getValueMembers().getNames().begin(), getValueMembers().getNames().end());
   types.insert(types.end(), getKeyMembers().getTypes().begin(), getKeyMembers().getTypes().end());
   types.insert(types.end(), getValueMembers().getTypes().begin(), getValueMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}
subop::StateMembersAttr subop::ExternalHashIndexType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getKeyMembers().getNames().begin(), getKeyMembers().getNames().end());
   names.insert(names.end(), getValueMembers().getNames().begin(), getValueMembers().getNames().end());
   types.insert(types.end(), getKeyMembers().getTypes().begin(), getKeyMembers().getTypes().end());
   types.insert(types.end(), getValueMembers().getTypes().begin(), getValueMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}
subop::StateMembersAttr subop::MapType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getKeyMembers().getNames().begin(), getKeyMembers().getNames().end());
   names.insert(names.end(), getValueMembers().getNames().begin(), getValueMembers().getNames().end());
   types.insert(types.end(), getKeyMembers().getTypes().begin(), getKeyMembers().getTypes().end());
   types.insert(types.end(), getValueMembers().getTypes().begin(), getValueMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}
subop::StateMembersAttr subop::HashIndexedViewType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getKeyMembers().getNames().begin(), getKeyMembers().getNames().end());
   names.insert(names.end(), getValueMembers().getNames().begin(), getValueMembers().getNames().end());
   types.insert(types.end(), getKeyMembers().getTypes().begin(), getKeyMembers().getTypes().end());
   types.insert(types.end(), getValueMembers().getTypes().begin(), getValueMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
}
subop::StateMembersAttr subop::SegmentTreeViewType::getMembers() {
   std::vector<mlir::Attribute> names;
   std::vector<mlir::Attribute> types;
   names.insert(names.end(), getKeyMembers().getNames().begin(), getKeyMembers().getNames().end());
   names.insert(names.end(), getValueMembers().getNames().begin(), getValueMembers().getNames().end());
   types.insert(types.end(), getKeyMembers().getTypes().begin(), getKeyMembers().getTypes().end());
   types.insert(types.end(), getValueMembers().getTypes().begin(), getValueMembers().getTypes().end());
   return subop::StateMembersAttr::get(this->getContext(), mlir::ArrayAttr::get(this->getContext(), names), mlir::ArrayAttr::get(this->getContext(), types));
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
