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
   llvm::SmallVector<Member> members;
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
   return StateMembersAttr::get(odsParser.getContext(), members);
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
   auto& memberManager = odsParser.getContext()->getOrLoadDialect<SubOperatorDialect>()->getMemberManager();
   if (odsParser.parseLSquare()) return {};
   llvm::SmallVector<std::pair<Member, tuples::ColumnRefAttr>> columns;
   while (true) {
      if (!odsParser.parseOptionalRSquare()) { break; }
      llvm::StringRef colName;
      tuples::ColumnRefAttr colRefAttr;
      if (odsParser.parseKeyword(&colName) || odsParser.parseColon() || odsParser.parseAttribute(colRefAttr)) {
         return {};
      }
      columns.push_back({memberManager.lookupMember(colName.str()), colRefAttr});
      if (!odsParser.parseOptionalComma()) { continue; }
      if (odsParser.parseRSquare()) { return {}; }
      break;
   }
   return ColumnRefMemberMappingAttr::get(odsParser.getContext(), columns);
}
void ColumnRefMemberMappingAttr::print(::mlir::AsmPrinter& odsPrinter) const {
   auto& memberManager = getContext()->getOrLoadDialect<SubOperatorDialect>()->getMemberManager();
   odsPrinter << "[";
   auto first = true;
   for (auto& [member, col] : getMapping()) {
      if (first) {
         first = false;
      } else {
         odsPrinter << ", ";
      }
      odsPrinter << memberManager.getName(member) << " : " << col;
   }
}

void ColumnDefMemberMappingAttr::print(::mlir::AsmPrinter& odsPrinter) const {
   auto& memberManager = getContext()->getOrLoadDialect<SubOperatorDialect>()->getMemberManager();
   odsPrinter << "[";
   auto first = true;
   for (auto& [member, col] : getMapping()) {
      if (first) {
         first = false;
      } else {
         odsPrinter << ", ";
      }
      odsPrinter << memberManager.getName(member) << " : " << col;
   }
}
::mlir::Attribute ColumnDefMemberMappingAttr::parse(::mlir::AsmParser& odsParser, ::mlir::Type odsType) {
   auto& memberManager = odsParser.getContext()->getOrLoadDialect<SubOperatorDialect>()->getMemberManager();
   if (odsParser.parseLSquare()) return {};
   llvm::SmallVector<std::pair<Member, tuples::ColumnDefAttr>> columns;
   while (true) {
      if (!odsParser.parseOptionalRSquare()) { break; }
      llvm::StringRef colName;
      tuples::ColumnDefAttr colDefAttr;
      if (odsParser.parseKeyword(&colName) || odsParser.parseColon() || odsParser.parseAttribute(colDefAttr)) {
         return {};
      }
      columns.push_back({memberManager.lookupMember(colName.str()), colDefAttr});
      if (!odsParser.parseOptionalComma()) { continue; }
      if (odsParser.parseRSquare()) { return {}; }
      break;
   }
   return ColumnDefMemberMappingAttr::get(odsParser.getContext(), columns);
}

void MemberAttr::print(::mlir::AsmPrinter& odsPrinter) const {
   auto& memberManager = getContext()->getOrLoadDialect<SubOperatorDialect>()->getMemberManager();
   odsPrinter.printString(memberManager.getName(getMember()));
}
::mlir::Attribute MemberAttr::parse(::mlir::AsmParser& odsParser, ::mlir::Type odsType) {
   auto& memberManager = odsParser.getContext()->getOrLoadDialect<SubOperatorDialect>()->getMemberManager();
   std::string memberName;
   if (odsParser.parseString(&memberName).failed()) {
      return {};
   }
   return MemberAttr::get(odsParser.getContext(), memberManager.lookupMember(memberName));
}
} // namespace lingodb::compiler::dialect::subop

namespace lingodb::compiler::dialect::subop::detail {
struct StateMembersAttrStorage : public ::mlir::AttributeStorage {
   using KeyTy = llvm::SmallVector<Member>;

   StateMembersAttrStorage(KeyTy members) : members(std::move(members)) {}

   static StateMembersAttrStorage* construct(::mlir::AttributeStorageAllocator& allocator, const KeyTy& key) {
      return new (allocator.allocate<StateMembersAttrStorage>()) StateMembersAttrStorage(key);
   }
   bool operator==(const StateMembersAttrStorage& other) const {
      if (members.size() != other.members.size()) return false;
      for (size_t i = 0; i < members.size(); ++i) {
         if (members[i] != other.members[i]) return false;
      }
      return true;
   }
   static ::llvm::hash_code hashKey(const KeyTy& key) {
      ::llvm::hash_code hash = 0;
      for (const auto& member : key) {
         hash = llvm::hash_combine(hash, member.internal);
      }
      return hash;
   }
   KeyTy members;
};
struct MemberAttrStorage : public ::mlir::AttributeStorage {
   using KeyTy = subop::Member;

   MemberAttrStorage(KeyTy member) : member(std::move(member)) {}

   static MemberAttrStorage* construct(::mlir::AttributeStorageAllocator& allocator, const KeyTy& key) {
      return new (allocator.allocate<MemberAttrStorage>()) MemberAttrStorage(key);
   }
   bool operator==(const MemberAttrStorage& other) const {
      return member == other.member;
   }
   static ::llvm::hash_code hashKey(const KeyTy& key) {
      return llvm::hash_value(key.internal);
   }
   KeyTy member;
};
struct ColumnDefMemberMappingAttrStorage : public ::mlir::AttributeStorage {
   using KeyTy = llvm::SmallVector<std::pair<Member, tuples::ColumnDefAttr>>;

   ColumnDefMemberMappingAttrStorage(KeyTy mapping) : mapping(std::move(mapping)) {}

   static ColumnDefMemberMappingAttrStorage* construct(::mlir::AttributeStorageAllocator& allocator, const KeyTy& key) {
      return new (allocator.allocate<ColumnDefMemberMappingAttrStorage>()) ColumnDefMemberMappingAttrStorage(key);
   }
   bool operator==(const ColumnDefMemberMappingAttrStorage& other) const {
      if (mapping.size() != other.mapping.size()) return false;
      for (size_t i = 0; i < mapping.size(); ++i) {
         if (mapping[i].first != other.mapping[i].first || mapping[i].second != other.mapping[i].second) {
            return false;
         }
      }
      return true;
   }
   static ::llvm::hash_code hashKey(const KeyTy& key) {
      ::llvm::hash_code hash = 0;
      for (const auto& pair : key) {
         hash = llvm::hash_combine(hash, pair.first.internal, pair.second);
      }
      return hash;
   }
   KeyTy mapping;
};
struct ColumnRefMemberMappingAttrStorage : public ::mlir::AttributeStorage {
   using KeyTy = llvm::SmallVector<std::pair<Member, tuples::ColumnRefAttr>>;

   ColumnRefMemberMappingAttrStorage(KeyTy mapping) : mapping(std::move(mapping)) {}

   static ColumnRefMemberMappingAttrStorage* construct(::mlir::AttributeStorageAllocator& allocator, const KeyTy& key) {
      return new (allocator.allocate<ColumnRefMemberMappingAttrStorage>()) ColumnRefMemberMappingAttrStorage(key);
   }
   bool operator==(const ColumnRefMemberMappingAttrStorage& other) const {
      if (mapping.size() != other.mapping.size()) return false;
      for (size_t i = 0; i < mapping.size(); ++i) {
         if (mapping[i].first != other.mapping[i].first || mapping[i].second != other.mapping[i].second) {
            return false;
         }
      }
      return true;
   }
   static ::llvm::hash_code hashKey(const KeyTy& key) {
      ::llvm::hash_code hash = 0;
      for (const auto& pair : key) {
         hash = llvm::hash_combine(hash, pair.first.internal, pair.second);
      }
      return hash;
   }
   KeyTy mapping;
};
} // namespace lingodb::compiler::dialect::subop::detail

namespace lingodb::compiler::dialect::subop {
llvm::SmallVector<std::pair<Member, tuples::ColumnDefAttr>> ColumnDefMemberMappingAttr::getMappingList() const {
   return getImpl()->mapping;
}
llvm::SmallVector<std::pair<Member, tuples::ColumnRefAttr>> ColumnRefMemberMappingAttr::getMappingList() const {
   return getImpl()->mapping;
}
llvm::SmallVector<Member> StateMembersAttr::getMemberList() const {
   return getImpl()->members;
}
Member MemberAttr::getMember() const {
   return getImpl()->member;
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