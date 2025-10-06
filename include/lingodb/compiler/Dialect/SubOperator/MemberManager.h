#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_MEMBERMANAGER_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_MEMBERMANAGER_H

#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include <mlir/IR/Types.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace lingodb::compiler::dialect::subop {
namespace internal {
struct MemberInternal {
   std::string name;
   mlir::Type type;
   MemberInternal(std::string name, mlir::Type type) : name(std::move(name)), type(std::move(type)) {}
};
} // namespace internal
struct Member {
   internal::MemberInternal* internal;
   Member() : internal(nullptr) {}
   Member(internal::MemberInternal* internal) : internal(internal) {}
   bool operator==(const Member& other) const {
      return internal == other.internal;
   }
   operator bool() const {
      return internal != nullptr;
   }
};

class MemberManager {
   std::unordered_map<std::string, size_t> counts;
   std::string getUniqueName(std::string name) {
      auto sanitizedName = name.substr(0, name.find("$"));
      std::replace(sanitizedName.begin(), sanitizedName.end(), ' ', '_');
      auto id = counts[sanitizedName]++;
      auto res = sanitizedName + "$" + std::to_string(id);
      return res;
   };

   private:
   std::unordered_map<std::string, std::shared_ptr<internal::MemberInternal>> members;

   public:
   Member createMember(std::string name, mlir::Type type) {
      auto uniqueName = getUniqueName(name);
      auto member = std::make_shared<internal::MemberInternal>(uniqueName, type);
      assert(!members.count(uniqueName));
      members[uniqueName] = member;
      return member.get();
   }
   Member createMemberDirect(std::string name, mlir::Type type) {
      if (members.contains(name)) {
         assert(type == members[name]->type && "Member type mismatch");
         return members[name].get();
      } else {
         auto member = std::make_shared<internal::MemberInternal>(name, type);
         members[name] = member;
         if (name.find("$") != std::string::npos) {
            auto baseName = name.substr(0, name.find("$"));
            size_t id = std::stoi(name.substr(name.find("$") + 1));
            counts[baseName] = std::max(counts[baseName], id + 1);
         }
         return member.get();
      }
   }
   Member lookupMember(std::string name) {
      auto it = members.find(name);
      if (it != members.end()) {
         return it->second.get();
      }
      assert(false && "Member not found");
      return nullptr;
   }
   const std::string& getName(Member member) const {
      return member.internal->name;
   }
   mlir::Type getType(Member member) const {
      return member.internal->type;
   }
};
} // namespace lingodb::compiler::dialect::subop

namespace std {
template <>
struct hash<lingodb::compiler::dialect::subop::Member> {
   std::size_t operator()(const lingodb::compiler::dialect::subop::Member& m) const {
      return std::hash<decltype(m.internal)>{}(m.internal);
   }
};
} // namespace std

namespace llvm {
template <>
struct DenseMapInfo<lingodb::compiler::dialect::subop::Member> {
   using Member = lingodb::compiler::dialect::subop::Member;
   using MemberInternal = lingodb::compiler::dialect::subop::internal::MemberInternal;
   static inline Member getEmptyKey() {
      return Member(DenseMapInfo<MemberInternal*>::getEmptyKey());
   }

   static inline Member getTombstoneKey() {
      return Member(DenseMapInfo<MemberInternal*>::getTombstoneKey());
   }

   static unsigned getHashValue(const Member& m) {
      return DenseMapInfo<MemberInternal*>::getHashValue(m.internal);
   }

   static bool isEqual(const Member& lhs, const Member& rhs) {
      return lhs.internal == rhs.internal;
   }
};
} // namespace llvm

#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_MEMBERMANAGER_H
