#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_MEMBERMANAGER_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_MEMBERMANAGER_H
#include "llvm/ADT/SmallVector.h"
#include <mlir/IR/Types.h>

#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <memory>
namespace lingodb::compiler::dialect::subop {
namespace internal {
struct MemberInternal;
} // namespace internal
struct Member {
   internal::MemberInternal* internal;
   Member() : internal(nullptr) {}
   Member(internal::MemberInternal* internal) : internal(internal) {}
   // add copy constructor
   Member(const Member& other) : internal(other.internal) {}

   // add copy assignment
   Member& operator=(const Member& other) {
      if (this != &other) {
         internal = other.internal;
      }
      return *this;
   }
   bool operator==(const Member& other) const {
      return internal == other.internal;
   }
   operator bool() const {
      return internal != nullptr;
   }
};


} // namespace lingodb::compiler::dialect::subop

namespace lingodb::compiler::dialect::subop {
namespace internal {
struct MemberInternal {
   std::string name;
   mlir::Type type;
   MemberInternal(std::string name, mlir::Type type) : name(std::move(name)), type(std::move(type)) {}
};
} // namespace internal
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
   std::unordered_set<internal::MemberInternal*> memberSet;

   public:
   Member createMember(std::string name, mlir::Type type) {
      auto uniqueName = getUniqueName(name);
      auto member = std::make_shared<internal::MemberInternal>(uniqueName, type);
      members[uniqueName] = member;
      memberSet.insert(member.get());
      return member.get();
   }
   Member createMemberDirect(std::string name, mlir::Type type) {
      if (members.contains(name)) {
         assert(type == members[name]->type && "Member type mismatch");
         return members[name].get();
      } else {
         auto member = std::make_shared<internal::MemberInternal>(name, type);
         members[name] = member;
         memberSet.insert(member.get());
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
      assert(memberSet.contains(member.internal) && "Member not found in member set");
      return member.internal->name;
   }
   mlir::Type getType(Member member) const {
      assert(memberSet.contains(member.internal) && "Member not found in member set");
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
