#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_MEMBERMANAGER_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_MEMBERMANAGER_H
#include <iostream>
#include <memory>
namespace lingodb::compiler::dialect::subop {
namespace internal {
struct MemberInternal;
} // namespace internal
template <class T>
struct Holder {
   std::shared_ptr<T> ptr;
   Holder() : ptr(nullptr) {}
   Holder(std::shared_ptr<T> ptr) : ptr(std::move(ptr)) {}

   bool operator==(const Holder& other) const {
      return *ptr == *other.ptr;
   }
};
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
class Members;
class ColumnRefMemberMapping;
class ColumnDefMemberMapping;
} // namespace lingodb::compiler::dialect::subop
namespace llvm {
class hash_code; // NOLINT (readability-identifier-naming)

llvm::hash_code hash_value(const lingodb::compiler::dialect::subop::Member& m); // NOLINT (readability-identifier-naming)
llvm::hash_code hash_value(lingodb::compiler::dialect::subop::Holder<lingodb::compiler::dialect::subop::Members> arg); // NOLINT (readability-identifier-naming)
llvm::hash_code hash_value(std::shared_ptr<lingodb::compiler::dialect::subop::ColumnDefMemberMapping> arg); // NOLINT (readability-identifier-naming)
llvm::hash_code hash_value(std::shared_ptr<lingodb::compiler::dialect::subop::ColumnRefMemberMapping> arg); // NOLINT (readability-identifier-naming)

} // end namespace llvm
#include <mlir/IR/Types.h>

#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llvm {
inline hash_code hash_value(const lingodb::compiler::dialect::subop::Member& m) {
   // Assuming Member has `id` and `name` fields for example
   return hash_value(m.internal);
}
} // namespace llvm

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

   public:
   Member createMember(std::string name, mlir::Type type) {
      auto uniqueName = getUniqueName(name);
      auto member = std::make_shared<internal::MemberInternal>(uniqueName, type);
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

namespace lingodb::compiler::dialect::subop {
class Members {
   llvm::SmallVector<Member> members;
   llvm::SmallDenseSet<Member> memberSet;

   public:
   Members() = default;
   Members(std::initializer_list<Member> membersList) {
      members.reserve(membersList.size());
      for (Member member : membersList) {
         members.push_back(member);
         memberSet.insert(member);
      }
   }
   Members(const llvm::SmallVectorImpl<std::shared_ptr<Members>>& membersList) {
      for (const auto& memberGroup : membersList) {
         members.insert(members.end(), memberGroup->members.begin(), memberGroup->members.end());
         memberSet.insert(memberGroup->memberSet.begin(), memberGroup->memberSet.end());
      }
   }
   Members(const std::vector<Member>& membersList) : members(membersList.begin(), membersList.end()), memberSet(membersList.begin(), membersList.end()) {}
   Members(const llvm::SmallVector<Member>& membersList)
      : members(membersList.begin(), membersList.end()), memberSet(membersList.begin(), membersList.end()) {}

   Members(const llvm::SmallVector<Member>& membersListA, const llvm::SmallVector<Member>& membersListB) {
      members.reserve(membersListA.size() + membersListB.size());
      members.insert(members.end(), membersListA.begin(), membersListA.end());
      members.insert(members.end(), membersListB.begin(), membersListB.end());
      memberSet.insert(membersListA.begin(), membersListA.end());
      memberSet.insert(membersListB.begin(), membersListB.end());
   }
   const llvm::SmallVector<Member>& getMembers() const {
      return members;
   }
   bool empty() const {
      return members.empty();
   }
   bool contains(const Member& member) const {
      return memberSet.contains(member);
   }
   bool operator==(const Members& other) const {
      for (auto [m1, m2] : llvm::zip(members, other.members)) {
         if (m1 != m2) {
            return false;
         }
      }
      return true;
   }
};

class ColumnRefMemberMapping {
   llvm::SmallVector<std::pair<Member, tuples::ColumnRefAttr>> mapping;

   public:
   using pairType = std::pair<Member, tuples::ColumnRefAttr>;
   ColumnRefMemberMapping() = default;
   ColumnRefMemberMapping(llvm::SmallVector<std::pair<Member, tuples::ColumnRefAttr>> mapping) : mapping(std::move(mapping)) {}
   tuples::ColumnRefAttr getRefAttr(Member member) const {
      for (const auto& pair : mapping) {
         if (pair.first == member) {
            return pair.second;
         }
      }
      assert(false && "Member not found in mapping");
      return tuples::ColumnRefAttr();
   }
   Member getMember(tuples::ColumnRefAttr columnRef) const {
      for (const auto& pair : mapping) {
         if (pair.second == columnRef) {
            return pair.first;
         }
      }
      assert(false && "ColumnRef not found in mapping");
      return Member(nullptr);
   }
   std::shared_ptr<Members> toMembers() {
      llvm::SmallVector<Member> members;
      for (const auto& pair : mapping) {
         members.push_back(pair.first);
      }
      return std::make_shared<Members>(std::move(members));
   }

   const llvm::SmallVector<std::pair<Member, tuples::ColumnRefAttr>>& getMapping() const {
      return mapping;
   }
};
class ColumnDefMemberMapping {
   llvm::SmallVector<std::pair<Member, tuples::ColumnDefAttr>> mapping;

   public:
   using pairType = std::pair<Member, tuples::ColumnDefAttr>;
   ColumnDefMemberMapping() = default;
   ColumnDefMemberMapping(llvm::SmallVector<std::pair<Member, tuples::ColumnDefAttr>> mapping) : mapping(std::move(mapping)) {}
   tuples::ColumnDefAttr getDefAttr(Member member) const {
      for (const auto& pair : mapping) {
         if (pair.first == member) {
            return pair.second;
         }
      }
      assert(false && "Member not found in mapping");
      return tuples::ColumnDefAttr();
   }
   Member getMember(tuples::ColumnDefAttr columnDef) const {
      for (const auto& pair : mapping) {
         if (pair.second == columnDef) {
            return pair.first;
         }
      }
      assert(false && "ColumnDef not found in mapping");
      return Member(nullptr);
   }

   bool hasMember(Member member) const {
      return std::any_of(mapping.begin(), mapping.end(), [&member](const pairType& pair) { return pair.first == member; });
   }
   const llvm::SmallVector<std::pair<Member, tuples::ColumnDefAttr>>& getMapping() const {
      return mapping;
   }

   std::shared_ptr<Members> toMembers() {
      llvm::SmallVector<Member> members;
      for (const auto& pair : mapping) {
         members.push_back(pair.first);
      }
      return std::make_shared<Members>(std::move(members));
   }
};
} // namespace lingodb::compiler::dialect::subop

#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_MEMBERMANAGER_H
