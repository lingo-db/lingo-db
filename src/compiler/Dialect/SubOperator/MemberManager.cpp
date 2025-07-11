#include "lingodb/compiler/Dialect/SubOperator/MemberManager.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
namespace llvm {
llvm::hash_code hash_value(lingodb::compiler::dialect::subop::Holder<lingodb::compiler::dialect::subop::Members> arg) { // NOLINT (readability-identifier-naming)
   return llvm::hash_combine_range(arg.ptr->getMembers().begin(), arg.ptr->getMembers().end());
}
llvm::hash_code hash_value(std::shared_ptr<lingodb::compiler::dialect::subop::ColumnDefMemberMapping> arg) { // NOLINT (readability-identifier-naming)
   return llvm::hash_value(arg.get()); //todo: fix
}
llvm::hash_code hash_value(std::shared_ptr<lingodb::compiler::dialect::subop::ColumnRefMemberMapping> arg) { // NOLINT (readability-identifier-naming)
   return llvm::hash_value(arg.get()); //todo: fix
}

} // end namespace llvm