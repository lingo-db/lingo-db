#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATOROPSATTRIBUTES_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATOROPSATTRIBUTES_H

#include "lingodb/compiler/Dialect/SubOperator/MemberManager.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
namespace lingodb::compiler::dialect::subop::detail {
struct StateMembersAttrStorage;
struct ColumnRefMemberMappingAttrStorage;
struct ColumnDefMemberMappingAttrStorage;
struct MemberAttrStorage;
} // namespace lingodb::compiler::dialect::subop::detail

namespace lingodb::compiler::dialect::subop {
using DefMappingPairT = std::pair<Member, tuples::ColumnDefAttr>;
using RefMappingPairT = std::pair<Member, tuples::ColumnRefAttr>;
} // namespace lingodb::compiler::dialect::subop
#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h.inc"

#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATOROPSATTRIBUTES_H