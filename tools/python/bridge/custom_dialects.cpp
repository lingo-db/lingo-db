#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include <iostream>

#include "custom_dialects.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/Dialect/util/UtilOps.h"

//----------------------------------------------------------------------------------------------------------------------
// Util Dialect
//----------------------------------------------------------------------------------------------------------------------
using namespace lingodb::compiler::dialect;
MlirType mlirUtilRefTypeGet(MlirType elementType) {
   return wrap(util::RefType::get(unwrap(elementType)));
}
MlirTypeID mlirUtilRefTypeGetTypeID() {
   return wrap(util::RefType::getTypeID());
}

bool mlirTypeIsAUtilRefType(MlirType type) {
   return llvm::isa<util::RefType>(unwrap(type));
}

MlirType mlirUtilBufferTypeGet(MlirType elementType) {
   return wrap(util::BufferType::get(unwrap(elementType).getContext(),unwrap(elementType)));
}
MlirTypeID mlirUtilBufferTypeGetTypeID() {
   return wrap(util::BufferType::getTypeID());
}

bool mlirTypeIsAUtilBufferType(MlirType type) {
   return llvm::isa<util::BufferType>(unwrap(type));
}

MlirType mlirUtilVarLen32TypeGet(MlirContext context) {
   return wrap(util::VarLen32Type::get(unwrap(context)));
}
MlirTypeID mlirUtilVarLen32TypeGetTypeID() {
   return wrap(util::VarLen32Type::getTypeID());
}
bool mlirTypeIsAUtilVarLen32Type(MlirType type) {
   return llvm::isa<util::VarLen32Type>(unwrap(type));
}

//----------------------------------------------------------------------------------------------------------------------
// TupleStream Dialect
//----------------------------------------------------------------------------------------------------------------------

MlirType mlirTuplesTupleTypeGet(MlirContext context) {
   return wrap(tuples::TupleType::get(unwrap(context)));
}
MlirTypeID mlirTuplesTupleTypeGetTypeID() {
   return wrap(tuples::TupleType::getTypeID());
}
bool mlirTypeIsATuplesTupleType(MlirType type) {
   return llvm::isa<tuples::TupleType>(unwrap(type));
}
MlirType mlirTuplesTupleStreamTypeGet(MlirContext context) {
   return wrap(tuples::TupleStreamType::get(unwrap(context)));
}
MlirTypeID mlirTuplesTupleStreamTypeGetTypeID() {
   return wrap(tuples::TupleStreamType::getTypeID());
}
bool mlirTypeIsATuplesTupleStreamType(MlirType type) {
   return llvm::isa<tuples::TupleStreamType>(unwrap(type));
}
MlirAttribute mlirTuplesColumnDefAttributeGet(MlirContext context, MlirStringRef scope, MlirStringRef name, MlirType type) {
   auto* c = unwrap(context);
   auto& colManager = c->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto res = colManager.createDef(unwrap(scope), unwrap(name));
   res.getColumn().type = unwrap(type);
   return wrap(res);
}
bool mlirAttributeIsATuplesColumnDefAttribute(MlirAttribute attribute) {
   return llvm::isa<tuples::ColumnDefAttr>(unwrap(attribute));
}
MlirAttribute mlirTuplesColumnRefAttributeGet(MlirContext context, MlirStringRef scope, MlirStringRef name) {
   auto* c = unwrap(context);
   auto& colManager = c->getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
   auto res = colManager.createRef(unwrap(scope), unwrap(name));
   return wrap(res);
}
bool mlirAttributeIsATuplesColumnRefAttribute(MlirAttribute attribute) {
   return llvm::isa<tuples::ColumnRefAttr>(unwrap(attribute));
}

//----------------------------------------------------------------------------------------------------------------------
// DB Dialect
//----------------------------------------------------------------------------------------------------------------------

MlirType mlirDBNullableTypeGet(MlirType type) {
   return wrap(db::NullableType::get(unwrap(type)));
}
MlirTypeID mlirDBNullableTypeGetTypeID() {
   return wrap(db::NullableType::getTypeID());
}
bool mlirTypeIsADBNullableType(MlirType type) {
   return llvm::isa<db::NullableType>(unwrap(type));
}
MlirType mlirDBCharTypeGet(MlirContext context, size_t bytes) {
   return wrap(db::CharType::get(unwrap(context), bytes));
}
MlirTypeID mlirDBCharTypeGetTypeID() {
   return wrap(db::CharType::getTypeID());
}
bool mlirTypeIsADBCharType(MlirType type) {
   return llvm::isa<db::CharType>(unwrap(type));
}
MlirType mlirDBDateTypeGet(MlirContext context, size_t unit) {
   return wrap(db::DateType::get(unwrap(context), static_cast<db::DateUnitAttr>(unit)));
}
MlirTypeID mlirDBDateTypeGetTypeID() {
   return wrap(db::DateType::getTypeID());
}
bool mlirTypeIsADBDateType(MlirType type) {
   return llvm::isa<db::DateType>(unwrap(type));
}
MlirType mlirDBIntervalTypeGet(MlirContext context, size_t unit) {
   return wrap(db::IntervalType::get(unwrap(context), static_cast<db::IntervalUnitAttr>(unit)));
}
MlirTypeID mlirDBIntervalTypeGetTypeID() {
   return wrap(db::IntervalType::getTypeID());
}
bool mlirTypeIsADBIntervalType(MlirType type) {
   return llvm::isa<db::IntervalType>(unwrap(type));
}
MlirType mlirDBTimestampTypeGet(MlirContext context, size_t unit) {
   return wrap(db::TimestampType::get(unwrap(context), static_cast<db::TimeUnitAttr>(unit)));
}
MlirTypeID mlirDBTimestampTypeGetTypeID() {
   return wrap(db::TimestampType::getTypeID());
}
bool mlirTypeIsADBTimestampType(MlirType type) {
   return llvm::isa<db::TimestampType>(unwrap(type));
}
MlirType mlirDBDecimalTypeGet(MlirContext context, int p, int s) {
   return wrap(db::DecimalType::get(unwrap(context), p, s));
}
MlirTypeID mlirDBDecimalTypeGetTypeID() {
   return wrap(db::DecimalType::getTypeID());
}
bool mlirTypeIsADBDecimalType(MlirType type) {
   return llvm::isa<db::DecimalType>(unwrap(type));
}
MlirType mlirDBStringTypeGet(MlirContext context) {
   return wrap(db::StringType::get(unwrap(context)));
}
MlirTypeID mlirDBStringTypeGetTypeID() {
   return wrap(db::StringType::getTypeID());
}
bool mlirTypeIsADBStringType(MlirType type) {
   return llvm::isa<db::StringType>(unwrap(type));
}

//----------------------------------------------------------------------------------------------------------------------
// RelAlg Dialect
//----------------------------------------------------------------------------------------------------------------------

MlirAttribute mlirRelalgSortSpecAttributeGet(MlirAttribute colRef, size_t sortSpec) {
   return wrap(relalg::SortSpecificationAttr::get(unwrap(colRef).getContext(), mlir::cast<tuples::ColumnRefAttr>(unwrap(colRef)), static_cast<relalg::SortSpec>(sortSpec)));
}
bool mlirAttributeIsARelalgSortSpecAttribute(MlirAttribute attribute) {
   return llvm::isa<relalg::SortSpecificationAttr>(unwrap(attribute));
}

MlirAttribute mlirRelalgTableMetaDataAttrGetEmpty(MlirContext context) {
   return wrap(relalg::TableMetaDataAttr::get(unwrap(context), std::make_shared<lingodb::runtime::TableMetaData>()));
}
bool mlirAttributeIsARelalgTableMetaDataAttr(MlirAttribute attribute) {
   return llvm::isa<relalg::TableMetaDataAttr>(unwrap(attribute));
}


//----------------------------------------------------------------------------------------------------------------------
// SubOp Dialect
//----------------------------------------------------------------------------------------------------------------------

MlirAttribute mlirSubOpStateMembersAttributeGet(MlirAttribute names, MlirAttribute types) {
   return wrap(subop::StateMembersAttr::get(unwrap(names).getContext(), mlir::cast<mlir::ArrayAttr>(unwrap(names)), mlir::cast<mlir::ArrayAttr>(unwrap(types))));
}
bool mlirAttributeIsASubOpStateMembersAttribute(MlirAttribute attribute) {
   return llvm::isa<subop::StateMembersAttr>(unwrap(attribute));
}
MlirType mlirSubOpTableTypeGet(MlirAttribute members) {
   return wrap(subop::TableType::get(unwrap(members).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(members))));
}
MlirTypeID mlirSubOpTableTypeGetTypeID() {
   return wrap(subop::TableType::getTypeID());
}
bool mlirTypeIsASubOpTableType(MlirType type) {
   return llvm::isa<subop::TableType>(unwrap(type));
}

MlirType mlirSubOpLocalTableTypeGet(MlirAttribute members,MlirAttribute columns) {
   return wrap(subop::LocalTableType::get(unwrap(members).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(members)), mlir::cast<mlir::ArrayAttr>(unwrap(columns))));
}
MlirTypeID mlirSubOpLocalTableTypeGetTypeID() {
   return wrap(subop::LocalTableType::getTypeID());
}
bool mlirTypeIsASubOpLocalTableType(MlirType type) {
   return llvm::isa<subop::LocalTableType>(unwrap(type));
}

MlirType mlirSubOpResultTableTypeGet(MlirAttribute members) {
   return wrap(subop::ResultTableType::get(unwrap(members).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(members))));
}
MlirTypeID mlirSubOpResultTableTypeGetTypeID() {
   return wrap(subop::ResultTableType::getTypeID());
}
bool mlirTypeIsASubOpResultTableType(MlirType type) {
   return llvm::isa<subop::ResultTableType>(unwrap(type));
}

MlirType mlirSubOpSimpleStateTypeGet(MlirAttribute members) {
   return wrap(subop::SimpleStateType::get(unwrap(members).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(members))));
}
MlirTypeID mlirSubOpSimpleStateTypeGetTypeID() {
   return wrap(subop::SimpleStateType::getTypeID());
}
bool mlirTypeIsASubOpSimpleStateType(MlirType type) {
   return llvm::isa<subop::SimpleStateType>(unwrap(type));
}

MlirType mlirSubOpMapTypeGet(MlirAttribute keyMembers, MlirAttribute valMembers) {
   return wrap(subop::MapType::get(unwrap(keyMembers).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(keyMembers)), mlir::cast<subop::StateMembersAttr>(unwrap(valMembers))));
}
MlirTypeID mlirSubOpMapTypeGetTypeID() {
   return wrap(subop::MapType::getTypeID());
}
bool mlirTypeIsASubOpMapType(MlirType type) {
   return llvm::isa<subop::MapType>(unwrap(type));
}

MlirType mlirSubOpMultiMapTypeGet(MlirAttribute keyMembers, MlirAttribute valMembers) {
   return wrap(subop::MultiMapType::get(unwrap(keyMembers).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(keyMembers)), mlir::cast<subop::StateMembersAttr>(unwrap(valMembers))));
}
MlirTypeID mlirSubOpMultiMapTypeGetTypeID() {
   return wrap(subop::MultiMapType::getTypeID());
}
bool mlirTypeIsASubOpMultiMapType(MlirType type) {
   return llvm::isa<subop::MultiMapType>(unwrap(type));
}

MlirType mlirSubOpBufferTypeGet(MlirAttribute members) {
   return wrap(subop::BufferType::get(unwrap(members).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(members))));
}
MlirTypeID mlirSubOpBufferTypeGetTypeID() {
   return wrap(subop::BufferType::getTypeID());
}
bool mlirTypeIsASubOpBufferType(MlirType type) {
   return llvm::isa<subop::BufferType>(unwrap(type));
}

MlirType mlirSubOpArrayTypeGet(MlirAttribute members) {
   return wrap(subop::ArrayType::get(unwrap(members).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(members))));
}
MlirTypeID mlirSubOpArrayTypeGetTypeID() {
   return wrap(subop::ArrayType::getTypeID());
}
bool mlirTypeIsASubOpArrayType(MlirType type) {
   return llvm::isa<subop::ArrayType>(unwrap(type));
}

MlirType mlirSubOpContinuousViewTypeGet(MlirType basedOn) {
   return wrap(subop::ContinuousViewType::get(unwrap(basedOn).getContext(), mlir::cast<subop::State>(unwrap(basedOn))));
}
MlirTypeID mlirSubOpContinuousViewTypeGetTypeID() {
   return wrap(subop::ContinuousViewType::getTypeID());
}
bool mlirTypeIsASubOpContinuousViewType(MlirType type) {
   return llvm::isa<subop::ContinuousViewType>(unwrap(type));
}

MlirType mlirSubOpHeapTypeGet(MlirAttribute members, uint32_t maxElements) {
   return wrap(subop::HeapType::get(unwrap(members).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(members)), maxElements));
}
MlirTypeID mlirSubOpHeapTypeGetTypeID() {
   return wrap(subop::HeapType::getTypeID());
}
bool mlirTypeIsASubOpHeapType(MlirType type) {
   return llvm::isa<subop::HeapType>(unwrap(type));
}

MlirType mlirSubOpSegmentTreeViewTypeGet(MlirAttribute keyMembers, MlirAttribute valMembers) {
   return wrap(subop::SegmentTreeViewType::get(unwrap(keyMembers).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(keyMembers)), mlir::cast<subop::StateMembersAttr>(unwrap(valMembers))));
}
MlirTypeID mlirSubOpSegmentTreeViewTypeGetTypeID() {
   return wrap(subop::SegmentTreeViewType::getTypeID());
}
bool mlirTypeIsASubOpSegmentTreeViewType(MlirType type) {
   return llvm::isa<subop::SegmentTreeViewType>(unwrap(type));
}

MlirType mlirSubOpEntryTypeGet(MlirType t) {
   return wrap(subop::EntryType::get(unwrap(t).getContext(), unwrap(t)));
}
MlirTypeID mlirSubOpEntryTypeGetTypeID() {
   return wrap(subop::EntryType::getTypeID());
}
bool mlirTypeIsASubOpEntryType(MlirType type) {
   return llvm::isa<subop::EntryType>(unwrap(type));
}

MlirType mlirSubOpEntryRefTypeGet(MlirType t) {
   return wrap(subop::EntryRefType::get(unwrap(t).getContext(), mlir::cast<subop::State>(unwrap(t))));
}
MlirTypeID mlirSubOpEntryRefTypeGetTypeID() {
   return wrap(subop::EntryRefType::getTypeID());
}
bool mlirTypeIsASubOpEntryRefType(MlirType type) {
   return llvm::isa<subop::EntryRefType>(unwrap(type));
}

MlirType mlirSubOpMapEntryRefTypeGet(MlirType t) {
   return wrap(subop::MapEntryRefType::get(unwrap(t).getContext(), mlir::cast<subop::MapType>(unwrap(t))));
}
MlirTypeID mlirSubOpMapEntryRefTypeGetTypeID() {
   return wrap(subop::MapEntryRefType::getTypeID());
}
bool mlirTypeIsASubOpMapEntryRefType(MlirType type) {
   return llvm::isa<subop::MapEntryRefType>(unwrap(type));
}

MlirType mlirSubOpTableEntryRefTypeGet(MlirAttribute t) {
   return wrap(subop::TableEntryRefType::get(unwrap(t).getContext(), mlir::cast<subop::StateMembersAttr>(unwrap(t))));
}
MlirTypeID mlirSubOpTableEntryRefTypeGetTypeID() {
   return wrap(subop::TableEntryRefType::getTypeID());
}
bool mlirTypeIsASubOpTableEntryRefType(MlirType type) {
   return llvm::isa<subop::TableEntryRefType>(unwrap(type));
}


MlirType mlirSubOpLookupEntryRefTypeGet(MlirType t) {
   return wrap(subop::LookupEntryRefType::get(unwrap(t).getContext(), mlir::cast<subop::LookupAbleState>(unwrap(t))));
}
MlirTypeID mlirSubOpLookupEntryRefTypeGetTypeID() {
   return wrap(subop::LookupEntryRefType::getTypeID());
}
bool mlirTypeIsASubOpLookupEntryRefType(MlirType type) {
   return llvm::isa<subop::LookupEntryRefType>(unwrap(type));
}

MlirType mlirSubOpMultiMapEntryRefTypeGet(MlirType t) {
   return wrap(subop::MultiMapEntryRefType::get(unwrap(t).getContext(), mlir::cast<subop::MultiMapType>(unwrap(t))));
}
MlirTypeID mlirSubOpMultiMapEntryRefTypeGetTypeID() {
   return wrap(subop::MultiMapEntryRefType::getTypeID());
}
bool mlirTypeIsASubOpMultiMapEntryRefType(MlirType type) {
   return llvm::isa<subop::MultiMapEntryRefType>(unwrap(type));
}

MlirType mlirSubOpContinuousEntryRefTypeGet(MlirType t) {
   return wrap(subop::ContinuousEntryRefType::get(unwrap(t).getContext(), mlir::cast<subop::State>(unwrap(t))));
}
MlirTypeID mlirSubOpContinuousEntryRefTypeGetTypeID() {
   return wrap(subop::ContinuousEntryRefType::getTypeID());
}
bool mlirTypeIsASubOpContinuousEntryRefType(MlirType type) {
   return llvm::isa<subop::ContinuousEntryRefType>(unwrap(type));
}

MlirType mlirSubOpEntryListTypeGet(MlirType t) {
   return wrap(subop::ListType::get(unwrap(t).getContext(), mlir::cast<subop::StateEntryReference>(unwrap(t))));
}
MlirTypeID mlirSubOpEntryListTypeGetTypeID() {
   return wrap(subop::ListType::getTypeID());
}
bool mlirTypeIsASubOpEntryListType(MlirType type) {
   return llvm::isa<subop::ListType>(unwrap(type));
}
