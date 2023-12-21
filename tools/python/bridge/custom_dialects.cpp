#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include <iostream>

#include "custom_dialects.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/Dialect/util/UtilOps.h"

//----------------------------------------------------------------------------------------------------------------------
// Util Dialect
//----------------------------------------------------------------------------------------------------------------------

MlirType mlirUtilRefTypeGet(MlirType elementType) {
   return wrap(mlir::util::RefType::get(unwrap(elementType)));
}
MlirTypeID mlirUtilRefTypeGetTypeID() {
   return wrap(mlir::util::RefType::getTypeID());
}

bool mlirTypeIsAUtilRefType(MlirType type) {
   return llvm::isa<mlir::util::RefType>(unwrap(type));
}

MlirType mlirUtilBufferTypeGet(MlirType elementType) {
   return wrap(mlir::util::BufferType::get(unwrap(elementType).getContext(),unwrap(elementType)));
}
MlirTypeID mlirUtilBufferTypeGetTypeID() {
   return wrap(mlir::util::BufferType::getTypeID());
}

bool mlirTypeIsAUtilBufferType(MlirType type) {
   return llvm::isa<mlir::util::BufferType>(unwrap(type));
}

MlirType mlirUtilVarLen32TypeGet(MlirContext context) {
   return wrap(mlir::util::VarLen32Type::get(unwrap(context)));
}
MlirTypeID mlirUtilVarLen32TypeGetTypeID() {
   return wrap(mlir::util::VarLen32Type::getTypeID());
}
bool mlirTypeIsAUtilVarLen32Type(MlirType type) {
   return llvm::isa<mlir::util::VarLen32Type>(unwrap(type));
}

//----------------------------------------------------------------------------------------------------------------------
// TupleStream Dialect
//----------------------------------------------------------------------------------------------------------------------

MlirType mlirTuplesTupleTypeGet(MlirContext context) {
   return wrap(mlir::tuples::TupleType::get(unwrap(context)));
}
MlirTypeID mlirTuplesTupleTypeGetTypeID() {
   return wrap(mlir::tuples::TupleType::getTypeID());
}
bool mlirTypeIsATuplesTupleType(MlirType type) {
   return llvm::isa<mlir::tuples::TupleType>(unwrap(type));
}
MlirType mlirTuplesTupleStreamTypeGet(MlirContext context) {
   return wrap(mlir::tuples::TupleStreamType::get(unwrap(context)));
}
MlirTypeID mlirTuplesTupleStreamTypeGetTypeID() {
   return wrap(mlir::tuples::TupleStreamType::getTypeID());
}
bool mlirTypeIsATuplesTupleStreamType(MlirType type) {
   return llvm::isa<mlir::tuples::TupleStreamType>(unwrap(type));
}
MlirAttribute mlirTuplesColumnDefAttributeGet(MlirContext context, MlirStringRef scope, MlirStringRef name, MlirType type) {
   auto* c = unwrap(context);
   auto& colManager = c->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   auto res = colManager.createDef(unwrap(scope), unwrap(name));
   res.getColumn().type = unwrap(type);
   return wrap(res);
}
bool mlirAttributeIsATuplesColumnDefAttribute(MlirAttribute attribute) {
   return llvm::isa<mlir::tuples::ColumnDefAttr>(unwrap(attribute));
}
MlirAttribute mlirTuplesColumnRefAttributeGet(MlirContext context, MlirStringRef scope, MlirStringRef name) {
   auto* c = unwrap(context);
   auto& colManager = c->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
   auto res = colManager.createRef(unwrap(scope), unwrap(name));
   return wrap(res);
}
bool mlirAttributeIsATuplesColumnRefAttribute(MlirAttribute attribute) {
   return llvm::isa<mlir::tuples::ColumnRefAttr>(unwrap(attribute));
}

//----------------------------------------------------------------------------------------------------------------------
// DB Dialect
//----------------------------------------------------------------------------------------------------------------------

MlirType mlirDBNullableTypeGet(MlirType type) {
   return wrap(mlir::db::NullableType::get(unwrap(type)));
}
MlirTypeID mlirDBNullableTypeGetTypeID() {
   return wrap(mlir::db::NullableType::getTypeID());
}
bool mlirTypeIsADBNullableType(MlirType type) {
   return llvm::isa<mlir::db::NullableType>(unwrap(type));
}
MlirType mlirDBCharTypeGet(MlirContext context, size_t bytes) {
   return wrap(mlir::db::CharType::get(unwrap(context), bytes));
}
MlirTypeID mlirDBCharTypeGetTypeID() {
   return wrap(mlir::db::CharType::getTypeID());
}
bool mlirTypeIsADBCharType(MlirType type) {
   return llvm::isa<mlir::db::CharType>(unwrap(type));
}
MlirType mlirDBDateTypeGet(MlirContext context, size_t unit) {
   return wrap(mlir::db::DateType::get(unwrap(context), static_cast<mlir::db::DateUnitAttr>(unit)));
}
MlirTypeID mlirDBDateTypeGetTypeID() {
   return wrap(mlir::db::DateType::getTypeID());
}
bool mlirTypeIsADBDateType(MlirType type) {
   return llvm::isa<mlir::db::DateType>(unwrap(type));
}
MlirType mlirDBIntervalTypeGet(MlirContext context, size_t unit) {
   return wrap(mlir::db::IntervalType::get(unwrap(context), static_cast<mlir::db::IntervalUnitAttr>(unit)));
}
MlirTypeID mlirDBIntervalTypeGetTypeID() {
   return wrap(mlir::db::IntervalType::getTypeID());
}
bool mlirTypeIsADBIntervalType(MlirType type) {
   return llvm::isa<mlir::db::IntervalType>(unwrap(type));
}
MlirType mlirDBTimestampTypeGet(MlirContext context, size_t unit) {
   return wrap(mlir::db::TimestampType::get(unwrap(context), static_cast<mlir::db::TimeUnitAttr>(unit)));
}
MlirTypeID mlirDBTimestampTypeGetTypeID() {
   return wrap(mlir::db::TimestampType::getTypeID());
}
bool mlirTypeIsADBTimestampType(MlirType type) {
   return llvm::isa<mlir::db::TimestampType>(unwrap(type));
}
MlirType mlirDBDecimalTypeGet(MlirContext context, int p, int s) {
   return wrap(mlir::db::DecimalType::get(unwrap(context), p, s));
}
MlirTypeID mlirDBDecimalTypeGetTypeID() {
   return wrap(mlir::db::DecimalType::getTypeID());
}
bool mlirTypeIsADBDecimalType(MlirType type) {
   return llvm::isa<mlir::db::DecimalType>(unwrap(type));
}
MlirType mlirDBStringTypeGet(MlirContext context) {
   return wrap(mlir::db::StringType::get(unwrap(context)));
}
MlirTypeID mlirDBStringTypeGetTypeID() {
   return wrap(mlir::db::StringType::getTypeID());
}
bool mlirTypeIsADBStringType(MlirType type) {
   return llvm::isa<mlir::db::StringType>(unwrap(type));
}

//----------------------------------------------------------------------------------------------------------------------
// RelAlg Dialect
//----------------------------------------------------------------------------------------------------------------------

MlirAttribute mlirRelalgSortSpecAttributeGet(MlirAttribute colRef, size_t sortSpec) {
   return wrap(mlir::relalg::SortSpecificationAttr::get(unwrap(colRef).getContext(), mlir::cast<mlir::tuples::ColumnRefAttr>(unwrap(colRef)), static_cast<mlir::relalg::SortSpec>(sortSpec)));
}
bool mlirAttributeIsARelalgSortSpecAttribute(MlirAttribute attribute) {
   return llvm::isa<mlir::relalg::SortSpecificationAttr>(unwrap(attribute));
}

MlirAttribute mlirRelalgTableMetaDataAttrGetEmpty(MlirContext context) {
   return wrap(mlir::relalg::TableMetaDataAttr::get(unwrap(context), std::make_shared<runtime::TableMetaData>()));
}
bool mlirAttributeIsARelalgTableMetaDataAttr(MlirAttribute attribute) {
   return llvm::isa<mlir::relalg::TableMetaDataAttr>(unwrap(attribute));
}


//----------------------------------------------------------------------------------------------------------------------
// SubOp Dialect
//----------------------------------------------------------------------------------------------------------------------

MlirAttribute mlirSubOpStateMembersAttributeGet(MlirAttribute names, MlirAttribute types) {
   return wrap(mlir::subop::StateMembersAttr::get(unwrap(names).getContext(), mlir::cast<mlir::ArrayAttr>(unwrap(names)), mlir::cast<mlir::ArrayAttr>(unwrap(types))));
}
bool mlirAttributeIsASubOpStateMembersAttribute(MlirAttribute attribute) {
   return llvm::isa<mlir::subop::StateMembersAttr>(unwrap(attribute));
}
MlirType mlirSubOpTableTypeGet(MlirAttribute members) {
   return wrap(mlir::subop::TableType::get(unwrap(members).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(members))));
}
MlirTypeID mlirSubOpTableTypeGetTypeID() {
   return wrap(mlir::subop::TableType::getTypeID());
}
bool mlirTypeIsASubOpTableType(MlirType type) {
   return llvm::isa<mlir::subop::TableType>(unwrap(type));
}

MlirType mlirSubOpLocalTableTypeGet(MlirAttribute members,MlirAttribute columns) {
   return wrap(mlir::subop::LocalTableType::get(unwrap(members).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(members)), mlir::cast<mlir::ArrayAttr>(unwrap(columns))));
}
MlirTypeID mlirSubOpLocalTableTypeGetTypeID() {
   return wrap(mlir::subop::LocalTableType::getTypeID());
}
bool mlirTypeIsASubOpLocalTableType(MlirType type) {
   return llvm::isa<mlir::subop::LocalTableType>(unwrap(type));
}

MlirType mlirSubOpResultTableTypeGet(MlirAttribute members) {
   return wrap(mlir::subop::ResultTableType::get(unwrap(members).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(members))));
}
MlirTypeID mlirSubOpResultTableTypeGetTypeID() {
   return wrap(mlir::subop::ResultTableType::getTypeID());
}
bool mlirTypeIsASubOpResultTableType(MlirType type) {
   return llvm::isa<mlir::subop::ResultTableType>(unwrap(type));
}

MlirType mlirSubOpSimpleStateTypeGet(MlirAttribute members) {
   return wrap(mlir::subop::SimpleStateType::get(unwrap(members).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(members))));
}
MlirTypeID mlirSubOpSimpleStateTypeGetTypeID() {
   return wrap(mlir::subop::SimpleStateType::getTypeID());
}
bool mlirTypeIsASubOpSimpleStateType(MlirType type) {
   return llvm::isa<mlir::subop::SimpleStateType>(unwrap(type));
}

MlirType mlirSubOpMapTypeGet(MlirAttribute keyMembers, MlirAttribute valMembers) {
   return wrap(mlir::subop::MapType::get(unwrap(keyMembers).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(keyMembers)), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(valMembers))));
}
MlirTypeID mlirSubOpMapTypeGetTypeID() {
   return wrap(mlir::subop::MapType::getTypeID());
}
bool mlirTypeIsASubOpMapType(MlirType type) {
   return llvm::isa<mlir::subop::MapType>(unwrap(type));
}

MlirType mlirSubOpMultiMapTypeGet(MlirAttribute keyMembers, MlirAttribute valMembers) {
   return wrap(mlir::subop::MultiMapType::get(unwrap(keyMembers).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(keyMembers)), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(valMembers))));
}
MlirTypeID mlirSubOpMultiMapTypeGetTypeID() {
   return wrap(mlir::subop::MultiMapType::getTypeID());
}
bool mlirTypeIsASubOpMultiMapType(MlirType type) {
   return llvm::isa<mlir::subop::MultiMapType>(unwrap(type));
}

MlirType mlirSubOpBufferTypeGet(MlirAttribute members) {
   return wrap(mlir::subop::BufferType::get(unwrap(members).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(members))));
}
MlirTypeID mlirSubOpBufferTypeGetTypeID() {
   return wrap(mlir::subop::BufferType::getTypeID());
}
bool mlirTypeIsASubOpBufferType(MlirType type) {
   return llvm::isa<mlir::subop::BufferType>(unwrap(type));
}

MlirType mlirSubOpArrayTypeGet(MlirAttribute members) {
   return wrap(mlir::subop::ArrayType::get(unwrap(members).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(members))));
}
MlirTypeID mlirSubOpArrayTypeGetTypeID() {
   return wrap(mlir::subop::ArrayType::getTypeID());
}
bool mlirTypeIsASubOpArrayType(MlirType type) {
   return llvm::isa<mlir::subop::ArrayType>(unwrap(type));
}

MlirType mlirSubOpContinuousViewTypeGet(MlirType basedOn) {
   return wrap(mlir::subop::ContinuousViewType::get(unwrap(basedOn).getContext(), mlir::cast<mlir::subop::State>(unwrap(basedOn))));
}
MlirTypeID mlirSubOpContinuousViewTypeGetTypeID() {
   return wrap(mlir::subop::ContinuousViewType::getTypeID());
}
bool mlirTypeIsASubOpContinuousViewType(MlirType type) {
   return llvm::isa<mlir::subop::ContinuousViewType>(unwrap(type));
}

MlirType mlirSubOpHeapTypeGet(MlirAttribute members, uint32_t maxElements) {
   return wrap(mlir::subop::HeapType::get(unwrap(members).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(members)), maxElements));
}
MlirTypeID mlirSubOpHeapTypeGetTypeID() {
   return wrap(mlir::subop::HeapType::getTypeID());
}
bool mlirTypeIsASubOpHeapType(MlirType type) {
   return llvm::isa<mlir::subop::HeapType>(unwrap(type));
}

MlirType mlirSubOpSegmentTreeViewTypeGet(MlirAttribute keyMembers, MlirAttribute valMembers) {
   return wrap(mlir::subop::SegmentTreeViewType::get(unwrap(keyMembers).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(keyMembers)), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(valMembers))));
}
MlirTypeID mlirSubOpSegmentTreeViewTypeGetTypeID() {
   return wrap(mlir::subop::SegmentTreeViewType::getTypeID());
}
bool mlirTypeIsASubOpSegmentTreeViewType(MlirType type) {
   return llvm::isa<mlir::subop::SegmentTreeViewType>(unwrap(type));
}

MlirType mlirSubOpEntryTypeGet(MlirType t) {
   return wrap(mlir::subop::EntryType::get(unwrap(t).getContext(), unwrap(t)));
}
MlirTypeID mlirSubOpEntryTypeGetTypeID() {
   return wrap(mlir::subop::EntryType::getTypeID());
}
bool mlirTypeIsASubOpEntryType(MlirType type) {
   return llvm::isa<mlir::subop::EntryType>(unwrap(type));
}

MlirType mlirSubOpEntryRefTypeGet(MlirType t) {
   return wrap(mlir::subop::EntryRefType::get(unwrap(t).getContext(), mlir::cast<mlir::subop::State>(unwrap(t))));
}
MlirTypeID mlirSubOpEntryRefTypeGetTypeID() {
   return wrap(mlir::subop::EntryRefType::getTypeID());
}
bool mlirTypeIsASubOpEntryRefType(MlirType type) {
   return llvm::isa<mlir::subop::EntryRefType>(unwrap(type));
}

MlirType mlirSubOpMapEntryRefTypeGet(MlirType t) {
   return wrap(mlir::subop::MapEntryRefType::get(unwrap(t).getContext(), mlir::cast<mlir::subop::MapType>(unwrap(t))));
}
MlirTypeID mlirSubOpMapEntryRefTypeGetTypeID() {
   return wrap(mlir::subop::MapEntryRefType::getTypeID());
}
bool mlirTypeIsASubOpMapEntryRefType(MlirType type) {
   return llvm::isa<mlir::subop::MapEntryRefType>(unwrap(type));
}

MlirType mlirSubOpTableEntryRefTypeGet(MlirAttribute t) {
   return wrap(mlir::subop::TableEntryRefType::get(unwrap(t).getContext(), mlir::cast<mlir::subop::StateMembersAttr>(unwrap(t))));
}
MlirTypeID mlirSubOpTableEntryRefTypeGetTypeID() {
   return wrap(mlir::subop::TableEntryRefType::getTypeID());
}
bool mlirTypeIsASubOpTableEntryRefType(MlirType type) {
   return llvm::isa<mlir::subop::TableEntryRefType>(unwrap(type));
}


MlirType mlirSubOpLookupEntryRefTypeGet(MlirType t) {
   return wrap(mlir::subop::LookupEntryRefType::get(unwrap(t).getContext(), mlir::cast<mlir::subop::LookupAbleState>(unwrap(t))));
}
MlirTypeID mlirSubOpLookupEntryRefTypeGetTypeID() {
   return wrap(mlir::subop::LookupEntryRefType::getTypeID());
}
bool mlirTypeIsASubOpLookupEntryRefType(MlirType type) {
   return llvm::isa<mlir::subop::LookupEntryRefType>(unwrap(type));
}

MlirType mlirSubOpMultiMapEntryRefTypeGet(MlirType t) {
   return wrap(mlir::subop::MultiMapEntryRefType::get(unwrap(t).getContext(), mlir::cast<mlir::subop::MultiMapType>(unwrap(t))));
}
MlirTypeID mlirSubOpMultiMapEntryRefTypeGetTypeID() {
   return wrap(mlir::subop::MultiMapEntryRefType::getTypeID());
}
bool mlirTypeIsASubOpMultiMapEntryRefType(MlirType type) {
   return llvm::isa<mlir::subop::MultiMapEntryRefType>(unwrap(type));
}

MlirType mlirSubOpContinuousEntryRefTypeGet(MlirType t) {
   return wrap(mlir::subop::ContinuousEntryRefType::get(unwrap(t).getContext(), mlir::cast<mlir::subop::State>(unwrap(t))));
}
MlirTypeID mlirSubOpContinuousEntryRefTypeGetTypeID() {
   return wrap(mlir::subop::ContinuousEntryRefType::getTypeID());
}
bool mlirTypeIsASubOpContinuousEntryRefType(MlirType type) {
   return llvm::isa<mlir::subop::ContinuousEntryRefType>(unwrap(type));
}

MlirType mlirSubOpEntryListTypeGet(MlirType t) {
   return wrap(mlir::subop::ListType::get(unwrap(t).getContext(), mlir::cast<mlir::subop::StateEntryReference>(unwrap(t))));
}
MlirTypeID mlirSubOpEntryListTypeGetTypeID() {
   return wrap(mlir::subop::ListType::getTypeID());
}
bool mlirTypeIsASubOpEntryListType(MlirType type) {
   return llvm::isa<mlir::subop::ListType>(unwrap(type));
}
