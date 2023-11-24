#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include <iostream>

#include "custom_dialects.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
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
