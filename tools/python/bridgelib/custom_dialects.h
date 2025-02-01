#ifndef MLIR_C_CUSTOM_DIALECTS_H
#define MLIR_C_CUSTOM_DIALECTS_H
#include "mlir-c/IR.h"
#ifdef __cplusplus
extern "C" {
#endif
//----------------------------------------------------------------------------------------------------------------------
// Util Dialect
//----------------------------------------------------------------------------------------------------------------------

MLIR_CAPI_EXPORTED MlirType mlirUtilRefTypeGet(MlirType elementType);
MLIR_CAPI_EXPORTED MlirTypeID mlirUtilRefTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsAUtilRefType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirUtilBufferTypeGet(MlirType elementType);
MLIR_CAPI_EXPORTED MlirTypeID mlirUtilBufferTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsAUtilBufferType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirUtilVarLen32TypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirTypeID mlirUtilVarLen32TypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsAUtilVarLen32Type(MlirType type);

//----------------------------------------------------------------------------------------------------------------------
// TupleStream Dialect
//----------------------------------------------------------------------------------------------------------------------

MLIR_CAPI_EXPORTED MlirType mlirTuplesTupleTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirTypeID mlirTuplesTupleTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsATuplesTupleType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirTuplesTupleStreamTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirTypeID mlirTuplesTupleStreamTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsATuplesTupleStreamType(MlirType type);

MLIR_CAPI_EXPORTED MlirAttribute mlirTuplesColumnDefAttributeGet(MlirContext context, MlirStringRef scope, MlirStringRef name, MlirType type);
MLIR_CAPI_EXPORTED bool mlirAttributeIsATuplesColumnDefAttribute(MlirAttribute attribute);

MLIR_CAPI_EXPORTED MlirAttribute mlirTuplesColumnRefAttributeGet(MlirContext context, MlirStringRef scope, MlirStringRef name);
MLIR_CAPI_EXPORTED bool mlirAttributeIsATuplesColumnRefAttribute(MlirAttribute attribute);

//----------------------------------------------------------------------------------------------------------------------
// DB Dialect
//----------------------------------------------------------------------------------------------------------------------

MLIR_CAPI_EXPORTED MlirType mlirDBNullableTypeGet(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirDBNullableTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsADBNullableType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirDBCharTypeGet(MlirContext context, size_t bytes);
MLIR_CAPI_EXPORTED MlirTypeID mlirDBCharTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsADBCharType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirDBDateTypeGet(MlirContext context, size_t unit);
MLIR_CAPI_EXPORTED MlirTypeID mlirDBDateTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsADBDateType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirDBIntervalTypeGet(MlirContext context, size_t unit);
MLIR_CAPI_EXPORTED MlirTypeID mlirDBIntervalTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsADBIntervalType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirDBTimestampTypeGet(MlirContext context, size_t unit);
MLIR_CAPI_EXPORTED MlirTypeID mlirDBTimestampTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsADBTimestampType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirDBDecimalTypeGet(MlirContext context, int p, int s);
MLIR_CAPI_EXPORTED MlirTypeID mlirDBDecimalTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsADBDecimalType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirDBStringTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirTypeID mlirDBStringTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsADBStringType(MlirType type);

//----------------------------------------------------------------------------------------------------------------------
// RelAlg Dialect
//----------------------------------------------------------------------------------------------------------------------

MLIR_CAPI_EXPORTED MlirAttribute mlirRelalgSortSpecAttributeGet(MlirAttribute colRef, size_t sortSpec);
MLIR_CAPI_EXPORTED bool mlirAttributeIsARelalgSortSpecAttribute(MlirAttribute attribute);

MLIR_CAPI_EXPORTED MlirAttribute mlirRelalgTableMetaDataAttrGetEmpty(MlirContext);
MLIR_CAPI_EXPORTED bool mlirAttributeIsARelalgTableMetaDataAttr(MlirAttribute attribute);

//----------------------------------------------------------------------------------------------------------------------
// SubOp Dialect
//----------------------------------------------------------------------------------------------------------------------
MLIR_CAPI_EXPORTED MlirAttribute mlirSubOpStateMembersAttributeGet(MlirAttribute names, MlirAttribute types);
MLIR_CAPI_EXPORTED bool mlirAttributeIsASubOpStateMembersAttribute(MlirAttribute attribute);

MLIR_CAPI_EXPORTED MlirType mlirSubOpTableTypeGet(MlirAttribute members);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpTableTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpTableType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpLocalTableTypeGet(MlirAttribute members,MlirAttribute columns);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpLocalTableTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpLocalTableType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpResultTableTypeGet(MlirAttribute members);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpResultTableTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpResultTableType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpSimpleStateTypeGet(MlirAttribute members);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpSimpleStateTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpSimpleStateType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpMapTypeGet(MlirAttribute keyMembers, MlirAttribute valMembers);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpMapTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpMapType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpMultiMapTypeGet(MlirAttribute keyMembers, MlirAttribute valMembers);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpMultiMapTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpMultiMapType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpBufferTypeGet(MlirAttribute members);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpBufferTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpBufferType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpArrayTypeGet(MlirAttribute members);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpArrayTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpArrayType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpContinuousViewTypeGet(MlirType basedOn);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpContinuousViewTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpContinuousViewType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpHeapTypeGet(MlirAttribute members, uint32_t maxElements);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpHeapTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpHeapType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpSegmentTreeViewTypeGet(MlirAttribute keyMembers, MlirAttribute valMembers);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpSegmentTreeViewTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpSegmentTreeViewType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpEntryTypeGet(MlirType t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpEntryTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpEntryType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpEntryRefTypeGet(MlirType t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpEntryRefTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpEntryRefType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpMapEntryRefTypeGet(MlirType t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpMapEntryRefTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpMapEntryRefType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpTableEntryRefTypeGet(MlirAttribute t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpTableEntryRefTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpTableEntryRefType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpLookupEntryRefTypeGet(MlirType t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpLookupEntryRefTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpLookupEntryRefType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpMultiMapEntryRefTypeGet(MlirType t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpMultiMapEntryRefTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpMultiMapEntryRefType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpContinuousEntryRefTypeGet(MlirType t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpContinuousEntryRefTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpContinuousEntryRefType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirSubOpEntryListTypeGet(MlirType t);
MLIR_CAPI_EXPORTED MlirTypeID mlirSubOpEntryListTypeGetTypeID();
MLIR_CAPI_EXPORTED bool mlirTypeIsASubOpEntryListType(MlirType type);
#ifdef __cplusplus
}
#endif
#endif //MLIR_C_CUSTOM_DIALECTS_H
