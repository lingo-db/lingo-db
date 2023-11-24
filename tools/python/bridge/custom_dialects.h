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

#ifdef __cplusplus
}
#endif
#endif //MLIR_C_CUSTOM_DIALECTS_H
