#include <nanobind/nanobind.h>

#include "custom_dialects.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include <iostream>
namespace {
static MlirStringRef toMlirStringRef(const std::string& s) {
   return mlirStringRefCreate(s.data(), s.size());
}
}

namespace nb = nanobind;

NB_MODULE(mlir_lingodb, m) {
   //----------------------------------------------------------------------------------------------------------------------
   // Util Dialect
   //----------------------------------------------------------------------------------------------------------------------
   auto utilModule = m.def_submodule("util", "extensions required for util dialect");
   mlir::python::nanobind_adaptors::mlir_type_subclass(utilModule, "RefType", mlirTypeIsAUtilRefType, mlirUtilRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType type) {
            return cls(mlirUtilRefTypeGet(type));
         },
         nb::arg("cls"), nb::arg("type"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(utilModule, "BufferType", mlirTypeIsAUtilBufferType, mlirUtilBufferTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType type) {
            return cls(mlirUtilBufferTypeGet(type));
         },
         nb::arg("cls"), nb::arg("type"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(utilModule, "VarLen32Type", mlirTypeIsAUtilVarLen32Type, mlirUtilVarLen32TypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context) {
            return cls(mlirUtilVarLen32TypeGet(context));
         },
         nb::arg("cls"), nb::arg("context"));

   //----------------------------------------------------------------------------------------------------------------------
   // TupleStream Dialect
   //----------------------------------------------------------------------------------------------------------------------

   auto tuplesModule = m.def_submodule("tuples", "extensions required for tuples dialect");
   mlir::python::nanobind_adaptors::mlir_type_subclass(tuplesModule, "TupleType", mlirTypeIsATuplesTupleType, mlirTuplesTupleTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context) {
            return cls(mlirTuplesTupleTypeGet(context));
         },
         nb::arg("cls"), nb::arg("context"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(tuplesModule, "TupleStreamType", mlirTypeIsATuplesTupleStreamType, mlirTuplesTupleStreamTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context) {
            return cls(mlirTuplesTupleStreamTypeGet(context));
         },
         nb::arg("cls"), nb::arg("context"));

   mlir::python::nanobind_adaptors::mlir_attribute_subclass(tuplesModule, "ColumnDefAttr", mlirAttributeIsATuplesColumnDefAttribute)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context, std::string scope, std::string name, MlirType type) {
            return cls(mlirTuplesColumnDefAttributeGet(context, toMlirStringRef(scope), toMlirStringRef(name), type));
         },
         nb::arg("cls"), nb::arg("context"), nb::arg("scope"), nb::arg("name"), nb::arg("type"));
   mlir::python::nanobind_adaptors::mlir_attribute_subclass(tuplesModule, "ColumnRefAttr", mlirAttributeIsATuplesColumnRefAttribute)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context, std::string scope, std::string name) {
            return cls(mlirTuplesColumnRefAttributeGet(context, toMlirStringRef(scope), toMlirStringRef(name)));
         },
         nb::arg("cls"), nb::arg("context"), nb::arg("scope"), nb::arg("name"));
   //----------------------------------------------------------------------------------------------------------------------
   // DB Dialect
   //----------------------------------------------------------------------------------------------------------------------
   auto dbModule = m.def_submodule("db", "extensions required for db dialect");
   mlir::python::nanobind_adaptors::mlir_type_subclass(dbModule, "NullableType", mlirTypeIsADBNullableType, mlirDBNullableTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType type) {
            return cls(mlirDBNullableTypeGet(type));
         },
         nb::arg("cls"), nb::arg("type"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(dbModule, "CharType", mlirTypeIsADBCharType, mlirDBCharTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context, size_t bytes) {
            return cls(mlirDBCharTypeGet(context, bytes));
         },
         nb::arg("cls"), nb::arg("context"), nb::arg("bytes"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(dbModule, "DateType", mlirTypeIsADBDateType, mlirDBDateTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context, size_t unit) {
            return cls(mlirDBDateTypeGet(context, unit));
         },
         nb::arg("cls"), nb::arg("context"), nb::arg("unit"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(dbModule, "IntervalType", mlirTypeIsADBIntervalType, mlirDBIntervalTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context, size_t unit) {
            return cls(mlirDBIntervalTypeGet(context, unit));
         },
         nb::arg("cls"), nb::arg("context"), nb::arg("unit"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(dbModule, "TimestampType", mlirTypeIsADBTimestampType, mlirDBTimestampTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context, size_t unit) {
            return cls(mlirDBTimestampTypeGet(context, unit));
         },
         nb::arg("cls"), nb::arg("context"), nb::arg("unit"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(dbModule, "DecimalType", mlirTypeIsADBDecimalType, mlirDBDecimalTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context, size_t p, size_t s) {
            return cls(mlirDBDecimalTypeGet(context, p, s));
         },
         nb::arg("cls"), nb::arg("context"), nb::arg("p"), nb::arg("s"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(dbModule, "StringType", mlirTypeIsADBStringType, mlirDBStringTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirContext context) {
            return cls(mlirDBStringTypeGet(context));
         },
         nb::arg("cls"), nb::arg("context"));
   //----------------------------------------------------------------------------------------------------------------------
   // RelAlg Dialect
   //----------------------------------------------------------------------------------------------------------------------
   auto relalgModule = m.def_submodule("relalg", "extensions required for relalg dialect");
   mlir::python::nanobind_adaptors::mlir_attribute_subclass(relalgModule, "SortSpecificationAttr", mlirAttributeIsARelalgSortSpecAttribute)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute colRef, size_t sortSpec) {
            return cls(mlirRelalgSortSpecAttributeGet(colRef, sortSpec));
         },
         nb::arg("cls"), nb::arg("colRef"), nb::arg("sortSpec"));
   mlir::python::nanobind_adaptors::mlir_attribute_subclass(relalgModule, "TableMetaDataAttr", mlirAttributeIsARelalgTableMetaDataAttr)
      .def_classmethod(
         "get_empty",
         [](nb::object cls, MlirContext context) {
            return cls(mlirRelalgTableMetaDataAttrGetEmpty(context));
         },
         nb::arg("cls"), nb::arg("context"));

   //----------------------------------------------------------------------------------------------------------------------
   // SubOp Dialect
   //----------------------------------------------------------------------------------------------------------------------
   auto subOpModule = m.def_submodule("subop", "extensions required for sub-operator dialect");
   mlir::python::nanobind_adaptors::mlir_attribute_subclass(subOpModule, "StateMembersAttr", mlirAttributeIsASubOpStateMembersAttribute)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute names, MlirAttribute types) {
            return cls(mlirSubOpStateMembersAttributeGet(names, types));
         },
         nb::arg("cls"), nb::arg("names"), nb::arg("types"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "TableType", mlirTypeIsASubOpTableType, mlirSubOpTableTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute members) {
            return cls(mlirSubOpTableTypeGet(members));
         },
         nb::arg("cls"), nb::arg("members"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "LocalTableType", mlirTypeIsASubOpLocalTableType, mlirSubOpLocalTableTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute members,MlirAttribute columns) {
            return cls(mlirSubOpLocalTableTypeGet(members,columns));
         },
         nb::arg("cls"), nb::arg("members"),nb::arg("columns"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "ResultTableType", mlirTypeIsASubOpResultTableType, mlirSubOpResultTableTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute members) {
            return cls(mlirSubOpResultTableTypeGet(members));
         },
         nb::arg("cls"), nb::arg("members"));
   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "SimpleStateType", mlirTypeIsASubOpSimpleStateType, mlirSubOpSimpleStateTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute members) {
            return cls(mlirSubOpSimpleStateTypeGet(members));
         },
         nb::arg("cls"), nb::arg("members"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "MapType", mlirTypeIsASubOpMapType, mlirSubOpMapTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute keyMembers, MlirAttribute valMembers) {
            return cls(mlirSubOpMapTypeGet(keyMembers, valMembers));
         },
         nb::arg("cls"), nb::arg("keyMembers"), nb::arg("valMembers"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "MultiMapType", mlirTypeIsASubOpMultiMapType, mlirSubOpMultiMapTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute keyMembers, MlirAttribute valMembers) {
            return cls(mlirSubOpMultiMapTypeGet(keyMembers, valMembers));
         },
         nb::arg("cls"), nb::arg("keyMembers"), nb::arg("valMembers"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "BufferType", mlirTypeIsASubOpBufferType, mlirSubOpBufferTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute members) {
            return cls(mlirSubOpBufferTypeGet(members));
         },
         nb::arg("cls"), nb::arg("members"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "ArrayType", mlirTypeIsASubOpArrayType, mlirSubOpArrayTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute members) {
            return cls(mlirSubOpArrayTypeGet(members));
         },
         nb::arg("cls"), nb::arg("members"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "ContinuousViewType", mlirTypeIsASubOpContinuousViewType, mlirSubOpContinuousViewTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType basedOn) {
            return cls(mlirSubOpContinuousViewTypeGet(basedOn));
         },
         nb::arg("cls"), nb::arg("basedOn"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "HeapType", mlirTypeIsASubOpHeapType, mlirSubOpHeapTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute members, uint32_t maxElements) {
            return cls(mlirSubOpHeapTypeGet(members,maxElements));
         },
         nb::arg("cls"), nb::arg("members"), nb::arg("maxElements"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "SegmentTreeViewType", mlirTypeIsASubOpSegmentTreeViewType, mlirSubOpSegmentTreeViewTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute keyMembers, MlirAttribute valMembers) {
            return cls(mlirSubOpSegmentTreeViewTypeGet(keyMembers, valMembers));
         },
         nb::arg("cls"), nb::arg("keyMembers"), nb::arg("valMembers"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "EntryType", mlirTypeIsASubOpEntryType, mlirSubOpEntryTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType t) {
            return cls(mlirSubOpEntryTypeGet(t));
         },
         nb::arg("cls"), nb::arg("t"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "EntryRefType", mlirTypeIsASubOpEntryRefType, mlirSubOpEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType t) {
            return cls(mlirSubOpEntryRefTypeGet(t));
         },
         nb::arg("cls"), nb::arg("t"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "MapEntryRefType", mlirTypeIsASubOpMapEntryRefType, mlirSubOpMapEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType t) {
            return cls(mlirSubOpMapEntryRefTypeGet(t));
         },
         nb::arg("cls"), nb::arg("t"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "TableEntryRefType", mlirTypeIsASubOpTableEntryRefType, mlirSubOpTableEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirAttribute a) {
            return cls(mlirSubOpTableEntryRefTypeGet(a));
         },
         nb::arg("cls"), nb::arg("a"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "LookupEntryRefType", mlirTypeIsASubOpLookupEntryRefType, mlirSubOpLookupEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType t) {
            return cls(mlirSubOpLookupEntryRefTypeGet(t));
         },
         nb::arg("cls"), nb::arg("t"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "MultiMapEntryRefType", mlirTypeIsASubOpMultiMapEntryRefType, mlirSubOpMultiMapEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType t) {
            return cls(mlirSubOpMultiMapEntryRefTypeGet(t));
         },
         nb::arg("cls"), nb::arg("t"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "ContinuousEntryRefType", mlirTypeIsASubOpContinuousEntryRefType, mlirSubOpContinuousEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType t) {
            return cls(mlirSubOpContinuousEntryRefTypeGet(t));
         },
         nb::arg("cls"), nb::arg("t"));

   mlir::python::nanobind_adaptors::mlir_type_subclass(subOpModule, "EntryListType", mlirTypeIsASubOpEntryListType, mlirSubOpEntryListTypeGetTypeID)
      .def_classmethod(
         "get",
         [](nb::object cls, MlirType t) {
            return cls(mlirSubOpEntryListTypeGet(t));
         },
         nb::arg("cls"), nb::arg("t"));
}
