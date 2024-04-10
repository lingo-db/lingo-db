#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "custom_dialects.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <iostream>
namespace {
static MlirStringRef toMlirStringRef(const std::string& s) {
   return mlirStringRefCreate(s.data(), s.size());
}
}
PYBIND11_MODULE(mlir_lingodb, m) {
   //----------------------------------------------------------------------------------------------------------------------
   // Util Dialect
   //----------------------------------------------------------------------------------------------------------------------
   auto utilModule = m.def_submodule("util", "extensions required for util dialect");
   mlir::python::adaptors::mlir_type_subclass(utilModule, "RefType", mlirTypeIsAUtilRefType, mlirUtilRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirType type) {
            return cls(mlirUtilRefTypeGet(type));
         },
         py::arg("cls"), py::arg("type"));
   mlir::python::adaptors::mlir_type_subclass(utilModule, "VarLen32Type", mlirTypeIsAUtilVarLen32Type, mlirUtilVarLen32TypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context) {
            return cls(mlirUtilVarLen32TypeGet(context));
         },
         py::arg("cls"), py::arg("context"));

   //----------------------------------------------------------------------------------------------------------------------
   // TupleStream Dialect
   //----------------------------------------------------------------------------------------------------------------------

   auto tuplesModule = m.def_submodule("tuples", "extensions required for tuples dialect");
   mlir::python::adaptors::mlir_type_subclass(tuplesModule, "TupleType", mlirTypeIsATuplesTupleType, mlirTuplesTupleTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context) {
            return cls(mlirTuplesTupleTypeGet(context));
         },
         py::arg("cls"), py::arg("context"));
   mlir::python::adaptors::mlir_type_subclass(tuplesModule, "TupleStreamType", mlirTypeIsATuplesTupleStreamType, mlirTuplesTupleStreamTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context) {
            return cls(mlirTuplesTupleStreamTypeGet(context));
         },
         py::arg("cls"), py::arg("context"));

   mlir::python::adaptors::mlir_attribute_subclass(tuplesModule, "ColumnDefAttr", mlirAttributeIsATuplesColumnDefAttribute)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context, std::string scope, std::string name, MlirType type) {
            return cls(mlirTuplesColumnDefAttributeGet(context, toMlirStringRef(scope), toMlirStringRef(name), type));
         },
         py::arg("cls"), py::arg("context"), py::arg("scope"), py::arg("name"), py::arg("type"));
   mlir::python::adaptors::mlir_attribute_subclass(tuplesModule, "ColumnRefAttr", mlirAttributeIsATuplesColumnRefAttribute)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context, std::string scope, std::string name) {
            return cls(mlirTuplesColumnRefAttributeGet(context, toMlirStringRef(scope), toMlirStringRef(name)));
         },
         py::arg("cls"), py::arg("context"), py::arg("scope"), py::arg("name"));
   //----------------------------------------------------------------------------------------------------------------------
   // DB Dialect
   //----------------------------------------------------------------------------------------------------------------------
   auto dbModule = m.def_submodule("db", "extensions required for db dialect");
   mlir::python::adaptors::mlir_type_subclass(dbModule, "NullableType", mlirTypeIsADBNullableType, mlirDBNullableTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirType type) {
            return cls(mlirDBNullableTypeGet(type));
         },
         py::arg("cls"), py::arg("type"));
   mlir::python::adaptors::mlir_type_subclass(dbModule, "CharType", mlirTypeIsADBCharType, mlirDBCharTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context, size_t bytes) {
            return cls(mlirDBCharTypeGet(context, bytes));
         },
         py::arg("cls"), py::arg("context"), py::arg("bytes"));
   mlir::python::adaptors::mlir_type_subclass(dbModule, "DateType", mlirTypeIsADBDateType, mlirDBDateTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context, size_t unit) {
            return cls(mlirDBDateTypeGet(context, unit));
         },
         py::arg("cls"), py::arg("context"), py::arg("unit"));
   mlir::python::adaptors::mlir_type_subclass(dbModule, "IntervalType", mlirTypeIsADBIntervalType, mlirDBIntervalTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context, size_t unit) {
            return cls(mlirDBIntervalTypeGet(context, unit));
         },
         py::arg("cls"), py::arg("context"), py::arg("unit"));
   mlir::python::adaptors::mlir_type_subclass(dbModule, "TimestampType", mlirTypeIsADBTimestampType, mlirDBTimestampTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context, size_t unit) {
            return cls(mlirDBTimestampTypeGet(context, unit));
         },
         py::arg("cls"), py::arg("context"), py::arg("unit"));
   mlir::python::adaptors::mlir_type_subclass(dbModule, "DecimalType", mlirTypeIsADBDecimalType, mlirDBDecimalTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context, size_t p, size_t s) {
            return cls(mlirDBDecimalTypeGet(context, p, s));
         },
         py::arg("cls"), py::arg("context"), py::arg("p"), py::arg("s"));
   mlir::python::adaptors::mlir_type_subclass(dbModule, "StringType", mlirTypeIsADBStringType, mlirDBStringTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirContext context) {
            return cls(mlirDBStringTypeGet(context));
         },
         py::arg("cls"), py::arg("context"));
   //----------------------------------------------------------------------------------------------------------------------
   // RelAlg Dialect
   //----------------------------------------------------------------------------------------------------------------------
   auto relalgModule = m.def_submodule("relalg", "extensions required for relalg dialect");
   mlir::python::adaptors::mlir_attribute_subclass(relalgModule, "SortSpecificationAttr", mlirAttributeIsARelalgSortSpecAttribute)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute colRef, size_t sortSpec) {
            return cls(mlirRelalgSortSpecAttributeGet(colRef, sortSpec));
         },
         py::arg("cls"), py::arg("colRef"), py::arg("sortSpec"));
   mlir::python::adaptors::mlir_attribute_subclass(relalgModule, "TableMetaDataAttr", mlirAttributeIsARelalgTableMetaDataAttr)
      .def_classmethod(
         "get_empty",
         [](py::object cls, MlirContext context) {
            return cls(mlirRelalgTableMetaDataAttrGetEmpty(context));
         },
         py::arg("cls"), py::arg("context"));

   //----------------------------------------------------------------------------------------------------------------------
   // SubOp Dialect
   //----------------------------------------------------------------------------------------------------------------------
   auto subOpModule = m.def_submodule("subop", "extensions required for sub-operator dialect");
   mlir::python::adaptors::mlir_attribute_subclass(subOpModule, "StateMembersAttr", mlirAttributeIsASubOpStateMembersAttribute)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute names, MlirAttribute types) {
            return cls(mlirSubOpStateMembersAttributeGet(names, types));
         },
         py::arg("cls"), py::arg("names"), py::arg("types"));
   mlir::python::adaptors::mlir_type_subclass(subOpModule, "TableType", mlirTypeIsASubOpTableType, mlirSubOpTableTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute members) {
            return cls(mlirSubOpTableTypeGet(members));
         },
         py::arg("cls"), py::arg("members"));
   mlir::python::adaptors::mlir_type_subclass(subOpModule, "LocalTableType", mlirTypeIsASubOpLocalTableType, mlirSubOpLocalTableTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute members,MlirAttribute columns) {
            return cls(mlirSubOpLocalTableTypeGet(members,columns));
         },
         py::arg("cls"), py::arg("members"),py::arg("columns"));
   mlir::python::adaptors::mlir_type_subclass(subOpModule, "ResultTableType", mlirTypeIsASubOpResultTableType, mlirSubOpResultTableTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute members) {
            return cls(mlirSubOpResultTableTypeGet(members));
         },
         py::arg("cls"), py::arg("members"));
   mlir::python::adaptors::mlir_type_subclass(subOpModule, "SimpleStateType", mlirTypeIsASubOpSimpleStateType, mlirSubOpSimpleStateTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute members) {
            return cls(mlirSubOpSimpleStateTypeGet(members));
         },
         py::arg("cls"), py::arg("members"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "MapType", mlirTypeIsASubOpMapType, mlirSubOpMapTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute keyMembers, MlirAttribute valMembers) {
            return cls(mlirSubOpMapTypeGet(keyMembers, valMembers));
         },
         py::arg("cls"), py::arg("keyMembers"), py::arg("valMembers"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "MultiMapType", mlirTypeIsASubOpMultiMapType, mlirSubOpMultiMapTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute keyMembers, MlirAttribute valMembers) {
            return cls(mlirSubOpMultiMapTypeGet(keyMembers, valMembers));
         },
         py::arg("cls"), py::arg("keyMembers"), py::arg("valMembers"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "BufferType", mlirTypeIsASubOpBufferType, mlirSubOpBufferTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute members) {
            return cls(mlirSubOpBufferTypeGet(members));
         },
         py::arg("cls"), py::arg("members"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "ArrayType", mlirTypeIsASubOpArrayType, mlirSubOpArrayTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute members) {
            return cls(mlirSubOpArrayTypeGet(members));
         },
         py::arg("cls"), py::arg("members"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "ContinuousViewType", mlirTypeIsASubOpContinuousViewType, mlirSubOpContinuousViewTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirType basedOn) {
            return cls(mlirSubOpContinuousViewTypeGet(basedOn));
         },
         py::arg("cls"), py::arg("basedOn"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "HeapType", mlirTypeIsASubOpHeapType, mlirSubOpHeapTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute members, uint32_t maxElements) {
            return cls(mlirSubOpHeapTypeGet(members,maxElements));
         },
         py::arg("cls"), py::arg("members"), py::arg("maxElements"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "SegmentTreeViewType", mlirTypeIsASubOpSegmentTreeViewType, mlirSubOpSegmentTreeViewTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute keyMembers, MlirAttribute valMembers) {
            return cls(mlirSubOpSegmentTreeViewTypeGet(keyMembers, valMembers));
         },
         py::arg("cls"), py::arg("keyMembers"), py::arg("valMembers"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "EntryType", mlirTypeIsASubOpEntryType, mlirSubOpEntryTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirType t) {
            return cls(mlirSubOpEntryTypeGet(t));
         },
         py::arg("cls"), py::arg("t"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "EntryRefType", mlirTypeIsASubOpEntryRefType, mlirSubOpEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirType t) {
            return cls(mlirSubOpEntryRefTypeGet(t));
         },
         py::arg("cls"), py::arg("t"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "MapEntryRefType", mlirTypeIsASubOpMapEntryRefType, mlirSubOpMapEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirType t) {
            return cls(mlirSubOpMapEntryRefTypeGet(t));
         },
         py::arg("cls"), py::arg("t"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "TableEntryRefType", mlirTypeIsASubOpTableEntryRefType, mlirSubOpTableEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirAttribute a) {
            return cls(mlirSubOpTableEntryRefTypeGet(a));
         },
         py::arg("cls"), py::arg("a"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "LookupEntryRefType", mlirTypeIsASubOpLookupEntryRefType, mlirSubOpLookupEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirType t) {
            return cls(mlirSubOpLookupEntryRefTypeGet(t));
         },
         py::arg("cls"), py::arg("t"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "MultiMapEntryRefType", mlirTypeIsASubOpMultiMapEntryRefType, mlirSubOpMultiMapEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirType t) {
            return cls(mlirSubOpMultiMapEntryRefTypeGet(t));
         },
         py::arg("cls"), py::arg("t"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "ContinuousEntryRefType", mlirTypeIsASubOpContinuousEntryRefType, mlirSubOpContinuousEntryRefTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirType t) {
            return cls(mlirSubOpContinuousEntryRefTypeGet(t));
         },
         py::arg("cls"), py::arg("t"));

   mlir::python::adaptors::mlir_type_subclass(subOpModule, "EntryListType", mlirTypeIsASubOpEntryListType, mlirSubOpEntryListTypeGetTypeID)
      .def_classmethod(
         "get",
         [](py::object cls, MlirType t) {
            return cls(mlirSubOpEntryListTypeGet(t));
         },
         py::arg("cls"), py::arg("t"));
}
