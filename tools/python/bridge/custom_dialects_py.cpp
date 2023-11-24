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
}
