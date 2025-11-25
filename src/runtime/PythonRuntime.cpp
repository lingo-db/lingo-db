#include "lingodb/runtime/PythonRuntime.h"

#include <iostream>

namespace lingodb::runtime {
PyObject* PythonRuntime::import(runtime::VarLen32 val) {
   auto valStr = val.str();
   return PyImport_ImportModule(valStr.c_str());
}

inline void throw_python_error() {
   PyObject *ptype, *pvalue, *ptraceback;
   PyErr_Fetch(&ptype, &pvalue, &ptraceback);
   PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

   std::string msg = "Unknown Python error";

   if (pvalue) {
      PyObject* str_exc = PyObject_Str(pvalue);  // or PyObject_Repr
      if (str_exc) {
         msg = PyUnicode_AsUTF8(str_exc);
         Py_DECREF(str_exc);
      }
   }

   // Cleanup references
   Py_XDECREF(ptype);
   Py_XDECREF(pvalue);
   Py_XDECREF(ptraceback);

   throw std::runtime_error(msg);
}

PyObject* PythonRuntime::createModule(runtime::VarLen32 modname, runtime::VarLen32 source){
   auto modStr=modname.str();
   auto sourceStr = source.str();

   PyObject *sys_modules = PyImport_GetModuleDict();  // borrowed

   // 1) Check if module already exists
   PyObject *mod = PyDict_GetItemString(sys_modules, modStr.c_str());  // borrowed
   if (mod) {
      Py_INCREF(mod);      // convert borrowed → owned
      return mod;          // caller owns reference
   }

   // 2) Create new module
   mod = PyModule_New(modname.data());
   if (!mod) return NULL;

   PyObject *globals = PyModule_GetDict(mod);

   // Ensure __builtins__
   if (!PyDict_GetItemString(globals, "__builtins__")) {
      PyObject *builtins = PyEval_GetBuiltins();
      if (PyDict_SetItemString(globals, "__builtins__", builtins) < 0) {
         Py_DECREF(mod);
         return NULL;
      }
   }

   // Execute Python source inside module
   PyObject *res = PyRun_String(
       sourceStr.c_str(), Py_file_input, globals, globals
   );
   if (!res) {
      throw_python_error();
      Py_DECREF(mod);
      return NULL;
   }
   Py_DECREF(res);

   // 3) Insert into sys.modules
   if (PyDict_SetItemString(sys_modules, modStr.c_str(), mod) < 0) {
      Py_DECREF(mod);
      return NULL;
   }

   // sys.modules now owns a reference; caller gets a new one
   //Py_INCREF(mod);
   return mod;
}
PyObject* PythonRuntime::getAttr(PyObject* obj, runtime::VarLen32 attr){
   auto attrStr = attr.str();
   return PyObject_GetAttrString(obj, attrStr.c_str());
}
PyObject* PythonRuntime::call0(PyObject* callable){
   return PyObject_CallFunctionObjArgs(callable, NULL);
}
PyObject* PythonRuntime::call1(PyObject* callable, PyObject* arg1){
        return PyObject_CallFunctionObjArgs(callable, arg1, NULL);
}
PyObject* PythonRuntime::call2(PyObject* callable, PyObject* arg1, PyObject* arg2){
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, NULL);
}
PyObject* PythonRuntime::call3(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3) {
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, NULL);
}
PyObject* PythonRuntime::call4(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4){
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, NULL);
}
PyObject* PythonRuntime::call5(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5){
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, NULL);
}
PyObject* PythonRuntime::call6(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6){
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, arg6, NULL);
}
PyObject* PythonRuntime::call7(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7){
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, NULL);
}
PyObject* PythonRuntime::call8(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7, PyObject* arg8){
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, NULL);
}
PyObject* PythonRuntime::call9(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7, PyObject* arg8, PyObject* arg9){
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, NULL);
}
PyObject* PythonRuntime::call10(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7, PyObject* arg8, PyObject* arg9, PyObject* arg10){
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, NULL);
}
void PythonRuntime::setAttr(PyObject* obj, runtime::VarLen32 attr, PyObject* value) {
   auto attrStr = attr.str();
   PyObject_SetAttrString(obj, attrStr.c_str(), value);
}


int64_t PythonRuntime::toInt64(PyObject* obj) {
   return PyLong_AsLongLong(obj);
}
PyObject* PythonRuntime::fromInt64(int64_t value){
   return PyLong_FromLongLong(value);
}
bool PythonRuntime::toBool(PyObject* obj) {
   return PyObject_IsTrue(obj); //todo
}
PyObject* PythonRuntime::fromBool(bool value){
   return PyBool_FromLong(value ? 1 : 0);
}
runtime::VarLen32 PythonRuntime::toVarLen32(PyObject* obj){
   if (PyUnicode_Check(obj)) {
      Py_ssize_t size;
      const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
      if (data) {
         return runtime::VarLen32::fromString(std::string_view(data, size), StorageClass::REFCOUNTED);
      }
   }
   throw std::runtime_error("Object is not a string");
}
PyObject* PythonRuntime::fromVarLen32(runtime::VarLen32 value){
   auto str = value.str();
   return PyUnicode_FromStringAndSize(str.data(), str.size());
}
double PythonRuntime::toDouble(PyObject* obj){
   return PyFloat_AsDouble(obj);
}
PyObject* PythonRuntime::fromDouble(double value){
   return PyFloat_FromDouble(value);
}
void PythonRuntime::decref(PyObject* obj){
   Py_DECREF(obj);
}
void PythonRuntime::incref(PyObject* obj){
   Py_INCREF(obj);
}


}