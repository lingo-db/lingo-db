#include "lingodb/runtime/PythonRuntime.h"

#include <iostream>

namespace lingodb::runtime {
PyObject* PythonRuntime::test() {
   PyObject *globals = PyDict_New();
   PyObject* dict2 = PyDict_New();
   auto res = PyRun_String("print(1)", Py_single_input, globals, dict2);
   Py_DECREF(res);
   Py_DECREF(globals);
   Py_DECREF(dict2);
   return res;
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
int64_t PythonRuntime::toInt64(PyObject* obj) {
   return PyLong_AsLongLong(obj);
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


}