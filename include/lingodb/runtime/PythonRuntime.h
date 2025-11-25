#ifndef LINGODB_RUNTIME_PYTHON_H
#define LINGODB_RUNTIME_PYTHON_H

#ifdef USE_CPYTHON_RUNTIME
#include "Python.h"
#else
// If CPython support is not enabled, forward-declare PyObject so headers compile.
struct _object;
typedef _object PyObject;
#endif

#include "helpers.h"
#include <cstdint>

namespace lingodb::runtime {
class PythonRuntime {
   public:
   static PyObject* createModule(runtime::VarLen32 modname, runtime::VarLen32 source);
   static PyObject* getAttr(PyObject* obj, runtime::VarLen32 attr);
   static void setAttr(PyObject* obj, runtime::VarLen32 attr, PyObject* value);
   static PyObject* call0(PyObject* callable);
   static PyObject* call1(PyObject* callable, PyObject* arg1);
   static PyObject* call2(PyObject* callable, PyObject* arg1, PyObject* arg2);
   static PyObject* call3(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3);
   static PyObject* call4(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4);
   static PyObject* call5(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5);
   static PyObject* call6(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6);
   static PyObject* call7(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7);
   static PyObject* call8(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7, PyObject* arg8);
   static PyObject* call9(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7, PyObject* arg8, PyObject* arg9);
   static PyObject* call10(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7, PyObject* arg8, PyObject* arg9, PyObject* arg10);
   static int64_t toInt64(PyObject* obj);
   static PyObject* fromInt64(int64_t value);
   static bool toBool(PyObject* obj);
   static PyObject* fromBool(bool value);
   static runtime::VarLen32 toVarLen32(PyObject* obj);
   static PyObject* fromVarLen32(runtime::VarLen32 value);
   static double toDouble(PyObject* obj);
   static PyObject* fromDouble(double value);
   static void decref(PyObject* obj);
   static void incref(PyObject* obj);
   static PyObject* import(runtime::VarLen32 val);
};
}

#endif //LINGODB_RUNTIME_PYTHON_H
