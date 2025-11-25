#ifndef LINGODB_RUNTIME_PYTHON_H
#define LINGODB_RUNTIME_PYTHON_H
#include "Python.h"
#include "helpers.h"

namespace lingodb::runtime {
class PythonRuntime {
   public:
   static PyObject* createModule(runtime::VarLen32 modname, runtime::VarLen32 source);
   static PyObject* getAttr(PyObject* obj, runtime::VarLen32 attr);
   static PyObject* call0(PyObject* callable);
   static PyObject* call1(PyObject* callable, PyObject* arg1);
   static PyObject* call2(PyObject* callable, PyObject* arg1, PyObject* arg2);
   static PyObject* call3(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3);
   static int64_t toInt64(PyObject* obj);
   static double toDouble(PyObject* obj);
   static PyObject* fromDouble(double value);
   static void decref(PyObject* obj);
   static PyObject* test();
};
}

#endif //LINGODB_RUNTIME_PYTHON_H
