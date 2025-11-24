#include "lingodb/runtime/PythonRuntime.h"

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

}