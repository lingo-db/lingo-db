#ifndef LINGODB_RUNTIME_PYTHON_H
#define LINGODB_RUNTIME_PYTHON_H
#include "Python.h"

namespace lingodb::runtime {
class PythonRuntime {
   public:
   static PyObject* test();
};
}

#endif //LINGODB_RUNTIME_PYTHON_H
