#ifndef LINGODB_RUNTIME_PYTHONRUNTIME_H
#define LINGODB_RUNTIME_PYTHONRUNTIME_H

#ifdef USE_CPYTHON_RUNTIME
#include "Python.h"
using PyObjectPtr = PyObject*;
#else
#ifdef USE_CPYTHON_WASM_RUNTIME
// In the WASM runtime, PyObject* lives in linear memory and is identified by a
// 32-bit offset. The host side never dereferences it natively.
#include <cstdint>
using PyObjectPtr = uint32_t;
#else
// If CPython support is not enabled, forward-declare PyObject so headers compile.
struct _object;
typedef _object PyObject;
using PyObjectPtr = PyObject*;
#endif
#endif

#include "helpers.h"
#include <cstdint>
#include <vector>
namespace lingodb::wasm {
struct WASMSession;
} // namespace lingodb::wasm

namespace lingodb::runtime {
class ArrowTable;
#ifdef USE_CPYTHON_RUNTIME
struct PythonExtState {
   std::vector<PyObjectPtr> cachedObjects;
   PyObjectPtr get(size_t idx) {
      if (idx >= cachedObjects.size()) {
         cachedObjects.resize(idx + 1, nullptr);
      }
      return cachedObjects[idx];
   }
   void set(size_t idx, PyObjectPtr obj) {
      if (idx >= cachedObjects.size()) {
         cachedObjects.resize(idx + 1, nullptr);
      }
      cachedObjects[idx] = obj;
   }
   void clearCache() {
      cachedObjects.clear();
   }
};
#else
struct PythonExtState;
#endif
class PythonRuntime {
   public:
   static PyObjectPtr createModule(size_t x, runtime::VarLen32 modname, runtime::VarLen32 source);
   static PyObjectPtr getAttr(PyObjectPtr obj, runtime::VarLen32 attr);
   static PyObjectPtr call0(PyObjectPtr callable);
   static PyObjectPtr call1(PyObjectPtr callable, PyObjectPtr arg1);
   static PyObjectPtr call2(PyObjectPtr callable, PyObjectPtr arg1, PyObjectPtr arg2);
   static PyObjectPtr call3(PyObjectPtr callable, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3);
   static PyObjectPtr call4(PyObjectPtr callable, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4);
   static PyObjectPtr call5(PyObjectPtr callable, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5);
   static PyObjectPtr call6(PyObjectPtr callable, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5, PyObjectPtr arg6);
   static PyObjectPtr call7(PyObjectPtr callable, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5, PyObjectPtr arg6, PyObjectPtr arg7);
   static PyObjectPtr call8(PyObjectPtr callable, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5, PyObjectPtr arg6, PyObjectPtr arg7, PyObjectPtr arg8);
   static PyObjectPtr call9(PyObjectPtr callable, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5, PyObjectPtr arg6, PyObjectPtr arg7, PyObjectPtr arg8, PyObjectPtr arg9);
   static PyObjectPtr call10(PyObjectPtr callable, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5, PyObjectPtr arg6, PyObjectPtr arg7, PyObjectPtr arg8, PyObjectPtr arg9, PyObjectPtr arg10);
   static int64_t toInt64(PyObjectPtr obj);
   static PyObjectPtr fromInt64(int64_t value);
   static bool toBool(PyObjectPtr obj);
   static PyObjectPtr fromBool(bool value);
   static PyObjectPtr fromDate(int64_t value);
   static runtime::VarLen32 toVarLen32(PyObjectPtr obj);
   static PyObjectPtr fromVarLen32(runtime::VarLen32 value);
   static double toDouble(PyObjectPtr obj);
   static PyObjectPtr fromDouble(double value);
   // Convert between a runtime-owned arrow::Table (lingodb::runtime::ArrowTable*)
   // and a pyarrow.Table PyObject. Used by tabular Python UDFs.
   static PyObjectPtr fromArrowTable(ArrowTable* table);
   static ArrowTable* toArrowTable(PyObjectPtr obj);
   static void decref(PyObjectPtr obj);
   static void incref(PyObjectPtr obj);
   static PyObjectPtr import(size_t x, runtime::VarLen32 val);
   //CPython specific
   static PythonExtState* createPythonExtState();
   //WASM CPython specific
   static void setWasmSession(wasm::WASMSession* session);
};
} // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_PYTHONRUNTIME_H
