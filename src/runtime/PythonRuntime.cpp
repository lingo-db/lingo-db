#include "lingodb/runtime/PythonRuntime.h"
#include "lingodb/runtime/ArrowTable.h"
#include "lingodb/runtime/DateRuntime.h"
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/Session.h"
#include "lingodb/scheduler/Scheduler.h"
#ifdef USE_CPYTHON_WASM_RUNTIME
#include "lingodb/runtime/WASM.h"
#endif
#ifdef USE_CPYTHON_RUNTIME
#include "datetime.h"
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#endif
#include <iostream>
#include <stdexcept>
namespace lingodb::runtime {
#ifdef USE_CPYTHON_RUNTIME

PyObject* PythonRuntime::import(size_t x, runtime::VarLen32 val) {
   auto& extState = getCurrentExecutionContext()->getSession().pythonExtStates[scheduler::currentWorkerId()];
   PyObject* cached = extState->get(x);
   if (cached) {
      return cached;
   }
   auto valStr = val.str();
   auto res = PyImport_ImportModule(valStr.c_str());
   extState->set(x, res);
   return res;
}
PythonExtState* PythonRuntime::createPythonExtState() {
   return new PythonExtState();
}

inline void throw_python_error() {
   PyObject *ptype, *pvalue, *ptraceback;
   PyErr_Fetch(&ptype, &pvalue, &ptraceback);
   PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

   std::string msg = "Unknown Python error";

   if (pvalue) {
      PyObject* str_exc = PyObject_Str(pvalue);
      if (str_exc) {
         msg = PyUnicode_AsUTF8(str_exc);
         Py_DECREF(str_exc);
      }
   }

   Py_XDECREF(ptype);
   Py_XDECREF(pvalue);
   Py_XDECREF(ptraceback);

   throw std::runtime_error(msg);
}

PyObject* PythonRuntime::createModule(size_t x, runtime::VarLen32 modname, runtime::VarLen32 source) {
   auto& extState = getCurrentExecutionContext()->getSession().pythonExtStates[scheduler::currentWorkerId()];
   PyObject* cached = extState->get(x);
   if (cached) {
      return cached;
   }
   auto modStr = modname.str();

   PyObject* sys_modules = PyImport_GetModuleDict(); // borrowed

   // 1) Check if module already exists
   PyObject* mod = PyDict_GetItemString(sys_modules, modStr.c_str()); // borrowed
   if (mod) {
      Py_INCREF(mod); // convert borrowed → owned
      extState->set(x, mod);
      return mod;
   }
   auto sourceStr = source.str();

   // 2) Create new module
   mod = PyModule_New(modStr.c_str());
   if (!mod) return NULL;

   PyObject* globals = PyModule_GetDict(mod);

   // Ensure __builtins__
   if (!PyDict_GetItemString(globals, "__builtins__")) {
      PyObject* builtins = PyEval_GetBuiltins();
      if (PyDict_SetItemString(globals, "__builtins__", builtins) < 0) {
         Py_DECREF(mod);
         return NULL;
      }
   }

   // Execute Python source inside module
   PyObject* res = PyRun_String(
      sourceStr.c_str(), Py_file_input, globals, globals);
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

   extState->set(x, mod);
   return mod;
}
PyObject* PythonRuntime::getAttr(PyObject* obj, runtime::VarLen32 attr) {
   auto attrStr = attr.str();
   return PyObject_GetAttrString(obj, attrStr.c_str());
}
PyObject* PythonRuntime::call0(PyObject* callable) {
   auto res = PyObject_CallFunctionObjArgs(callable, NULL);
   if (!res) {
      throw_python_error();
   }
   return res;
}
PyObject* PythonRuntime::call1(PyObject* callable, PyObject* arg1) {
   std::array<PyObject*, 1> args = {arg1};
   return PyObject_Vectorcall(callable, args.data(), 1, NULL);
}
PyObject* PythonRuntime::call2(PyObject* callable, PyObject* arg1, PyObject* arg2) {
   std::array<PyObject*, 2> args = {arg1, arg2};
   return PyObject_Vectorcall(callable, args.data(), 2, NULL);
}
PyObject* PythonRuntime::call3(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3) {
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, NULL);
}
PyObject* PythonRuntime::call4(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4) {
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, NULL);
}
PyObject* PythonRuntime::call5(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5) {
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, NULL);
}
PyObject* PythonRuntime::call6(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6) {
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, arg6, NULL);
}
PyObject* PythonRuntime::call7(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7) {
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, NULL);
}
PyObject* PythonRuntime::call8(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7, PyObject* arg8) {
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, NULL);
}
PyObject* PythonRuntime::call9(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7, PyObject* arg8, PyObject* arg9) {
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, NULL);
}
PyObject* PythonRuntime::call10(PyObject* callable, PyObject* arg1, PyObject* arg2, PyObject* arg3, PyObject* arg4, PyObject* arg5, PyObject* arg6, PyObject* arg7, PyObject* arg8, PyObject* arg9, PyObject* arg10) {
   return PyObject_CallFunctionObjArgs(callable, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, NULL);
}

int64_t PythonRuntime::toInt64(PyObject* obj) {
   return PyLong_AsLongLong(obj);
}
PyObject* PythonRuntime::fromInt64(int64_t value) {
   return PyLong_FromLongLong(value);
}
bool PythonRuntime::toBool(PyObject* obj) {
   return PyObject_IsTrue(obj);
}
PyObject* PythonRuntime::fromBool(bool value) {
   return PyBool_FromLong(value ? 1 : 0);
}
runtime::VarLen32 PythonRuntime::toVarLen32(PyObject* obj) {
   if (PyUnicode_Check(obj)) {
      Py_ssize_t size;
      const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
      if (data) {
         // Copy into the execution context's string arena (caller may DecRef obj afterwards).
         return runtime::VarLen32::fromString(std::string(data, size));
      }
   }
   throw std::runtime_error("Object is not a string");
}
PyObject* PythonRuntime::fromVarLen32(runtime::VarLen32 value) {
   return PyUnicode_FromStringAndSize(value.data(), value.getLen());
}
double PythonRuntime::toDouble(PyObject* obj) {
   return PyFloat_AsDouble(obj);
}
PyObject* PythonRuntime::fromDouble(double value) {
   return PyFloat_FromDouble(value);
}

PyObjectPtr PythonRuntime::fromDate(int64_t value) {
   auto* dateTimeAPI = (PyDateTime_CAPI*) PyCapsule_Import(PyDateTime_CAPSULE_NAME, 0);
   auto year = DateRuntime::extractYear(value);
   auto month = DateRuntime::extractMonth(value);
   auto day = DateRuntime::extractDay(value);
   return dateTimeAPI->Date_FromDate((year), (month), (day), dateTimeAPI->DateType);
}

namespace {
// Bridge an arrow::Table across to pyarrow via the Arrow C Data Interface
// (https://arrow.apache.org/docs/format/CDataInterface.html). The host side
// only links against arrow_static (no `arrow_python` / pyarrow C++ helper).
// Pyarrow is reached purely through its public Python `_import_from_c` /
// `_export_to_c` methods on RecordBatchReader, with the struct address
// passed as a Python integer.

// Cache pa.RecordBatchReader._import_from_c per-thread; each sub-interpreter
// has its own import state, so resolve once per worker and reuse.
std::string formatCurrentPyError() {
   if (!PyErr_Occurred()) return "<no Python error set>";
   PyObject *ptype = nullptr, *pvalue = nullptr, *ptraceback = nullptr;
   PyErr_Fetch(&ptype, &pvalue, &ptraceback);
   PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
   std::string msg = "<unknown>";
   if (pvalue) {
      if (PyObject* str = PyObject_Str(pvalue)) {
         if (const char* utf8 = PyUnicode_AsUTF8(str)) msg = utf8;
         Py_DECREF(str);
      }
   }
   Py_XDECREF(ptype);
   Py_XDECREF(pvalue);
   Py_XDECREF(ptraceback);
   return msg;
}

PyObject* getReaderImportFromC() {
   thread_local PyObject* cached = nullptr;
   if (cached) return cached;
   PyObject* mod = PyImport_ImportModule("pyarrow");
   if (!mod) {
      throw std::runtime_error(
         "Tabular Python UDFs require the 'pyarrow' package to be importable "
         "inside the embedded Python interpreter. Underlying error: " + formatCurrentPyError());
   }
   PyObject* cls = PyObject_GetAttrString(mod, "RecordBatchReader");
   Py_DECREF(mod);
   if (!cls) throw std::runtime_error("pyarrow.RecordBatchReader not found");
   PyObject* method = PyObject_GetAttrString(cls, "_import_from_c");
   Py_DECREF(cls);
   if (!method) throw std::runtime_error("pyarrow.RecordBatchReader._import_from_c not found");
   cached = method; // owned, leaked at thread teardown — cheap and matches sub-interpreter lifetime
   return cached;
}

[[noreturn]] void throw_arrow_error(const arrow::Status& s, const char* what) {
   throw std::runtime_error(std::string(what) + ": " + s.ToString());
}
} // namespace

PyObjectPtr PythonRuntime::fromArrowTable(ArrowTable* table) {
   // Wrap the arrow::Table in a TableBatchReader; ExportRecordBatchReader keeps
   // it alive through the stream's release callback.
   auto reader = std::make_shared<arrow::TableBatchReader>(*table->get());

   ArrowArrayStream stream{};
   auto status = arrow::ExportRecordBatchReader(reader, &stream);
   if (!status.ok()) throw_arrow_error(status, "ExportRecordBatchReader");

   PyObject* importFromC = getReaderImportFromC();
   PyObject* addr = PyLong_FromUnsignedLongLong(reinterpret_cast<uintptr_t>(&stream));
   PyObject* readerObj = PyObject_CallFunctionObjArgs(importFromC, addr, nullptr);
   Py_DECREF(addr);
   if (!readerObj) {
      // _import_from_c failed before consuming the stream — release it ourselves.
      if (stream.release) stream.release(&stream);
      throw_python_error();
   }
   // On success, _import_from_c moved stream's contents into pyarrow's reader
   // (zeroing the source); nothing to release on the host side.

   PyObject* table_obj = PyObject_CallMethod(readerObj, "read_all", nullptr);
   Py_DECREF(readerObj);
   if (!table_obj) throw_python_error();
   return table_obj;
}
ArrowTable* PythonRuntime::toArrowTable(PyObject* obj) {
   // Convert Python pa.Table → RecordBatchReader → ArrowArrayStream, then
   // import on the host side without touching pyarrow C++ helpers.
   PyObject* reader = PyObject_CallMethod(obj, "to_reader", nullptr);
   if (!reader) {
      throw std::runtime_error("Tabular Python UDF must return a pyarrow.Table; got an object of a different type");
   }
   ArrowArrayStream stream{};
   PyObject* addr = PyLong_FromUnsignedLongLong(reinterpret_cast<uintptr_t>(&stream));
   PyObject* res = PyObject_CallMethod(reader, "_export_to_c", "O", addr);
   Py_DECREF(addr);
   Py_DECREF(reader);
   if (!res) throw_python_error();
   Py_DECREF(res);

   auto reader_result = arrow::ImportRecordBatchReader(&stream);
   if (!reader_result.ok()) throw_arrow_error(reader_result.status(), "ImportRecordBatchReader");
   auto table_result = reader_result.ValueOrDie()->ToTable();
   if (!table_result.ok()) throw_arrow_error(table_result.status(), "RecordBatchReader::ToTable");
   return new ArrowTable(table_result.ValueOrDie());
}

void PythonRuntime::decref(PyObject* obj) {
   Py_DECREF(obj);
}
void PythonRuntime::incref(PyObject* obj) {
   Py_INCREF(obj);
}
// CPython embed mode does not use the WASM session; provide a no-op so the
// symbol resolves and the runtime function table can still hold a pointer.
void PythonRuntime::setWasmSession(wasm::WASMSession*) {}

#elif defined(USE_CPYTHON_WASM_RUNTIME)

namespace {
thread_local wasm::WASMSession* currentWasmSession = nullptr;
[[noreturn]] inline void throw_python_error_wasm() {
   throw std::runtime_error("python error inside WASM sandbox");
}
} // namespace
void PythonRuntime::setWasmSession(wasm::WASMSession* session) {
   currentWasmSession = session;
}
PythonExtState* PythonRuntime::createPythonExtState() { return nullptr; }
PyObjectPtr PythonRuntime::import(size_t /*x*/, runtime::VarLen32 val) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   auto tmpScope = wasmSession.createTmpScope();
   auto wasmStr = tmpScope.allocateString(val.strView());
   return wasmSession.callPyFunc<PyObjectPtr>(wasm::PyImport_ImportModule, wasmStr.getAddr()).at(0).of.i32;
}
PyObjectPtr PythonRuntime::createModule(size_t /*x*/, runtime::VarLen32 modname, runtime::VarLen32 source) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   auto tmpScope = wasmSession.createTmpScope();

   auto modWasmStr = tmpScope.allocateString(modname.strView());
   PyObjectPtr sys_modules = wasmSession.callPyFunc<PyObjectPtr>(wasm::PyImport_GetModuleDict).at(0).of.i32;
   PyObjectPtr mod = wasmSession.callPyFunc<PyObjectPtr>(wasm::PyDict_GetItemString, sys_modules, modWasmStr.getAddr()).at(0).of.i32;
   if (mod) {
      wasmSession.callPyFunc<void>(wasm::Py_IncRef, mod);
      return mod;
   }
   auto sourceWasmStr = tmpScope.allocateString(source.strView());

   mod = wasmSession.callPyFunc<PyObjectPtr>(wasm::PyModule_New, modWasmStr.getAddr()).at(0).of.i32;
   if (!mod) return 0;

   PyObjectPtr globals = wasmSession.callPyFunc<PyObjectPtr>(wasm::PyModule_GetDict, mod).at(0).of.i32;
   auto builtinsWasmStr = tmpScope.allocateString("__builtins__");
   if (!wasmSession.callPyFunc<int>(wasm::PyDict_GetItemString, globals, builtinsWasmStr.getAddr()).at(0).of.i32) {
      PyObjectPtr builtins = wasmSession.callPyFunc<PyObjectPtr>(wasm::PyEval_GetBuiltins).at(0).of.i32;
      if (wasmSession.callPyFunc<int>(wasm::PyDict_SetItemString, globals, builtinsWasmStr.getAddr(), builtins).at(0).of.i32 < 0)
         return 0;
   }
   // Py_file_input == 257 in Python.h
   auto res = wasmSession.callPyFunc<PyObjectPtr>(wasm::PyRun_String, sourceWasmStr.getAddr(), 257, globals, globals).at(0).of.i32;
   if (!res) throw_python_error_wasm();
   if (wasmSession.callPyFunc<int>(wasm::PyDict_SetItemString, sys_modules, modWasmStr.getAddr(), mod).at(0).of.i32 < 0) {
      wasmSession.callPyFunc<void>(wasm::Py_DECREF, mod);
      return 0;
   }
   return mod;
}
PyObjectPtr PythonRuntime::getAttr(PyObjectPtr obj, runtime::VarLen32 attr) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   auto tmpScope = wasmSession.createTmpScope();
   auto wasmAttrStr = tmpScope.allocateString(attr.strView());
   return wasmSession.callPyFunc<PyObjectPtr>(wasm::PyObject_GetAttrString, obj, wasmAttrStr.getAddr()).at(0).of.i32;
}

template <typename... Args>
static uint32_t callPythonWASMUDF(PyObjectPtr callable, Args&&... args) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   auto tmpScope = wasmSession.createTmpScope();
   size_t numArgs = sizeof...(Args);
   auto raw = tmpScope.allocateRaw(numArgs * sizeof(uint32_t));
   uint32_t* argsBuf = reinterpret_cast<uint32_t*>(raw.getNativeAddr());
   size_t idx = 0;
   ((argsBuf[idx++] = args), ...);
   return wasmSession.callPyFunc<PyObjectPtr>(wasm::PyObject_Vectorcall, callable, raw.getAddr(), numArgs, 0).at(0).of.i32;
}
PyObjectPtr PythonRuntime::call0(PyObjectPtr c) { return callPythonWASMUDF(c); }
PyObjectPtr PythonRuntime::call1(PyObjectPtr c, PyObjectPtr a1) { return callPythonWASMUDF(c, a1); }
PyObjectPtr PythonRuntime::call2(PyObjectPtr c, PyObjectPtr a1, PyObjectPtr a2) { return callPythonWASMUDF(c, a1, a2); }
PyObjectPtr PythonRuntime::call3(PyObjectPtr c, PyObjectPtr a1, PyObjectPtr a2, PyObjectPtr a3) { return callPythonWASMUDF(c, a1, a2, a3); }
PyObjectPtr PythonRuntime::call4(PyObjectPtr c, PyObjectPtr a1, PyObjectPtr a2, PyObjectPtr a3, PyObjectPtr a4) { return callPythonWASMUDF(c, a1, a2, a3, a4); }
PyObjectPtr PythonRuntime::call5(PyObjectPtr c, PyObjectPtr a1, PyObjectPtr a2, PyObjectPtr a3, PyObjectPtr a4, PyObjectPtr a5) { return callPythonWASMUDF(c, a1, a2, a3, a4, a5); }
PyObjectPtr PythonRuntime::call6(PyObjectPtr c, PyObjectPtr a1, PyObjectPtr a2, PyObjectPtr a3, PyObjectPtr a4, PyObjectPtr a5, PyObjectPtr a6) { return callPythonWASMUDF(c, a1, a2, a3, a4, a5, a6); }
PyObjectPtr PythonRuntime::call7(PyObjectPtr c, PyObjectPtr a1, PyObjectPtr a2, PyObjectPtr a3, PyObjectPtr a4, PyObjectPtr a5, PyObjectPtr a6, PyObjectPtr a7) { return callPythonWASMUDF(c, a1, a2, a3, a4, a5, a6, a7); }
PyObjectPtr PythonRuntime::call8(PyObjectPtr c, PyObjectPtr a1, PyObjectPtr a2, PyObjectPtr a3, PyObjectPtr a4, PyObjectPtr a5, PyObjectPtr a6, PyObjectPtr a7, PyObjectPtr a8) { return callPythonWASMUDF(c, a1, a2, a3, a4, a5, a6, a7, a8); }
PyObjectPtr PythonRuntime::call9(PyObjectPtr c, PyObjectPtr a1, PyObjectPtr a2, PyObjectPtr a3, PyObjectPtr a4, PyObjectPtr a5, PyObjectPtr a6, PyObjectPtr a7, PyObjectPtr a8, PyObjectPtr a9) { return callPythonWASMUDF(c, a1, a2, a3, a4, a5, a6, a7, a8, a9); }
PyObjectPtr PythonRuntime::call10(PyObjectPtr c, PyObjectPtr a1, PyObjectPtr a2, PyObjectPtr a3, PyObjectPtr a4, PyObjectPtr a5, PyObjectPtr a6, PyObjectPtr a7, PyObjectPtr a8, PyObjectPtr a9, PyObjectPtr a10) { return callPythonWASMUDF(c, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); }

int64_t PythonRuntime::toInt64(PyObjectPtr obj) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   return wasmSession.callPyFunc<uint64_t>(wasm::PyLong_AsLongLong, obj).at(0).of.i64;
}
PyObjectPtr PythonRuntime::fromInt64(int64_t v) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   return wasmSession.callPyFunc<PyObjectPtr>(wasm::PyLong_FromLongLong, v).at(0).of.i32;
}
bool PythonRuntime::toBool(PyObjectPtr obj) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   return wasmSession.callPyFunc<int>(wasm::PyObject_IsTrue, obj).at(0).of.i32;
}
PyObjectPtr PythonRuntime::fromBool(bool v) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   return wasmSession.callPyFunc<PyObjectPtr>(wasm::PyBool_FromLong, v ? 1 : 0).at(0).of.i32;
}
runtime::VarLen32 PythonRuntime::toVarLen32(PyObjectPtr obj) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   auto wasmStrObj = wasmSession.callPyFunc<PyObjectPtr>(wasm::PyUnicode_AsUTF8, obj).at(0).of.i32;
   if (!wasmStrObj) throw std::runtime_error("Object is not a string");
   auto size = wasmSession.callPyFunc<int32_t>(wasm::PyUnicode_GetLength, obj).at(0).of.i32;
   const char* data = static_cast<const char*>(wasm_runtime_addr_app_to_native(wasmSession.moduleInst, wasmStrObj));
   if (!data) throw std::runtime_error("Failed to convert WASM string to host string");
   // Copy into the execution context's string arena (the WASM-side buffer is
   // unsafe to keep once the call returns).
   return runtime::VarLen32::fromString(std::string(data, size));
}
PyObjectPtr PythonRuntime::fromVarLen32(runtime::VarLen32 value) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   auto tmpScope = wasmSession.createTmpScope();
   auto wasmStr = tmpScope.allocateString(value.strView());
   return wasmSession.callPyFunc<PyObjectPtr>(wasm::PyUnicode_FromStringAndSize, wasmStr.getAddr(), value.getLen()).at(0).of.i32;
}
double PythonRuntime::toDouble(PyObjectPtr obj) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   return wasmSession.callPyFunc<double>(wasm::PyFloat_AsDouble, obj).at(0).of.f64;
}
PyObjectPtr PythonRuntime::fromDouble(double v) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   return wasmSession.callPyFunc<PyObjectPtr>(wasm::PyFloat_FromDouble, v).at(0).of.i32;
}
PyObjectPtr PythonRuntime::fromDate(int64_t value) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   auto tmpScope = wasmSession.createTmpScope();
   auto dateTimeStr = tmpScope.allocateString("datetime");
   auto dateStr = tmpScope.allocateString("date");
   auto datetimeModule = wasmSession.callPyFunc<PyObjectPtr>(wasm::PyImport_ImportModule, dateTimeStr.getAddr()).at(0).of.i32;
   auto dateClass = wasmSession.callPyFunc<PyObjectPtr>(wasm::PyObject_GetAttrString, datetimeModule, dateStr.getAddr()).at(0).of.i32;
   uint32_t year = DateRuntime::extractYear(value);
   uint32_t month = DateRuntime::extractMonth(value);
   uint32_t day = DateRuntime::extractDay(value);
   auto pyYear = fromInt64(year);
   auto pyMonth = fromInt64(month);
   auto pyDay = fromInt64(day);
   auto res = call3(dateClass, pyYear, pyMonth, pyDay);
   wasmSession.callPyFunc<void>(wasm::Py_DECREF, pyYear);
   wasmSession.callPyFunc<void>(wasm::Py_DECREF, pyMonth);
   wasmSession.callPyFunc<void>(wasm::Py_DECREF, pyDay);
   wasmSession.callPyFunc<void>(wasm::Py_DECREF, datetimeModule);
   wasmSession.callPyFunc<void>(wasm::Py_DECREF, dateClass);
   return res;
}
void PythonRuntime::decref(PyObjectPtr obj) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   wasmSession.callPyFunc<void>(wasm::Py_DECREF, obj);
}
void PythonRuntime::incref(PyObjectPtr obj) {
   wasm::WASMSession& wasmSession = *currentWasmSession;
   wasmSession.callPyFunc<void>(wasm::Py_IncRef, obj);
}
PyObjectPtr PythonRuntime::fromArrowTable(ArrowTable*) {
   throw std::runtime_error(
      "Tabular Python UDFs are not supported in the WASM Python backend. "
      "Build with -DENABLE_PYTHON=CPYTHON to enable them.");
}
ArrowTable* PythonRuntime::toArrowTable(PyObjectPtr) {
   throw std::runtime_error(
      "Tabular Python UDFs are not supported in the WASM Python backend. "
      "Build with -DENABLE_PYTHON=CPYTHON to enable them.");
}

#else

// Stubs when ENABLE_PYTHON=OFF — calling these means the module was compiled without Python support.
// Note for the lingodb wheel: the cp312 wheel intentionally does NOT include Python UDF support
// because cross-thread Py_EndInterpreter only became safe with CPython 3.13's Py_FinalizeEx
// auto-reaping of leftover sub-interpreters. Use the cp313 wheel for Python UDFs.
namespace {
[[noreturn]] void noPython() {
   throw std::runtime_error(
      "LingoDB Python UDFs are not available in this build. "
      "If you are using the lingodb Python wheel, install the cp313 wheel "
      "(Python 3.13 or newer) — sub-interpreter teardown requires the "
      "Py_FinalizeEx auto-reap added in CPython 3.13. "
      "If you are building lingodb yourself, configure with -DENABLE_PYTHON=CPYTHON.");
}
} // namespace
PyObjectPtr PythonRuntime::createModule(size_t, runtime::VarLen32, runtime::VarLen32) { noPython(); }
PyObjectPtr PythonRuntime::getAttr(PyObjectPtr, runtime::VarLen32) { noPython(); }
PyObjectPtr PythonRuntime::call0(PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::call1(PyObjectPtr, PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::call2(PyObjectPtr, PyObjectPtr, PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::call3(PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::call4(PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::call5(PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::call6(PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::call7(PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::call8(PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::call9(PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::call10(PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr, PyObjectPtr) { noPython(); }
int64_t PythonRuntime::toInt64(PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::fromInt64(int64_t) { noPython(); }
bool PythonRuntime::toBool(PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::fromBool(bool) { noPython(); }
PyObjectPtr PythonRuntime::fromDate(int64_t) { noPython(); }
runtime::VarLen32 PythonRuntime::toVarLen32(PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::fromVarLen32(runtime::VarLen32) { noPython(); }
double PythonRuntime::toDouble(PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::fromDouble(double) { noPython(); }
void PythonRuntime::decref(PyObjectPtr) {}
void PythonRuntime::incref(PyObjectPtr) {}
PyObjectPtr PythonRuntime::fromArrowTable(ArrowTable*) { noPython(); }
ArrowTable* PythonRuntime::toArrowTable(PyObjectPtr) { noPython(); }
PyObjectPtr PythonRuntime::import(size_t, runtime::VarLen32) { noPython(); }
PythonExtState* PythonRuntime::createPythonExtState() { noPython(); }
void PythonRuntime::setWasmSession(wasm::WASMSession*) {}

#endif
} // namespace lingodb::runtime
