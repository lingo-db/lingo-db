#include "lingodb/runtime/PythonRuntime.h"
#include "lingodb/runtime/DateRuntime.h"
#ifdef USE_CPYTHON_WASM_RUNTIME
#include "lingodb/runtime/WASM.h"
#endif
#ifdef USE_CPYTHON_RUNTIME
#include "datetime.h"
#endif
#include <iostream>
#include <stdexcept>
namespace lingodb::runtime{
#ifdef USE_CPYTHON_RUNTIME

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

   PyObject *sys_modules = PyImport_GetModuleDict();  // borrowed

   // 1) Check if module already exists
   PyObject *mod = PyDict_GetItemString(sys_modules, modStr.c_str());  // borrowed
   if (mod) {
      Py_INCREF(mod);      // convert borrowed → owned
      return mod;          // caller owns reference
   }
   auto sourceStr = source.str();


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
PyObject* PythonRuntime::getAttr2(PyObject* obj, PyObject* attr){
   return PyObject_GetAttr(obj, attr);
}
PyObject* PythonRuntime::call0(PyObject* callable){
   auto res= PyObject_CallFunctionObjArgs(callable, NULL);
   if (! res) {
      throw_python_error();
   }
   return res;
}
PyObject* PythonRuntime::call1(PyObject* callable, PyObject* arg1){
   auto res =PyObject_CallFunctionObjArgs(callable, arg1, NULL);
   if (! res) {
      throw_python_error();
   }
   return res;
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
   return PyUnicode_FromStringAndSize(value.data(), value.getLen());
}
double PythonRuntime::toDouble(PyObject* obj){
   return PyFloat_AsDouble(obj);
}
PyObject* PythonRuntime::fromDouble(double value){
   return PyFloat_FromDouble(value);
}

PyObjectPtr PythonRuntime::fromDate(int64_t value){
   auto *dateTimeAPI = (PyDateTime_CAPI *)PyCapsule_Import(PyDateTime_CAPSULE_NAME, 0);
   auto year = DateRuntime::extractYear(value);
   auto month = DateRuntime::extractMonth(value);
   auto day = DateRuntime::extractDay(value);
   return dateTimeAPI->Date_FromDate((year), (month), (day), dateTimeAPI->DateType);
}

void PythonRuntime::decref(PyObject* obj){
   Py_DECREF(obj);
}
void PythonRuntime::incref(PyObject* obj){
   Py_INCREF(obj);
}

#else

#ifdef USE_CPYTHON_WASM_RUNTIME
inline void throw_python_error(wasm::WASMSession session) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   //PyObjectPtr ptype = session.createWasmBuffer(sizeof(PyObject));
   //PyObjectPtr pvalue = session.createWasmBuffer(sizeof(PyObject));
   //PyObjectPtr ptraceback = session.createWasmBuffer(sizeof(PyObject));
   //session.callPyFunc<void>("PyErr_Fetch", ptype, pvalue, ptraceback);
   //throw std::runtime_error("msg");
   throw std::runtime_error("some python error");
}

PyObjectPtr PythonRuntime::createModule(runtime::VarLen32 modname, runtime::VarLen32 source) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   auto modStr = modname.str();
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   auto modWasmStr = wasmSession.createWasmStringBuffer(modStr);
   PyObjectPtr sys_modules = wasmSession.callPyFunc<PyObjectPtr>("PyImport_GetModuleDict").at(0).of.i32;
   assert(sys_modules);
   // 1) Check if module already exists
   PyObjectPtr mod = wasmSession.callPyFunc<PyObjectPtr>("PyDict_GetItemString", sys_modules, modWasmStr).at(0).of.i32;
   if (mod) {
      wasmSession.callPyFunc<void>("Py_IncRef", mod); // convert borrowed → owned
      wasmSession.freeWasmBuffer(modWasmStr);
      return mod;
   }
   auto sourceStr = source.str();
   auto sourceWasmStr = wasmSession.createWasmStringBuffer(sourceStr);

   // 2) Create new module
   mod = wasmSession.callPyFunc<PyObjectPtr>("PyModule_New", modWasmStr).at(0).of.i32;
   if (!mod) {
      std::cerr << "Error creating module" << std::endl;
      return 0;
   }


   PyObjectPtr globals = wasmSession.callPyFunc<PyObjectPtr>("PyModule_GetDict", mod).at(0).of.i32;
   assert(globals);
   // Ensure __builtins__
   auto builtinsWasmStr = wasmSession.createWasmStringBuffer("__builtins__");
   if (!wasmSession.callPyFunc<int>("PyDict_GetItemString", globals, builtinsWasmStr).at(0).of.i32) {
      PyObjectPtr builtins = wasmSession.callPyFunc<PyObjectPtr>("PyEval_GetBuiltins").at(0).of.i32;
      if (wasmSession.callPyFunc<int>("PyDict_SetItemString", globals, builtinsWasmStr, builtins).at(0).of.i32 < 0) {
         std::cerr << "Error " << std::endl;
         return 0;
      }
   }
   // Py_file_input = 257; //from Python.h
   auto res = wasmSession.callPyFunc<PyObjectPtr>("PyRun_String", sourceWasmStr, 257, globals, globals).at(0).of.i32;
   if (!res) {
      throw_python_error(wasmSession);
      assert(false);
   }

   // 3) Insert into sys.modules
   if (wasmSession.callPyFunc<int>("PyDict_SetItemString", sys_modules, modWasmStr, mod).at(0).of.i32 < 0) {
      wasmSession.callPyFunc<void>("Py_DECREF", mod);
      return 0;
   }
   wasmSession.freeWasmBuffer(modWasmStr);
   wasmSession.freeWasmBuffer(sourceWasmStr);
   wasmSession.freeWasmBuffer(builtinsWasmStr);

   return mod;
}
PyObjectPtr PythonRuntime::getAttr(PyObjectPtr obj, runtime::VarLen32 attr) {
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   auto wasmAttrStr = wasmSession.createWasmStringBuffer(attr.str());
   assert(wasmAttrStr && obj);
   uint32_t pyAttr = wasmSession.callPyFunc<PyObjectPtr>("PyObject_GetAttrString", obj, wasmAttrStr).at(0).of.i32;
   if (!pyAttr) {
      wasmSession.callPyFunc<void>("PyErr_Print");
   }
   wasmSession.freeWasmBuffer(wasmAttrStr);
   return pyAttr;
}
PyObjectPtr PythonRuntime::getAttr2(PyObjectPtr obj, PyObjectPtr attr) {
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   return wasmSession.callPyFunc<PyObjectPtr>("PyObject_GetAttr", obj, attr).at(0).of.i32;
}

void PythonRuntime::setAttr(PyObjectPtr obj, runtime::VarLen32 attr, PyObjectPtr value) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
        wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
        auto wasmAttrStr = wasmSession.createWasmStringBuffer(attr.str());
        assert(wasmAttrStr && obj);
        wasmSession.callPyFunc<void>("PyObject_SetAttrString", obj, wasmAttrStr, value);
        wasmSession.freeWasmBuffer(wasmAttrStr);
}

template <typename... Args>
static uint32_t callPythonWASMUDF(PyObjectPtr callable, Args&&... args) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   size_t numArgs = sizeof...(Args);
   void* nativeBufAddr = nullptr;

   uint32_t instBufAddr = wasm_runtime_module_malloc_internal(wasmSession.moduleInst, wasmSession.execEnv, numArgs*sizeof(uint32_t), &nativeBufAddr);
   if (!nativeBufAddr) {
      throw std::runtime_error(wasm_runtime_get_exception(wasmSession.moduleInst));
   }
   uint32_t* argsBuf = static_cast<uint32_t*>(nativeBufAddr);
        size_t idx = 0;
        ((argsBuf[idx++] = args), ...);
   auto result = wasmSession.callPyFunc<PyObjectPtr>("PyObject_Vectorcall", callable, instBufAddr, numArgs, 0).at(0).of.i32;
   assert(callable);
   wasmSession.freeWasmBuffer(instBufAddr);
   return result;
}

uint32_t PythonRuntime::call0(PyObjectPtr callable) {
   return callPythonWASMUDF(callable);
}

uint32_t PythonRuntime::call1(PyObjectPtr callable, PyObjectPtr arg) {
   return callPythonWASMUDF(callable, arg);
}

uint32_t PythonRuntime::call2(PyObjectPtr callable, PyObjectPtr arg, PyObjectPtr arg1) {
   return callPythonWASMUDF(callable, arg, arg1);
}

uint32_t PythonRuntime::call3(PyObjectPtr callable, PyObjectPtr arg, PyObjectPtr arg1, PyObjectPtr arg2) {
   return callPythonWASMUDF(callable, arg, arg1, arg2);
}

uint32_t PythonRuntime::call4(PyObjectPtr callable, PyObjectPtr arg, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3) {
   return callPythonWASMUDF(callable, arg, arg1, arg2, arg3);
}

uint32_t PythonRuntime::call5(PyObjectPtr callable, PyObjectPtr arg, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4) {
   return callPythonWASMUDF(callable, arg, arg1, arg2, arg3, arg4);
}

uint32_t PythonRuntime::call6(PyObjectPtr callable, PyObjectPtr arg, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5) {
   return callPythonWASMUDF(callable, arg, arg1, arg2, arg3, arg4, arg5);
}

uint32_t PythonRuntime::call7(PyObjectPtr callable, PyObjectPtr arg, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5, PyObjectPtr arg6) {
   return callPythonWASMUDF(callable, arg, arg1, arg2, arg3, arg4, arg5, arg6);
}

uint32_t PythonRuntime::call8(PyObjectPtr callable, PyObjectPtr arg, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5, PyObjectPtr arg6, PyObjectPtr arg7) {
   return callPythonWASMUDF(callable, arg, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
}

uint32_t PythonRuntime::call9(PyObjectPtr callable, PyObjectPtr arg, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5, PyObjectPtr arg6, PyObjectPtr arg7, PyObjectPtr arg8) {
   return callPythonWASMUDF(callable, arg, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
}

uint32_t PythonRuntime::call10(PyObjectPtr callable, PyObjectPtr arg, PyObjectPtr arg1, PyObjectPtr arg2, PyObjectPtr arg3, PyObjectPtr arg4, PyObjectPtr arg5, PyObjectPtr arg6, PyObjectPtr arg7, PyObjectPtr arg8, PyObjectPtr arg9) {
   return callPythonWASMUDF(callable, arg, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
}

int64_t PythonRuntime::toInt64(PyObjectPtr pyObj) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   return wasmSession.callPyFunc<uint64_t>("PyLong_AsLongLong", pyObj).at(0).of.i64;
}

PyObjectPtr PythonRuntime::fromInt64(int64_t obj) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   return wasmSession.callPyFunc<PyObjectPtr>("PyLong_FromLongLong", obj).at(0).of.i32;
}
bool PythonRuntime::toBool(PyObjectPtr obj) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   return wasmSession.callPyFunc<int>("PyObject_IsTrue", obj).at(0).of.i32; //todo
}
PyObjectPtr PythonRuntime::fromBool(bool value) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   return wasmSession.callPyFunc<PyObjectPtr>("PyBool_FromLong", value ? 1 : 0).at(0).of.i32;
}
runtime::VarLen32 PythonRuntime::toVarLen32(PyObjectPtr obj) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
    wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   auto wasmStrObj = wasmSession.callPyFunc<PyObjectPtr>("PyUnicode_AsUTF8", obj).at(0).of.i32;
   if (!wasmStrObj) {
      throw std::runtime_error("Object is not a string");
   }
   auto size = wasmSession.callPyFunc<int32_t>("PyUnicode_GetLength", obj).at(0).of.i32;
   // Copy string from WASM memory to host memory
   char* data = static_cast<char*>(wasm_runtime_addr_app_to_native(wasmSession.moduleInst, wasmStrObj));
   if (!data) {
      throw std::runtime_error("Failed to convert WASM string to host string");
   }
   return runtime::VarLen32::fromString(std::string_view(data, size), StorageClass::REFCOUNTED);
}
PyObjectPtr PythonRuntime::fromVarLen32(runtime::VarLen32 value) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
        auto str = value.str();
        auto wasmStr = wasmSession.createWasmStringBuffer(str);
        auto res= wasmSession.callPyFunc<PyObjectPtr>("PyUnicode_FromStringAndSize", wasmStr, str.size()).at(0).of.i32;
        wasmSession.freeWasmBuffer(wasmStr);
        return res;
}
double PythonRuntime::toDouble(PyObjectPtr obj) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   return wasmSession.callPyFunc<double>("PyFloat_AsDouble", obj).at(0).of.f64;
}
PyObjectPtr PythonRuntime::fromDouble(double value) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   return wasmSession.callPyFunc<PyObjectPtr>("PyFloat_FromDouble", value).at(0).of.i32;
}
PyObjectPtr PythonRuntime::fromDate(int64_t value){
   throw std::runtime_error("fromDate not implemented in WASM Python runtime");
}

void PythonRuntime::decref(PyObjectPtr obj) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   wasmSession.callPyFunc<void>("Py_DecRef", obj);
}
void PythonRuntime::incref(PyObjectPtr obj) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   wasmSession.callPyFunc<void>("Py_IncRef", obj);
}
PyObjectPtr PythonRuntime::import(runtime::VarLen32 val) {
#ifdef ASAN_ACTIVE
   while (!wasm_runtime_thread_env_inited()) {
      wasm_runtime_init_thread_env();
   }
#endif
   wasm::WASMSession wasmSession = getCurrentExecutionContext()->getWasmSession();
   auto wasmStr = wasmSession.createWasmStringBuffer(val.str());
   auto res= wasmSession.callPyFunc<PyObjectPtr>("PyImport_ImportModule", wasmStr).at(0).of.i32;
   wasmSession.freeWasmBuffer(wasmStr);
   return res;
}


#else // USE_CPYTHON_RUNTIME

PyObject* PythonRuntime::createModule(runtime::VarLen32 /*modname*/, runtime::VarLen32 /*source*/) {
    throw std::runtime_error("CPython runtime is not enabled");
}
PyObject* PythonRuntime::getAttr(PyObject* /*obj*/, runtime::VarLen32 /*attr*/) {
    throw std::runtime_error("CPython runtime is not enabled");
}
PyObjectPtr PythonRuntime::getAttr2(PyObjectPtr /*obj*/, PyObjectPtr /*attr*/) {
    throw std::runtime_error("CPython runtime is not enabled");
}
void PythonRuntime::setAttr(PyObject* /*obj*/, runtime::VarLen32 /*attr*/, PyObject* /*value*/) {
    throw std::runtime_error("CPython runtime is not enabled");
}
PyObject* PythonRuntime::call0(PyObject* /*callable*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::call1(PyObject* /*callable*/, PyObject* /*arg1*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::call2(PyObject* /*callable*/, PyObject* /*arg1*/, PyObject* /*arg2*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::call3(PyObject* /*callable*/, PyObject* /*arg1*/, PyObject* /*arg2*/, PyObject* /*arg3*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::call4(PyObject* /*callable*/, PyObject* /*arg1*/, PyObject* /*arg2*/, PyObject* /*arg3*/, PyObject* /*arg4*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::call5(PyObject* /*callable*/, PyObject* /*arg1*/, PyObject* /*arg2*/, PyObject* /*arg3*/, PyObject* /*arg4*/, PyObject* /*arg5*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::call6(PyObject* /*callable*/, PyObject* /*arg1*/, PyObject* /*arg2*/, PyObject* /*arg3*/, PyObject* /*arg4*/, PyObject* /*arg5*/, PyObject* /*arg6*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::call7(PyObject* /*callable*/, PyObject* /*arg1*/, PyObject* /*arg2*/, PyObject* /*arg3*/, PyObject* /*arg4*/, PyObject* /*arg5*/, PyObject* /*arg6*/, PyObject* /*arg7*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::call8(PyObject* /*callable*/, PyObject* /*arg1*/, PyObject* /*arg2*/, PyObject* /*arg3*/, PyObject* /*arg4*/, PyObject* /*arg5*/, PyObject* /*arg6*/, PyObject* /*arg7*/, PyObject* /*arg8*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::call9(PyObject* /*callable*/, PyObject* /*arg1*/, PyObject* /*arg2*/, PyObject* /*arg3*/, PyObject* /*arg4*/, PyObject* /*arg5*/, PyObject* /*arg6*/, PyObject* /*arg7*/, PyObject* /*arg8*/, PyObject* /*arg9*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::call10(PyObject* /*callable*/, PyObject* /*arg1*/, PyObject* /*arg2*/, PyObject* /*arg3*/, PyObject* /*arg4*/, PyObject* /*arg5*/, PyObject* /*arg6*/, PyObject* /*arg7*/, PyObject* /*arg8*/, PyObject* /*arg9*/, PyObject* /*arg10*/) { throw std::runtime_error("CPython runtime is not enabled"); }
int64_t PythonRuntime::toInt64(PyObject* /*obj*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::fromInt64(int64_t /*value*/) { throw std::runtime_error("CPython runtime is not enabled"); }
bool PythonRuntime::toBool(PyObject* /*obj*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::fromBool(bool /*value*/) { throw std::runtime_error("CPython runtime is not enabled"); }
runtime::VarLen32 PythonRuntime::toVarLen32(PyObject* /*obj*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::fromVarLen32(runtime::VarLen32 /*value*/) { throw std::runtime_error("CPython runtime is not enabled"); }
double PythonRuntime::toDouble(PyObject* /*obj*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::fromDouble(double /*value*/) { throw std::runtime_error("CPython runtime is not enabled");}
void PythonRuntime::decref(PyObject* /*obj*/) { throw std::runtime_error("CPython runtime is not enabled"); }
void PythonRuntime::incref(PyObject* /*obj*/) { throw std::runtime_error("CPython runtime is not enabled"); }
PyObject* PythonRuntime::import(runtime::VarLen32 /*val*/) { throw std::runtime_error("CPython runtime is not enabled"); }
#endif // USE_CPYTHON_WASM_RUNTIME
#endif // USE_CPYTHON_RUNTIME

}