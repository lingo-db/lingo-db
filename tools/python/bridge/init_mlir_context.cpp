#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include <iostream>
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace bridge {
__attribute__((visibility("default"))) void initContext(MlirContext context);
}

PYBIND11_MODULE(mlir_init, m) {
   m.def("init_context", &bridge::initContext);
}