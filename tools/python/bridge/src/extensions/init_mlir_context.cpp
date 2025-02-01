#include <nanobind/nanobind.h>

#include <iostream>
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace bridge {
__attribute__((visibility("default"))) void initContext(MlirContext context);
}

NB_MODULE(mlir_init, m) {
   m.def("init_context", &bridge::initContext);
}