import os


def get_mlir_dir():
    return os.path.dirname(os.path.abspath(__file__)) + "/llvm/lib/cmake/mlir"


def get_bin_dir():
    return os.path.dirname(os.path.abspath(__file__)) + "/llvm/bin"


def get_py_package_dir():
    return os.path.dirname(os.path.abspath(__file__)) + "/llvm/python_packages"


def get_py_bindings_dir():
    return os.path.dirname(os.path.abspath(__file__)) + "/llvm/mlir_python_bindings"
