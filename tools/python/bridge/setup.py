import pip

pip.main(['install', '-f', 'https://www.lingo-db.com/dev-dependencies/', 'lingodb-llvm'])
import lingodbllvm
import pyarrow as pa
import pybind11
from setuptools import setup, Extension

llvm_include_dir = lingodbllvm.get_bin_dir() + "/../include"
llvm_lib_dir = lingodbllvm.get_bin_dir() + "/../lib"
bindings_dir = lingodbllvm.get_py_bindings_dir()
ext = Extension("lingodbbridge.ext", ["pylingodb.cpp"])
ext.include_dirs.append(pa.get_include())
ext.libraries.extend(pa.get_libraries())
ext.library_dirs.extend(pa.get_library_dirs())
ext.include_dirs.append(pybind11.get_include())
ext.include_dirs.append(llvm_include_dir)
ext.define_macros.append(('MLIR_PYTHON_PACKAGE_PREFIX', 'lingodbbridge.mlir.'))

mlirir_ext = Extension("lingodbbridge.mlir._mlir_libs._mlir", [f"{bindings_dir}/{f}" for f in [
    "MainModule.cpp",
    "IRAffine.cpp",
    "IRAttributes.cpp",
    "IRCore.cpp",
    "IRInterfaces.cpp",
    "IRModule.cpp",
    "IRTypes.cpp",
    "Pass.cpp"]
                                                               ])
mlirir_ext.include_dirs.append(llvm_include_dir)
mlirir_ext.include_dirs.append(bindings_dir)
mlirir_ext.include_dirs.append(pybind11.get_include())
mlirir_ext.define_macros.append(('MLIR_PYTHON_PACKAGE_PREFIX', 'lingodbbridge.mlir.'))
mlirir_ext.library_dirs.append(llvm_lib_dir)
mlirir_ext.libraries.append("LLVMSupport")
mlirir_ext.libraries.append("LLVMDemangle")
mlirinit_ext = Extension("lingodbbridge.mlir._mlir_libs.mlir_init", ["init_mlir_context.cpp"]);
mlirinit_ext.include_dirs.append(llvm_include_dir)
mlirinit_ext.include_dirs.append(pybind11.get_include())
mlirinit_ext.define_macros.append(('MLIR_PYTHON_PACKAGE_PREFIX', 'lingodbbridge.mlir.'))
mlirinit_ext.library_dirs.append(llvm_lib_dir)
mlirinit_ext.libraries.append("LLVMSupport")
mlirinit_ext.libraries.append("LLVMDemangle")

mlir_lingodb_ext = Extension("lingodbbridge.mlir._mlir_libs.mlir_lingodb", ["custom_dialects_py.cpp"]);
mlir_lingodb_ext.include_dirs.append(llvm_include_dir)
mlir_lingodb_ext.include_dirs.append(pybind11.get_include())
mlir_lingodb_ext.define_macros.append(('MLIR_PYTHON_PACKAGE_PREFIX', 'lingodbbridge.mlir.'))
mlir_lingodb_ext.library_dirs.append(llvm_lib_dir)
mlir_lingodb_ext.libraries.append("LLVMSupport")
mlir_lingodb_ext.libraries.append("LLVMDemangle")

setup(name='lingodb-bridge',
      description='',
      version='0.0.1',
      packages=['lingodbbridge', 'lingodbbridge.mlir', 'lingodbbridge.mlir.dialects'],
      package_data={'lingodbbridge': ['libs/*.so']},
      ext_modules=[ext, mlirir_ext, mlirinit_ext, mlir_lingodb_ext]
      )
