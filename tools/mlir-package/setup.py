import fnmatch
import os

from setuptools import setup

clang_libs = [f"llvm/lib/*{n}*" for n in ["clangAST",
                                          "clangASTMatchers",
                                          "clangBasic",
                                          "clangFrontend",
                                          "clangSerialization",
                                          "clangTooling", "clangParse", "clangSerialization", "clangSema",
                                          "clangAnalysis", "clangSupport", "clangDriver", "clangFormat",
                                          "clangToolingInclusions", "clangToolingCore", "clangRewrite", "clangLex",
                                          "clangEdit"]]

bin_files = ["FileCheck", "mlir-tblgen", "clang-tidy"]
for filename in os.listdir("lingodbllvm/llvm/bin"):
    # Check if the current file is a regular file (not a directory)
    full_path = os.path.join("lingodbllvm/llvm/bin", filename)
    if os.path.isfile(full_path):
        if filename not in bin_files:
            with open("lingodbllvm/llvm/bin/" + filename, "w") as f:
                f.write("#!/usr/bin/env python\n")
                f.write("print('This is a dummy file')\n")

required_lib_files = ["llvm/lib/libMLIR*", "llvm/lib/libLLVM*"] + clang_libs
for filename in os.listdir("lingodbllvm/llvm/lib"):
    full_path = os.path.join("lingodbllvm/llvm/lib", filename)
    if os.path.isfile(full_path):
        if not any([fnmatch.fnmatch("llvm/lib/" + filename, pattern) for pattern in required_lib_files]):
            with open("lingodbllvm/llvm/lib/" + filename, "w") as f:
                f.write("This is a dummy file")

package_data = ['llvm/include/**/*',
                'llvm/lib/cmake/**/*', 'llvm/bin/*', 'llvm/lib/**/*', 'llvm/mlir_python_bindings/**/*',
                'llvm/python_packages/**/*']
setup(name='lingodb-llvm',
      description='',
      version='0.0.0.dev0',
      packages=['lingodbllvm'],
      package_data={'lingodbllvm': package_data},
      install_requires=[],

      )
