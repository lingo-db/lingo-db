from setuptools import setup,Extension
import pyarrow as pa
import pybind11
ext= Extension("lingodbbridge.ext", ["pylingodb.cpp"])
ext.include_dirs.append(pa.get_include())
ext.libraries.extend(pa.get_libraries())
ext.library_dirs.extend(pa.get_library_dirs())
ext.include_dirs.append(pybind11.get_include())

setup(name='lingodb-bridge',
      description='',
      version='0.0.0.dev0',
      packages=['lingodbbridge'],
      package_data={'lingodbbridge':['libs/*.so']},
      ext_modules=[ext],
      install_requires=[
            'pyarrow ==12.0.1',
      ],

)