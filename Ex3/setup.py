from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext_modules=[
        Extension("ex3", ["ex3.pyx"],
                include_dirs=[np.get_include()],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp']
                  )
]
#python setup.py build_ext --inplace
setup(
    ext_modules = cythonize(ext_modules)
)