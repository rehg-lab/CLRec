#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
	cmdclass = {'build_ext': build_ext},
	ext_modules = [Extension("jitter",
							 sources=["jitter.pyx", "cjitter.cpp"],
							 include_dirs=[numpy.get_include()],
							 language="c++",
							 extra_compile_args=["-std=c++11"],
							 extra_link_args=["-std=c++11"])],
)


