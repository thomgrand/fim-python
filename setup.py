from setuptools import setup
from distutils.core import setup, Extension
import sys
import os

if sys.platform.startswith("win32"):
     extra_compile_args = ["/O2", "/openmp"]
     extra_link_args = [] #["/openmp"]
else:
     extra_compile_args = ["-O3", "-fopenmp"]
     extra_link_args = ["-fopenmp"]

fim_cutils_extension = Extension(
                              name="fimpy.fim_cutils.fim_cutils",
                              sources=["fimpy/fim_cutils/fim_cutils.pyx"], # our Cython source
                              #sources=["Rectangle.cpp"],  # additional source file(s)
                              language="c++",             # generate C++ code
                              extra_compile_args=extra_compile_args,
                              extra_link_args=extra_link_args,
                              compiler_directives={'language_level' : "3"}
                         )

comp_cutils_extension = Extension(
                              name="fimpy.utils.cython.comp",
                              sources=["fimpy/utils/cython/comp.pyx"], # our Cython source
                              #sources=["Rectangle.cpp"],  # additional source file(s)
                              language="c++",             # generate C++ code
                              extra_compile_args=extra_compile_args,
                              extra_link_args=extra_link_args,
                              compiler_directives={'language_level' : "3"}
                         )

lib_requires_cpu = ["numpy", "Cython>=0.29.22"]
lib_requires_gpu = ["cupy>=9.0"]
test_requires_cpu = ["scipy", "pytest", "pytest-cov", "matplotlib", "pandas", "ipython"]    

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as readme:
     long_description = readme.read()

setup(name="fim-python",
    version="1.1",    
    description="This repository implements the Fast Iterative Method on tetrahedral domains and triangulated surfaces purely in python both for CPU (numpy) and GPU (cupy).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thomgrand/fim-python",
    packages=["fimpy", "fimpy.utils", "fimpy.fim_cutils", "fimpy.utils.cython"],
    install_requires=lib_requires_cpu,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Environment :: GPU :: NVIDIA CUDA",
        "License :: OSI Approved :: GNU Affero General Public License v3",
    ],
    python_requires='>=3.6',
     author="Thomas Grandits",
     author_email="tomdev@gmx.net",
     license="AGPL",     
     ext_modules = [fim_cutils_extension, comp_cutils_extension],
     extras_require = {
          'gpu': lib_requires_gpu,
          'tests': test_requires_cpu,
          'docs': ["sphinx", "pydata_sphinx_theme", "pandas", "ipython"]
     }
     )

