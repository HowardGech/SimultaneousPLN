from setuptools import setup, Extension
import platform
from Cython.Build import cythonize
import numpy

if platform.system() == "Darwin":
    extra_compile_args = ["-I/System/Library/Frameworks/vecLib.framework/Headers",'-fopenmp']
    if "ppc" in platform.machine():
        extra_compile_args.append("-faltivec")

    extra_link_args = ["-Wl,-framework", "-Wl,Accelerate",'-fopenmp']
    include_dirs = [numpy.get_include()]

else:
    include_dirs = [numpy.get_include(), "/usr/local/include"]
    extra_compile_args = ["-O2", "-fPIC", "-w",'-fopenmp']
    extra_link_args = ["-llapack",'-fopenmp']

if platform.machine() in ("x86_64", "AMD64"):
    extra_compile_args.append("-msse2")

ext_modules = [
    Extension(
        "SimultaneousPLN.admm_cy",
        sources=["SimultaneousPLN/admm_cy.pyx", "SimultaneousPLN/admm.c"],
        include_dirs=include_dirs,
        extra_compile_args = extra_compile_args,
        extra_link_args=extra_link_args,
    ),

    Extension(
    name="SimultaneousPLN.pyquic",
    sources=["SimultaneousPLN/QUIC.C", "SimultaneousPLN/pyquic.pyx"],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
),
    Extension(
        "SimultaneousPLN._SPLN",
        sources=["SimultaneousPLN/simultaneous_pln.pyx"],
        include_dirs=include_dirs,
    ),
]

setup(
    name="SimultaneousPLN",
    ext_modules=cythonize(ext_modules, language_level=3),
    include_dirs=include_dirs,
    packages=["SimultaneousPLN"],
    version="0.1.0",
    description="Simultaneous Poisson Log-Normal regression with Cython and QUIC solver",
    author="Changhao Ge"
)
