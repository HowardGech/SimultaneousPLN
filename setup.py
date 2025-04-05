from setuptools import setup, Extension
import platform
from Cython.Build import cythonize
import numpy
# export CC=/opt/homebrew/opt/llvm/bin/clang
# export CXX=/opt/homebrew/opt/llvm/bin/clang++

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
        "admm_cy",
        sources=["admm_cy.pyx", "admm.c"],
        include_dirs=include_dirs,
        extra_compile_args = extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    #     Extension(
    #     "pyquic",
    #     sources=["pyquic.pyx","QUIC.C"],
    #     include_dirs=[numpy.get_include()],
    #     extra_compile_args = ["-fopenmp" ],
    #     extra_link_args=['-fopenmp'],
    #     language="c++",
    # ),
    Extension(
    name="pyquic",
    sources=["QUIC.C", "pyquic.pyx"],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
),
    Extension(
        "simultaneous_pln",
        sources=["simultaneous_pln.pyx"],
        include_dirs=include_dirs,
    ),
]

setup(
    name="SimultaneousPLN",
    ext_modules=cythonize(ext_modules, language_level=3),
    include_dirs=include_dirs,
)


# from setuptools import setup, Extension
# from Cython.Build import cythonize
# import numpy

# extensions = [
#     Extension(
#         "simultaneous_pln",
#         sources=["simultaneous_pln.pyx", "admm_cy.pyx", "admm.c"],
#         extra_compile_args=['-fopenmp'],
#         extra_link_args=['-fopenmp'],
#         include_dirs=[numpy.get_include()]
#     )
# ]

# setup(
#     name="SimultaneousPLN",
#     ext_modules=cythonize(extensions),
#     include_dirs=[numpy.get_include()],
# )