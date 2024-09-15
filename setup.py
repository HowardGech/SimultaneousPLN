from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "admm_cy",
        sources=["admm_cy.pyx", "admm.c"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "simultaneous_pln",
        sources=["simultaneous_pln.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="SimultaneousPLN",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
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