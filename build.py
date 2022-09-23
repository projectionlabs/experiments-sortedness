from setuptools import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "experimentssortedness.wtau.wtau",
        ["src/experimentssortedness/wtau/wtau.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=["m"],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=['-fopenmp']
    )
]

compiler_directives = {"language_level": 3, "embedsignature": True}


def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": cythonize(
                extensions, compiler_directives=compiler_directives
            ),
        }
    )
