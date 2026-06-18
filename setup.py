"""
Build script for the cyminiball package.

Updated: 18.06.2026

This file contains the extension (Cython/C++) build logic only.
Project metadata and configuration are defined in `pyproject.toml`.

Cython is optional and only required when generating the C++ sources.
Pre-built distributions (wheels, sdists with .cpp) do not require Cython.

A Cython build is triggered if one of the following applies:
  - `src/_miniball_wrap.cpp` is missing
  - environment variable `CYMINIBALL_CYTHONIZE` is set
  - the `build_ext` command is executed

Optional build configuration:
  - `CYMINIBALL_TRACE=1` enables line tracing in the Cython extension
    (useful for coverage analysis)

Build and installation (modern workflow):
  - Build distributions:
        python -m build
  - Install from wheel:
        pip install dist/cyminiball-*.whl
  - Install in editable mode (development):
        pip install -e .

Testing and validation:
  - Run tests:
        pytest
  - Check package metadata:
        twine check dist/*

Publishing:
  - Upload to TestPyPI:
        twine upload --repository testpypi dist/*
  - Install from TestPyPI:
        pip install --index-url https://test.pypi.org/simple/ \
                     --no-deps cyminiball

Notes:
  - Direct invocation of `setup.py` (e.g., `python setup.py ...`) is deprecated
    and should be avoided.
  - This file remains necessary for dynamic extension building (Cython).
"""

import os
import sys
from pathlib import Path

import numpy
from setuptools import setup, Extension


# ---- Cython decision logic ----
nos = (None, "0", "false")
subcommand = sys.argv[1] if len(sys.argv) > 1 else None

use_cython = (
    subcommand == "build_ext"
    or not Path("src/_miniball_wrap.cpp").is_file()
    or (os.getenv("CYMINIBALL_CYTHONIZE") not in nos)
)

with_trace = os.getenv("CYMINIBALL_TRACE") not in nos


# ---- Extension definition ----
ext = ".pyx" if use_cython else ".cpp"
sources = [f"src/_miniball_wrap{ext}"]

extensions = [
    Extension(
        "cyminiball._wrap",
        sources=sources,
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"],
    )
]


# ---- Optional cythonization ----
if use_cython:
    try:
        from Cython.Build import cythonize

        extensions = cythonize(
            extensions,
            compiler_directives={"linetrace": with_trace},
        )
    except ModuleNotFoundError:
        raise RuntimeError(
            "Cython is required but not installed.\n"
            "Install with: python -m pip install cython"
        ) from None


# ---- Setup call (ONLY build-related config) ----
setup(
    ext_modules=extensions,
)
