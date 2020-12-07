"""Package build and install script. The Cython dependency is optional
and is not required for pre-built packages. Cython is needed only for
the packaging/deployment.

Trigger a Cython build if one of the following applies:
    ... src/_miniball_wrap.cpp is missing
    ... environment variable CYMINIBALL_USE_CYTHON is set
    ... --build_ext command line argument is supplied (a bit hacky)

This bundles everything into a bundle named miniball:
    miniball
    - __init__.py
    - _wrap.cpython-*.so (on MacOS)

Useful commands:
    python setup.py clean                   Clean temporary files
    python setup.py sdist                   Create source distr. (.tar.gz)
    python setup.py bdist_wheel             Create built distr. (.whl)
    python setup.py sdist bdist_wheel       Create both
    python setup.py build_ext --inplace     Build C/C++ and Cython extensions
    python setup.py flake8                  Run flake8 (coding style check)
    python -m pep517.build .                Build using module pep517
    python -m build .                       Build using PyPA's build module
    pip install dist/cyminiball...tar.gz    Install from tarball
    pip install dist/cyminiball....whl      Install from wheel
    pip show cyminiball                     Show package information
    pip uninstall cyminiball                Uninstall
    twine check dist/*                      Check the markup in the README
    twine upload --repository testpypi dist/* Upload everything to TestPyPI
    pip install --index-url https://test.pypi.org/simple/ --no-deps cyminiball
"""
import os
import sys
import numpy
from pathlib import Path
from setuptools import setup, Extension


# Decide whether a cythonization of the pyx-file is required.
# Cythonization is done automatically if the generated .cpp
# does not yet exist, if setup.py is called with positional
# argument build_ext or if the envvar CYMINIBALL_CYTHONIZE
# is set to 1.
nos = (None, "0", "false")
subcommand = sys.argv[1] if len(sys.argv) > 1 else None
use_cython = ((subcommand == "build_ext")  # This is possibly a bit hacky.
              or not Path("src/_miniball_wrap.cpp").is_file()
              or (os.getenv("CYMINIBALL_CYTHONIZE", None) not in nos))
# Decide whether the binaries should be built with tracing info.
# Tracing is required for a coverage analysis of the Cython code.
# - http://blog.behnel.de/posts/coverage-analysis-for-cython-modules.html
# - https://medium.com/@dfdeshom/631615eb197a
# To build a package with tracing information, build in two steps!
#   python setup.py build_ext --inplace --define CYTHON_TRACE
#   python python setup.py sdist bdist_wheel
with_trace = os.getenv("CYMINIBALL_TRACE", None) not in nos

if use_cython:
    print("Package is built with cythonization.")
if with_trace:
    print("Package is built with line trace information.")

packages = ["cyminiball"]
package_dir = {"cyminiball": "src",
               "cyminiball._wrap": "src"}
ext = ".pyx" if use_cython else ".cpp"
miniball_src = ["src/_miniball_wrap"+ext]
include_dirs = [str(Path(__file__).parent.absolute()),
                numpy.get_include()]

extensions = [Extension("cyminiball._wrap",
                        sources=miniball_src,
                        include_dirs=include_dirs,
                        language="c++",
                        extra_compile_args=["-std=c++11"])]

if use_cython:
    dirs = {"linetrace": with_trace}
    try:
        from Cython.Build import cythonize
        extensions = cythonize(extensions,
                               compiler_directives=dirs)
    except ModuleNotFoundError:
        msg = ("A Cython build was triggered but Cython is not available.\n"
               "Make sure to install Cython: python -m pip install Cython")
        raise RuntimeError(msg) from None

setup(ext_modules=extensions,
      packages=packages,
      package_dir=package_dir)
