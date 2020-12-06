"""Package build and install script. The Cython dependency is optional
and is not required for pre-built packages. Cython is needed only for
the packaging/deployment.

Trigger a Cython build if one of the following applies:
    ... bindings/_miniball_wrap.cpp is missing
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


subcommand = sys.argv[1] if len(sys.argv) > 1 else None
use_cython = ((subcommand == "build_ext") # This is possibly a bit hacky.
              or not Path("bindings/_miniball_wrap.cpp").is_file()
              or (os.getenv("CYMINIBALL_USE_CYTHON", False)
                  not in (False, "0", "false")))

packages = ["miniball"]
package_dir = {"miniball": "bindings",
               "miniball._wrap": "bindings"}
ext = ".pyx" if use_cython else ".cpp"
miniball_src = ["bindings/_miniball_wrap"+ext]
include_dirs = [str(Path(__file__).parent.absolute()),
                numpy.get_include()]

extensions = [Extension("miniball._wrap",
                        sources=miniball_src,
                        include_dirs=include_dirs,
                        language="c++",
                        extra_compile_args=["-std=c++11"])]

if use_cython:
    try:
        from Cython.Build import cythonize
        extensions = cythonize(extensions)
    except ModuleNotFoundError:
        msg = ("A Cython build was triggered but it is not available.\n"
               "Make sure to install Cython: python -m pip install Cython")
        raise RuntimeError(msg) from None

setup(ext_modules=extensions,
      packages=packages,
      package_dir=package_dir)
