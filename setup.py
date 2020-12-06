"""Package build and install script.

Useful commands:
    python setup.py clean                   Clean temporary files
    python setup.py sdist                   Create source distribution (.tar.gz)
    python setup.py bdist_wheel             Create built distribution (.whl)
    python setup.py sdist bdist_wheel       Create both
    python setup.py build_ext --inplace     Build C/C++ and Cython extensions
    python setup.py flake8                  Run flake8 (coding style check)
    pip install dist/cyminiball...tar.gz    Install from local tarball
    pip show cyminiball                     Show package information
    pip uninstall cyminiball                Uninstall
    twine check dist/*                      Check the markup in the README
    twine upload --repository testpypi dist/* Upload everything to TestPyPI
    pip install --index-url https://test.pypi.org/simple/ --no-deps cyminiball
"""

import sys
import numpy
from pathlib import Path
from setuptools import setup, Extension


subcommand = sys.argv[1] if len(sys.argv) > 1 else None
USE_CYTHON = subcommand == "build_ext"

packages = ["miniball"]
package_dir = {"miniball": "bindings"}
ext = ".pyx" if USE_CYTHON else ".cpp"
miniball_src = ["bindings/_miniball_wrap"+ext]
include_dirs = [str(Path(__file__).parent.absolute()),
                numpy.get_include()]

extensions = [Extension("miniball",
                        sources=miniball_src,
                        include_dirs=include_dirs,
                        language="c++",
                        extra_compile_args=["-std=c++11"])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(ext_modules=extensions,
      packages=packages,
      package_dir=package_dir)
