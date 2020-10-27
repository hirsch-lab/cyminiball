import sys
import numpy
from setuptools import setup, Extension

# Useful commands:
# python setup.py clean                   Clean temporary files
# python setup.py sdist                   Create source distribution (.tar.gz)
# python setup.py bdist_wheel             Create built distribution (.whl)
# python setup.py sdist bdist_wheel       Create both
# python setup.py build_ext --inplace     Build C/C++ and Cython extensions
# python setup.py flake8                  Run flake8 (coding style check)
# pip install dist/miniball-....tar.gz    Install from local tarball
# pip show miniball-wrap                  Show package information
# pip uninstall miniball-wrap             Uninstall
# twine check dist/*                      Check the markup in the README
# twine upload --repository testpypi dist/* Upload everything to TestPyPI
# pip install --index-url https://test.pypi.org/simple/ --no-deps miniball-wrap

cmdclass = {}
subcommand = sys.argv[1] if len(sys.argv) > 1 else None
if subcommand == "build_ext":
    # This requires Cython. We come here if the extension package is built.
    from Cython.Distutils import build_ext
    # To get some HTML output with an overview of the generate C code.
    import Cython.Compiler.Options
    Cython.Compiler.Options.annotate = True
    # Cython src.
    miniball_src = "bindings/_miniball_wrap.pyx"
    cmdclass["build_ext"] = build_ext
else:
    # This uses the "pre-compiled" Cython output.
    miniball_src = "bindings/_miniball_wrap.cpp"

with open("README.md", encoding="utf-8") as fid:
    long_description = fid.read()

setup(name="cyminiball",
      version="1.0.0",
      url="https://github.com/hirsch-lab/cyminiball",
      author="Norman Juchler",
      author_email="normanius@gmail.com",
      description=("Compute the smallest bounding ball of a point cloud. "
                   "Cython binding of the popular miniball utility. Fast!"),
      long_description=long_description,
      long_description_content_type="text/markdown",
      license="LGPLv3",
      keywords="miniball geometry fast",
      cmdclass=cmdclass,
      ext_modules=[Extension("miniball",
                             sources=[miniball_src],
                             include_dirs=[numpy.get_include(), "."],
                             language="c++",
                             extra_compile_args=["-std=c++11"])],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Software Development :: Libraries",
            "Topic :: Utilities",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
      ],
      # setup_requires=["flake8"]
      )
