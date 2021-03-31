#!/usr/bin/env bash
# Build and deploy the package with tracing information.
# Make sure to set the file .coveragerc to contain:
#       [run]
#       plugins = Cython.Coverage

# Clean
rm -rf cyminiball.egg-info
rm -f src/_miniball_wrap.cpp
rm -rf build
rm -rf dist
pip uninstall -y cyminiball
pip cache remove cyminiball

# Build
export CYMINIBALL_TRACE="1"
python setup.py build_ext --inplace --define CYTHON_TRACE
python setup.py sdist bdist_wheel

# Install
mv cyminiball.egg-info build
pip install dist/cyminiball-*.whl

# Run the tests. (Reads the configs from tox.ini)
# pytest
