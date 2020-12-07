#!/usr/bin/env bash
# Build and deploy the package without tracing information.

# Clean
rm -rf cyminiball.egg-info
rm -f src/_miniball_wrap.cpp
rm -rf build
rm -rf dist
pip uninstall -y cyminiball

# Build
export CYMINIBALL_TRACE="0"
python setup.py sdist bdist_wheel

# Install
mv cyminiball.egg-info build
pip install dist/cyminiball-*.whl

# Run the tests. (Reads the configs from tox.ini)
# pytest
