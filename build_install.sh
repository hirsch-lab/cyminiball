#!/usr/bin/env bash
# Build and install the package (no tracing information)

set -euo pipefail

echo "==> Cleaning previous build artifacts"
rm -rf build dist *.egg-info
rm -f src/_miniball_wrap.cpp

# Optional cleanup (ignore if not installed / not cached)
python -m pip uninstall -y cyminiball || true
python -m pip cache remove cyminiball || true

echo "==> Preparing build environment"
python -m pip install --quiet --upgrade pip build

echo "==> Building package"
export CYMINIBALL_TRACE="0"
python -m build

echo "==> Installing wheel"
# Ensure only one wheel is selected
WHEEL=$(ls dist/cyminiball-*.whl | head -n 1)
mkdir build
[ -d cyminiball.egg-info ] && mv cyminiball.egg-info build
python -m pip install "$WHEEL"

echo "==> Done"
