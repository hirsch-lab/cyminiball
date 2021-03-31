## v1.0.0 (27.10.2020)
- First functional release

## v2.0.0 (07.12.2020)
- Enhancement: Rename package cyminiball to comply with [PEP 423](https://www.python.org/dev/peps/pep-0423/)
- Enhancement: Refactor function signature of `compute_max_chord()`
- Enhancement: Add argument `tol` to control numerical validity check if (applies if `details=True`)
- Enhancement: Introduce basic setup for CI tools
- Enhancement: Enable coverage analysis for Cython code
- Enhancement: Improve coding style and test coverage
- Fix: Issue #1 Performance problem related to `compute(..., details=True)`
- Fix: Issue #2 (suppress unnecessary runtime errors)


## v2.1.0 (31.03.2021)
- Enhancement: Update documenation
- Enhancement: Add requirements.txt
- Miscellaneous: Add reference to project `miniballcpp`
- Miscellaneous: Add examples/comparison.py
- Fix: Deprecation warnings
- Fix: Animated example for matplotlib>=3.3