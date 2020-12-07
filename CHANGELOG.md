## v1.0.0 (27.10.2020)
- First functional release

## v2.0.0 (07.12.2020)
- Enhancement: Renamed package cyminiball to comply with [PEP 423](https://www.python.org/dev/peps/pep-0423/)
- Enhancement: Refactored function signature of `compute_max_chord()`
- Enhancement: Added argument `tol` to control numerical validity check if (applies if `details=True`)
- Enhancement: Introduced basic setup for CI tools
- Enhancement: Enable coverage analysis for Cython code
- Enhancement: Improved coding style and test coverage
- Fix: Issue #1 Performance problem related to `compute(..., details=True)`
- Fix: Issue #2 (suppress unnecessary runtime errors)
