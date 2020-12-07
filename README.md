# cyminiball

<!--https://raw.githubusercontent.com/yngvem/group-lasso/master/README.rst-->

<!--[![Downloads](https://pepy.tech/badge/cyminiball)](https://pepy.tech/project/cyminiball)-->
<!--https://pypistats.org/packages/cyminiball-->
[![image](https://img.shields.io/pypi/v/cyminiball.svg)](https://pypi.org/project/cyminiball/)
[![License](https://img.shields.io/pypi/l/cyminiball)](https://github.com/hirsch-lab/cyminiball/blob/main/LICENSE)
[![Build Status](https://travis-ci.org/hirsch-lab/cyminiball.svg?branch=main)](https://travis-ci.org/hirsch-lab/cyminiball)
[![Coverage Status](https://coveralls.io/repos/github/hirsch-lab/cyminiball/badge.svg?branch=main)](https://coveralls.io/github/hirsch-lab/cyminiball?branch=main)
[![CodeFactor](https://www.codefactor.io/repository/github/hirsch-lab/cyminiball/badge)](https://www.codefactor.io/repository/github/hirsch-lab/cyminiball)
[![DeepSource](https://deepsource.io/gh/hirsch-lab/cyminiball.svg/?label=active+issues)](https://deepsource.io/gh/hirsch-lab/cyminiball/?ref=repository-badge)


Cyminiball is a utility to compute the smallest bounding ball of a point cloud in arbitrary dimensions. A Python/Cython binding of the popular [miniball](https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html) utility by Bernd Gärtner.

The code is provided under the LGPLv3 license.

For an implementation in pure Python, see [`miniball`](https://pypi.org/project/miniball/). `cyminiball` can be used as a drop-in replacement for `miniball`; it runs much faster because it is based on an effcient C++ implementation.

### Installation:

    pip install cyminiball

### Usage:

```python
import cyminiball as miniball
import numpy as np

d = 2           # Number of dimensions
n = 10000       # Number of points
dt = np.float   # Data type

points = np.random.randn(n, d)
points = points.astype(dt)
C, r2 = miniball.compute(points)
print("Center:", C)
print("Radius:", np.sqrt(r2))
```

Additional output can be generated using the `details` flag and `compute_max_chord()`.

```python
C, r2, info = miniball.compute(points, details=True)
# Returns an info dict with the following keys:
#       center:         center
#       radius:         radius
#       support:        indices of the support points
#       relative_error: error measure realtive to r2
#       is_valid:       numerical validity
#       elapsed:        time required
#
# The maximal chord is the longest line connecting any
# two of the support points. The following extends the
# info dict by the following keys:
#       pts_max:        point coordinates of the two points
#       ids_max:        ids of the two extreme points
#       d_max:          length of the maximal chord
(p1, p2), d_max = miniball.compute_max_chord(points, info=info)
```

See [examples/examples.py](https://github.com/hirsch-lab/cyminiball) for further usage examples

### Build

To build the package requires

- Python 3.x
- Cython
- numpy

```bash
git clone https://github.com/hirsch-lab/cyminiball.git
cd cyminiball
python setup.py build_ext --inplace
python setup.py sdist bdist_wheel
python test/test_all.py
python examples/examples.py
```
