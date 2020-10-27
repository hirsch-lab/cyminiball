# cyminiball

Compute the smallest bounding ball of a point cloud in arbitrary dimensions. A Python/Cython binding of the popular [miniball](https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html) utility by Bernd Gärtner.

The code is provided under the LGPLv3 license.

For an implementation in pure Python, see [`miniball`](https://pypi.org/project/miniball/). `cyminiball` can be used as a drop-in replacement for `miniball`; it runs much faster because it is based on an effcient C++ implementation.

### Installation:

    pip install cyminiball

### Usage:

```python
import miniball 
import numpy as np

d = 2           # Number of dimensions
n = 10000       # Number of points 
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
#       relative_error: numerical error measure
#       is_valid:       numerical validity
#       elapsed:        time required 

# The maximal chord is the longest line connecting any
# two of the support points. The folowing extends the 
# info dict by the following keys:
#       ids_max:        ids of the two extreme points
#       d_max:          length of the maximal chord
info = compute_max_chord(points, info=info)
```

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
python test/run_tests.py
python test/example.py
```
