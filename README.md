# cyminiball

Compute the smallest bounding ball of a point cloud in arbitrary dimensions. A Python/Cython binding of the popular [miniball](https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html) utility by Bernd Gärtner.

The code is provided under the LGPLv3 license.

For an implementation in pure Python, see [`miniball`](https://pypi.org/project/miniball/). `cyminiball` can be used as a drop-in replacement. It runs much faster because it is based on an effcient C++ implementation.

### Installation:

    pip install miniball-wrap

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
```
