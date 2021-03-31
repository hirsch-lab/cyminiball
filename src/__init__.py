try:
    from importlib.metadata import metadata as _meta
except ModuleNotFoundError:
    from importlib_metadata import metadata as _meta

__version__ = _meta("cyminiball")["version"]
__author__ = _meta("cyminiball")["author"]

from ._wrap import (MiniballError,
                    MiniballTypeError,
                    MiniballValueError,
                    compute,
                    compute_no_checks,
                    compute_max_chord)

# To mimic the interface of other miniball projects
from ._compat import (get_bounding_ball,
                      Miniball)
