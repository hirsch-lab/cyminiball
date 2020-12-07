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
                    compute_max_chord,
                    get_bounding_ball)
