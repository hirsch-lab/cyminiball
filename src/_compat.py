from ._wrap import compute


def get_bounding_ball(points):
    """An alias for miniball.compute() with the purpose to make the
    cyminiball package a drop-in replacement for another miniball project
    available on PyPI: https://pypi.org/project/miniball/
    """
    return compute(points, details=False, tol=None)


class Miniball:
    """Mimic the interface of yet another miniball PyPI project:
    https://pypi.org/project/MiniballCpp/
    """
    def __init__(self, points):
        _, _, info = compute(points, details=True)
        self._info = info

    def center(self):
        return self._info["center"]

    def squared_radius(self):
        return self._info["radius"]**2

    def relative_error(self):
        return self._info["relative_error"]

    def is_valid(self):
        return self._info["is_valid"]

    def get_time(self):
        return self._info["elapsed"]
