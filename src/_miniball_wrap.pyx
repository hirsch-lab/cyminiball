# cython: infer_types=True
# cython: language_level=3
import time
import numpy as np
import cython
cimport numpy as cnp
cimport cython
from libcpp cimport bool, limits
from cpython.mem cimport PyMem_Malloc, PyMem_Free


# Good practice: Use the Numpy-C-API from Cython
cnp.import_array()

ctypedef fused float_type:
    # Warning: longdouble has limited support in cython:
    #          https://stackoverflow.com/questions/423404
    #          https://stackoverflow.com/questions/25380004
    # Effect:  The variable r2 below is represented as normal
    #          python float (cython.double). The ndarray however
    #          are represented correctly, and miniball also
    #          employs the long double data type.
    # Note:    Long doubles are equivalent to doubles on Windows
    cython.longdouble
    cython.float
    cython.double

class MiniballError(Exception):
    pass
class MiniballTypeError(TypeError, MiniballError):
    pass
class MiniballValueError(ValueError, MiniballError):
    pass


################################################################################
cdef extern from "_miniball_wrap.hpp" nogil:
    bool _compute_miniball[T](T** points, size_t n_points,
                              T* center, size_t n_dims,
                              T& r2, T tol)
    bool _compute_miniball_extended[T](T** points, size_t n_points,
                                       T* center, size_t n_dims, T& r2,
                                       int* support_ids, int& n_support,
                                       T& suboptimality, T& relative_error,
                                       T& elapsed, T tol)


################################################################################
def _compute_float(float_type[:,:] points not None,
                   bool details,
                   float_type tol):
    """
    The tolerance is used to set the is_valid flag (if details=True).
    is_valid is set as a function of the relative_error and suboptimality:
    Note that the relative error depends on the size of the miniball.
    For smaller miniballs, the relative error increases.
    In miniball.hpp, the suggested tolerance is set to 10 times the
    machine's tolerance.
    """
    # Handle different floating-point types properly.
    if float_type is cython.float:
        dtype = np.float32
    elif float_type is cython.double:
        dtype = np.float64
        #dtype = np.double
    elif float_type is cython.longdouble:
        # Don't use np.float128, it may not work on Windows/conda.
        # https://stackoverflow.com/questions/29820829
        # dtype = np.float128
        dtype = np.longdouble
    else:
        # In the unlikely event we come here, check the
        # available/supported cython floating point types
        # and extend the fused float_type above.
        # import cython; print(cython.float_types)
        assert(False)

    # Prepare return values.
    center = np.zeros(points.shape[1], dtype=dtype)
    cdef float_type[:] center_view = center
    cdef float_type r2 = 0.
    cdef int n_support = 0
    cdef float_type suboptimality = 0.
    cdef float_type relative_error = 0.
    cdef float_type elapsed = 0.
    cdef bool is_valid = False
    if tol < 0:
        tol = 10*limits.numeric_limits[float_type].epsilon()

    start = time.time()
    # If float_type is cython.longdouble, r2 is a normal float (cython.double).
    # TODO: Understand why! See comment above (ctypedef).
    # print(type(r2))

    # The following line often is a no-op, unless slices are involved.
    # The difference between asarray() and ascontiguousarray() are minimal:
    #       https://stackoverflow.com/questions/22105529
    # Usually, an array is not contiguous if it was sliced or transposed.
    #       points.T.flags["C_CONTIGUOUS"]            # False
    #       points[1:-1:2, ::3].flags["C_CONTIGUOUS"] # False
    # Note that for random access selections, numpy usually creates copies
    # and ensures contiguity.
    #       points[[1,2,3], ::3].flags["C_CONTIGUOUS"] # True
    # Cython cares nicely about strides. So far I didn't encounter any
    # problems with non-contiguity/slices.
    cdef float_type[:,::1] points_view = np.ascontiguousarray(points, dtype=dtype)

    #cdef float_type[:,:] points_view = np.asarray(points, dtype=dtype)
    cdef Py_ssize_t n_dims = points_view.shape[1]
    cdef Py_ssize_t n_points = points_view.shape[0]
    cdef size_t n_bytes = n_points * sizeof(float_type*)
    cdef size_t n_bytes_ids = n_dims * sizeof(int)

    #cdef const float_type** point_ptrs = <const float_type **>PyMem_Malloc(n_bytes)
    cdef float_type** point_ptrs = <float_type **>PyMem_Malloc(n_bytes)

    # Result container.
    support_ids = np.empty(n_dims+1, dtype=np.int32)
    cdef int[:] support_ids_view = support_ids

    if not point_ptrs:
        raise MemoryError
    try:
        for i in range(n_points):
            point_ptrs[i] = &points_view[i, 0]
        if details:
            # Compute miniball with extended output, at the cost of some
            # overhead (measured ~10% for large and medium size problems).
            is_valid = _compute_miniball_extended(
                point_ptrs, n_points,
                &center_view[0], n_dims, r2,
                &support_ids_view[0], n_support,
                suboptimality, relative_error,
                elapsed, tol)
        else:
            # Compute miniball with basic output: center and squared radius.
            is_valid = _compute_miniball(point_ptrs, n_points,
                                         &center_view[0], n_dims,
                                         r2, tol)
    finally:
        PyMem_Free(point_ptrs)
    stop = time.time()
    if np.isnan(center).all():
        center = None
        r2 = np.nan
    if details:
        support_ids = support_ids[:n_support]
        info = { "center": center,
                 "radius": np.sqrt(r2),
                 "support": support_ids,
                 "n_support": n_support,
                 "relative_error": relative_error,
                 "suboptimality": suboptimality,
                 "is_valid": is_valid,
                 "elapsed": elapsed,
                 #"elapsed_all": stop-start
               }
        ret = (center, r2, info)
    else:
        ret = (center, r2)
    #print(stop-start)
    return ret


################################################################################
def compute_no_checks(points, details=False, tol=None):
    """Compute the minimal bounding ball without any checks. Otherwise
    equivalent to miniball.compute().
    """
    tol = -1 if tol is None else tol
    return _compute_float(points, details, tol)


################################################################################
def compute(points, details=False, tol=None):
    """Compute the minimal bounding ball for a set of points with arbitrary
    dimensions. The code runs the popular and efficient miniball algorithm
    by Bernd Gärtner [1].

    [1] https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html

    Args:
        points: Data array of shape (n,d) containing n points in d dimensions.
        details: Enable additional output about the miniball. Default: False
        tol: Tolerance. Only relevant if details=True. It affects the
             is_valid flag of the returned info dictionary, which represents
             the result of a numerical validity test. It has no effect on the
             actual miniball (center, radius). By default, tol is set to 10x
             the machine epsilon of the dtype. The numerical validity test
             for a miniball depends on its actual size. Smaller balls are
             more likely to fail the test. The test is defined as follows:
                (relative_error < tol) && (suboptimality == 0)

    Returns:
        A tuple (c, r2) with the center c and  the squared radius r2 of the
        miniball. If details is True, a tuple (c, r2, info), where info is a
        dictionary with the following keys:

            center: center of the miniball
            radius: radius of the miniball
            support: indices of the points that define the miniball
            relative_error: numerical error measure
            is_valid: flag indicating the numerical validity of the miniball
            elapsed: time elapsed for the computation

        See the code documentation in [1] for further details.
    """
    if points is None:
        raise MiniballValueError("Argument points cannot be None.")
    elif issubclass(type(points), str):
        msg = "Expecting a 2D array but received a string."
        raise MiniballTypeError(msg)
    elif isinstance(points, set):
        # TODO: Extract pointers of the set and feed directly to
        # _compute_miniball instead of creating a superfluous copy.
        points = list(points)

    # Create an ndarray.
    points = np.asarray(points)

    # Check dtype, miniball supports only floating point data.
    if points.dtype.kind in ("i", "u", "b"):
        # In principle, we could also use np.single.
        points = np.asarray(points, dtype=float)
    if issubclass(points.dtype.type, complex):
        msg = "Complex arrays are not supported by miniball."
        raise MiniballValueError(msg)
    elif points.dtype.kind != "f":
        msg = ("Invalid dtype (%s) encountered. Expecting a "
               "numeric 2D array of points.")
        raise MiniballValueError(msg % points.dtype)
    if issubclass(points.dtype.type, np.float16):
        msg = ("Invalid dtype (np.float16) encountered. Use np.float instead.")
        raise MiniballValueError(msg)

    # Check shape.
    if points.size==0:
        raise MiniballValueError("No data to process, points is empty.")
    elif len(points.shape)<2:
        # Handle a 0D or 1D array as a list of points in 1D.
        points = np.atleast_2d(points).T
    elif len(points.shape)>2:
        msg = "Expecting a 2D array but received a %dD array (shape: %s)."
        raise MiniballValueError(msg % (len(points.shape), points.shape))
    return compute_no_checks(points, details, tol)


################################################################################
def get_bounding_ball(points):
    """An alias for miniball.compute() with the purpose to make the
    cyminiball package a drop-in replacement for another miniball project
    available on PyPi: https://pypi.org/project/miniball/
    """
    return compute(points, details=False, tol=None)


################################################################################
def compute_max_chord(points, info=None, details=False, tol=None):
    """Compute the longest chord between the support points of the miniball.
    If info is None, compute(points, details=True) will be called internally:

        # Alternative A:
        (p1, p2), d_max = compute_max_chord(points)
        # Alternative B:
        _, _, info = compute(..., detailed=True)
        (p1, p2), d_max = compute_max_chord(points=points, info=info)


    Returns:
        pts_max:    Point coordinates that form the maximum chord
        d_max:      The length of the maximum chord
        info:       Optional, if details=True. An info dictionary with
                    additional data about the miniball, extended by the
                    maximum chord info. Extends the info dictionary
                    (in-place) by the following keys: ids_max, d_max.
    """
    if points is None:
        raise MiniballValueError("Argument points cannot be None.")
    if info is None:
        _, _, info = compute(points=points, details=True, tol=tol)
    points = np.atleast_1d(points)
    if len(points.shape)<2:
        info["ids_max"] = np.array([info["support"][0], info["support"][-1]])
        info["pts_max"] = points[info["ids_max"]]
        info["d_max"] = info["radius"]*2
    else:
        support = points[info["support"]]
        pdist = np.linalg.norm(support[:,None,:] - support[None,:,:], axis=-1)
        ids_max = np.unravel_index(np.argmax(pdist, axis=None), pdist.shape)
        ids_max = list(ids_max)
        info["ids_max"] = info["support"][ids_max]
        info["pts_max"] = points[info["ids_max"]]
        info["d_max"] = pdist[tuple(ids_max)]
    if details:
        ret = (info["pts_max"], info["d_max"], info)
    else:
        ret = (info["pts_max"], info["d_max"])
    return ret

