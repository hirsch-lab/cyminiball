# cython: infer_types=True
# cython: language_level=3
import time
import numpy as np
import cython
cimport numpy as cnp
cimport cython
from libcpp cimport bool
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
    cython.longdouble
    cython.float
    cython.double

class MiniballError(Exception):
    pass
class MiniballTypeError(TypeError, MiniballError):
    pass

################################################################################
cdef extern from "_miniball_wrap.hpp" nogil:
    bool _compute_miniball[T](T** points, size_t n_points,
                              T* center, size_t n_dims, T& r2)
    bool _compute_miniball_extended[T](T** points, size_t n_points,
                                       T* center, size_t n_dims, T& r2,
                                       int* support_ids, int& n_support,
                                       T& suboptimality, T& relative_error,
                                       T& elapsed)

################################################################################
def _compute_float(float_type[:,:] points not None, bool details):
    # Handle different floating-point types properly.
    if float_type is cython.float:
        dtype = np.float32
    elif float_type is cython.double:
        dtype = np.float64
        #dtype = np.double
    elif float_type is cython.longdouble:
        dtype = np.float128
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
    start = time.time()
    # If float_type is cython.longdouble, r2 is a normal float (cython.double).
    # TODO: Understand why! See comment above (ctypedef).
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
                elapsed)
        else:
            # Compute miniball with basic output: center and squared radius.
            is_valid = _compute_miniball(point_ptrs, n_points,
                                         &center_view[0], n_dims, r2)
        if not is_valid:
            msg = "Encountered a problem when computing the miniball."
            raise MiniballError(msg)
    finally:
        PyMem_Free(point_ptrs)
    stop = time.time()
    if np.isnan(center).all():
        center = None
        r2 = np.nan
        is_valid = False
    if details:
        support_ids = support_ids[:n_support]
        dct = { "center": center,
                "r2": r2,
                "support": support_ids,
                "n_support": n_support,
                "relative_error": relative_error,
                "suboptimality": suboptimality,
                "is_valid": is_valid,
                "elapsed": elapsed,
                #"elapsed_all": stop-start
              }
        ret = (center, r2, dct)
    else:
        ret = (center, r2)
    #print(stop-start)
    return ret

################################################################################
def compute_no_checks(points, details=False):
    """Compute the bounding ball without any checks. Otherwise equivalent to
    :ref:`~miniball.compute`.
    """
    return _compute_float(points, details)


################################################################################
def compute(points, details=False):
    """Compute the bounding ball for a set of points with arbitrary dimensions.
    The code runs the popular and fast miniball algorithm by
    `Bernd Gärtner <https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html>`_.

    If details is False, a tuple (c, r2) is returned with the center c and
    the squared radius r2 of the miniball. If details is True, (c, r2, det)
    is returned, where det is a dictionary containing additional details about
    the bounding sphere.
    """
    ret_default = (None, 0, None) if details else (None, 0)
    if points is None:
        return ret_default
    elif issubclass(type(points), str):
        msg = "Expecting a 2D array but received a string."
        raise MiniballTypeError(msg)
    elif isinstance(points, set):
        # TODO: Extract pointers of the set and feed directly to
        # _compute_miniball instead of creating a superfluous copy.
        points = list(points)

    # Create an ndarray.
    points = np.asarray(points)

    # Check dtype, miniball supports only floating point data.
    if points.dtype.kind in ("i", "u", "b"):
        # In principle, we could also use np.single.
        points = np.asarray(points, dtype=float)
    if issubclass(points.dtype.type, complex):
        msg = "Complex arrays are not supported by miniball."
        raise MiniballTypeError(msg)
    elif points.dtype.kind != "f":
        msg = ("Invalid dtype (%s) encountered. Expecting a "
               "numeric 2D array of points.")
        raise MiniballTypeError(msg % points.dtype)
    if issubclass(points.dtype.type, np.float16):
        msg = ("Invalid dtype (np.float16) encountered. Use np.float instead.")
        raise MiniballTypeError(msg)

    # Check shape.
    if points.size==0:
        return ret_default
    elif len(points.shape)<2:
        # Handle a 0D or 1D array as a list of points in 1D.
        points = np.atleast_2d(points).T
    elif len(points.shape)>2:
        msg = "Expecting a 2D array but received a %dD array (shape: %s)."
        raise MiniballTypeError(msg % (len(points.shape), points.shape))
    return compute_no_checks(points, details)

################################################################################
def get_bounding_ball(points):
    """A synonym for :func:`~miniball.compute` with the purpose to make this
    package a drop-in replacement for another
    `miniball project <https://pypi.org/project/miniball/>`__ available on PyPi
    """
    return compute(points)

################################################################################
def compute_max_chord(details, points):
    """
    Compute the longest chord between the support points of the miniball.
    This requires the detailed result dictionary by compute():
        _, _, detailed = compute(..., detailed=True)
    """
    points = np.asarray(points)
    support = points[details["support"]]
    pdist = np.linalg.norm(support[:,None,:] - support[None,:,:], axis=-1)
    ids_max = list(np.unravel_index(np.argmax(pdist, axis=None), pdist.shape))
    details["ids_max"] = details["support"][ids_max]
    details["d_max"] = pdist[ids_max]
    return details

