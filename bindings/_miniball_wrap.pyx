# cython: infer_types=True
# cython: language_level=3
import numpy as np
import cython
cimport numpy as cnp
cimport cython

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
# cdefine the signature of our c function
# cdef extern from "_miniball_wrap_simple.hpp" nogil:
#     # T _compute_miniball[T](const T** points, size_t n_points,
#     #                        T* center, size_t n_dims)
#     T _compute_miniball[T](T** points, size_t n_points,
#                            T* center, size_t n_dims)

cdef extern from "_miniball_wrap.hpp" nogil:
    # T _compute_miniball[T](const T** points, size_t n_points,
    #                        T* center, size_t n_dims)
    T _compute_miniball[T](T** points, size_t n_points,
                           T* center, size_t n_dims)


################################################################################
def _compute_float(float_type[:,:] points not None):
    # TODO: add option "extended" to enable additional output:
    # {
    #     "center" : [ x, y, z ],             Center of minimal bounding sphere
    #     "radius" : r,                       Radius of minimal bounding sphere
    #     "support" : {                       Points that define the sphere
    #         "ids" : [ id1, id2, id3  ],     Ids of the supporting points
    #         "points" : [                    Coordinates of supporting points
    #             [ x1, y1, z1 ],
    #             [ x2, y2, z2 ],
    #             [ x3, y3, z3 ],
    #         ],
    #         "maxDist" : max,                Max pairwise dist between support
    #         "maxIds" : [ 1, 3 ]             Ids of the above points for maxDist
    #     },
    #     "validity" : 1,                     Typically 1 if valid
    #     "relError" : 4.68937e-15,           Typically very small if valid
    #     "suboptimality" : 0                 Typically very small if valid
    # }

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
    cdef size_t n_bytes = points_view.shape[0] * sizeof(float_type*)
    #cdef const float_type** point_ptrs = <const float_type **>PyMem_Malloc(n_bytes)
    cdef float_type** point_ptrs = <float_type **>PyMem_Malloc(n_bytes)

    if not point_ptrs:
        raise MemoryError
    try:
        for i in range(points_view.shape[0]):
            point_ptrs[i] = &points_view[i, 0]
        # Call the C function that expects a float_type**
        r2 = _compute_miniball(point_ptrs, points_view.shape[0],
                               &center_view[0], points_view.shape[1])
    finally:
        PyMem_Free(point_ptrs)
    if np.isnan(center).all():
        center = None
    ret = (center, r2)
    return ret


################################################################################
def compute_no_checks(points):
    """Compute the bounding ball without any checks. Otherwise equivalent to
    :ref:`~miniball.compute`.
    """
    return _compute_float(points)


################################################################################
def compute(points):
    """Compute the bounding ball for a set of points with arbitrary dimensions.
    The code runs the popular and fast miniball algorithm by
    `Bernd Gärtner <https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html>`_.

    Returns a tuple (c, r2) with the center c and the squared radius r2 of
    the miniball.
    """
    if points is None:
        return (None, 0)
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
        return (None, 0)
    elif len(points.shape)<2:
        # Handle a 0D or 1D array as a list of points in 1D.
        points = np.atleast_2d(points).T
    elif len(points.shape)>2:
        msg = "Expecting a 2D array but received a %dD array (shape: %s)."
        raise MiniballTypeError(msg % (len(points.shape), points.shape))
    return compute_no_checks(points)


################################################################################
def get_bounding_ball(points):
    """A synonym for :func:`~miniball.compute` with the purpose to make this
    package a drop-in replacement for another
    `miniball project <https://pypi.org/project/miniball/>`__ available on PyPi
    """
    return compute(points)
