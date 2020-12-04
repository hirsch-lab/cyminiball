#include "miniball/miniball.hpp"
#include <iostream>
#include <numeric>

template <typename T>
bool _is_valid(T rel_error, T subopt, T tol) {
    // Same test as mb.is_valid() with variable tol.
    return (rel_error <= tol) && (subopt == 0);
}


/**
C-style API function.
*/
template <typename T>
bool _compute_miniball(T** points, size_t n_points,
                       T* center, size_t n_dims,
                       T& r2, T tol) {
    typedef T* const* PIt;
    typedef const T* CIt;
    typedef Miniball::Miniball <Miniball::CoordAccessor<PIt, CIt> > MB;
    MB mb (n_dims, points, points+n_points);
    r2 = mb.squared_radius();
    for (size_t i=0; i<n_dims; ++i)
        center[i] = mb.center()[i];

    T subopt = 0;
    T rel_error = mb.relative_error(subopt);
    return _is_valid(rel_error, subopt, tol);
}


/**
C-style API function with additional output.

support_ids must be preallocated with size n_dims+1.
The actual number of support points varies and is returned by n_support.
*/
template <typename T>
bool _compute_miniball_extended(T** points, size_t n_points,
                                T* center, size_t n_dims, T& r2,
                                int* support_ids, int& n_support,
                                T& suboptimality, T& relative_error,
                                T& elapsed, T tol)
{
    typedef T* const* PIt;
    typedef const T* CIt;
    typedef Miniball::Miniball <Miniball::CoordAccessor<PIt, CIt> > MB;

    r2 = -1;
    suboptimality = 0.;
    relative_error = 0.;
    elapsed = 0.;

    if (n_points < 1 || n_dims < 1) {
        return false;
    }

    // Compute the miniball.
    MB mb (n_dims, points, points+n_points);

    // Squared radius and center of the ball.
    r2 = mb.squared_radius();
    for (size_t i=0; i<n_dims; ++i) {
        center[i] = mb.center()[i];
    }

    // Support points.
    n_support = mb.nr_support_points();
    typename MB::SupportPointIterator it = mb.support_points_begin();
    for (int i=0; i<n_support; ++i, ++it) {
        support_ids[i] = (*it)-points;
    }

    // Misc.
    elapsed = mb.get_time();
    suboptimality = 1;
    relative_error = mb.relative_error(suboptimality);
    return _is_valid(relative_error, suboptimality, tol);
}
