#include "miniball/miniball.hpp"
#include <iostream>
#include <numeric>

/**
C-style API function.
*/
template <typename T>
bool _compute_miniball(T** points, size_t n_points,
                       T* center, size_t n_dims,
                       T& r2) {
    typedef T* const* PIt;
    typedef const T* CIt;
    typedef Miniball::Miniball <Miniball::CoordAccessor<PIt, CIt> > MB;
    MB mb (n_dims, points, points+n_points);
    r2 = mb.squared_radius();
    for (size_t i=0; i<n_dims; ++i)
        center[i] = mb.center()[i];
    return mb.is_valid();
}


/**
C-style API function with additional output.
*/
template <typename T>
bool _compute_miniball_extended(T** points, size_t n_points,
                                T* center, size_t n_dims,
                                T& r2, T& d2_max,
                                int& id0_max, int& id1_max,
                                T& suboptimality, T& relative_error,
                                T& elapsed)
{
    typedef T* const* PIt;
    typedef const T* CIt;
    typedef Miniball::Miniball <Miniball::CoordAccessor<PIt, CIt> > MB;

    r2 = -1;
    d2_max = -1;
    id0_max = -1;
    id1_max = -1;
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

    // Support points
    int n_support = mb.nr_support_points();
    typename MB::SupportPointIterator it = mb.support_points_begin();
    T** support = new T*[n_support];
    int* support_ids = new int[n_support];
    for (int i=0; i<n_support; ++i, ++it) {
        support[i] = new T[n_dims];
        support_ids[i] = (*it)-points;
        std::copy((**it), (**it)+n_dims, support[i]);
    }

    // Max distances.
    auto squaredDistance = [&](const T* p1, const T* p2) {
        auto squareDiff = [](T v1, T v2) { return (v1-v2)*(v1-v2); };
        auto summand = [](T v1, T v2) { return v1+v2; };
        return std::inner_product<const T*, const T*, T>
                (p1, p1+n_dims, p2, 0., summand, squareDiff);
    };
    for (int i=0; i<n_support; ++i) {
        for (int j=0; j<n_support; ++j) {
            T d2 = squaredDistance(support[i], support[j]);
            if (d2>d2_max) {
                d2_max = d2;
                id0_max = support_ids[i];
                id1_max = support_ids[j];
            }
        }
    }
    for (int i=0; i<n_support; ++i) {
        delete [] support[i];
    }
    delete [] support;
    delete [] support_ids;
    elapsed = mb.get_time();
    suboptimality = 1;
    relative_error = mb.relative_error(suboptimality);
    return mb.is_valid();
}

// template <typename T>
// bool _compute_miniball(T** points, size_t n_points,
//                        T* center, size_t n_dims,
//                        T& r2) {
//     r2 = 0.;
//     T d2_max = 0.;
//     int id0_max = -1;
//     int id1_max = -1;
//     T suboptimality = 0.;
//     T relative_error = 0.;
//     bool is_valid = false;
//     T elapsed = 0.;
//     return _compute_miniball_extended(points, n_points,
//                                       center, n_dims,
//                                       r2, d2_max,
//                                       id0_max, id1_max,
//                                       suboptimality, relative_error,
//                                       is_valid, elapsed);
// }
