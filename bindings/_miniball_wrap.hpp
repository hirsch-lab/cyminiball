#include "miniball/miniball.hpp"
#include <iostream>

template <typename NT>
NT _compute_miniball(NT** points, size_t n_points,
                     NT* center, size_t n_dims)
{
    typedef NT* const* PIt;
    typedef const NT* CIt;
    typedef Miniball::Miniball <Miniball::CoordAccessor<PIt, CIt> > MB;
    MB mb (n_dims, points, points+n_points);
    NT ret = mb.squared_radius();
    for (size_t i=0; i<n_dims; ++i)
        center[i] = mb.center()[i];
    return ret;
}
