#ifndef TET_CUT_INTERSECTION_TESTS_HPP
#define TET_CUT_INTERSECTION_TESTS_HPP

#include <Eigen/Core>

namespace geometry {

/**
 * @brief
 * Computes intersection point, if any, of a triangle ABC in counterclockwise order and a finite
 * line segment
 * @param a Vertex 1
 * @param b Vertex 2
 * @param c Vertex 3
 * @param line Line segment as a pair of points (p,q)
 * @return Pair of intersection success boolean and intersection point (if any)
 */
std::pair<bool, Eigen::Vector3d> triangle_line_intersection(
    Eigen::Vector3d const& a,
    Eigen::Vector3d const& b,
    Eigen::Vector3d const& c,
    std::pair<Eigen::Vector3d, Eigen::Vector3d> const& line)
{
    Eigen::Vector3d const& p = line.first;
    Eigen::Vector3d const& q = line.second;
    Eigen::Vector3d const ab = b - a;
    Eigen::Vector3d const ac = c - a;
    Eigen::Vector3d const qp = p - q;

    Eigen::Vector3d const n = ab.cross(ac);

    double const d = qp.dot(n);
    if (d <= 0.)
        return {false, {}};

    Eigen::Vector3d const ap = p - a;

    double t = ap.dot(n);
    if (t < 0.)
        return {false, {}};
    if (t > d)
        return {false, {}};

    Eigen::Vector3d const e = qp.cross(ap);

    double v = ac.dot(e);
    if (v < 0. || v > d)
        return {false, {}};

    double w = -ab.dot(e);
    if (w < 0. || (v + w) > d)
        return {false, {}};

    double const ood = 1. / d;
    t *= ood;
    //v *= ood;
    //w *= ood;
    //double const u = 1. - v - w;

    Eigen::Vector3d const intersection = (1. - t) * p + t * q;
    return {true, intersection};
}

} // namespace geometry

#endif // TET_CUT_INTERSECTION_TESTS_HPP