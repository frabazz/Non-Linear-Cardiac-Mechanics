#pragma once

#include "fiber_field.hpp"
#include "../constants.hpp"

#include <cmath>
#include <algorithm>

using namespace dealii;

namespace cardiac {

class AnalyticalProlateFibers : public FiberField {
public:
  struct Params {
    double d          = constants::guccione::PROLATE_D;
    double xi_endo    = constants::guccione::PROLATE_XI_ENDO;
    double xi_epi     = constants::guccione::PROLATE_XI_EPI;
    double alpha_endo = constants::FIBER_ALPHA_ENDO;
    double alpha_epi  = constants::FIBER_ALPHA_EPI;
  };

  AnalyticalProlateFibers() = default;
  explicit AnalyticalProlateFibers(const Params &p) : p_(p) {}

  void frame_at(const DoFHandler<3>::active_cell_iterator & /*cell*/,
                 unsigned int                               /*q_point*/,
                 const Point<3>                            &x,
                 Tensor<1, 3>                              &f0,
                 Tensor<1, 3>                              &s0,
                 Tensor<1, 3>                              &n0) const override
  {
    const double d = p_.d;
    const double xc = x[0], y = x[1], z = x[2];

    const double r1 = std::sqrt(xc*xc + y*y + (z-d)*(z-d));
    const double r2 = std::sqrt(xc*xc + y*y + (z+d)*(z+d));
    const double xi = std::acosh(std::max(1.0, (r1 + r2) / (2.0 * d)));
    const double theta = std::acos(std::max(-1.0, std::min(1.0, (r2 - r1) / (2.0 * d))));
    const double phi = std::atan2(y, xc);

    double cosphi = std::cos(phi);
    double sinphi = std::sin(phi);
    const double r = std::sqrt(cosphi*cosphi + sinphi*sinphi);
    if (r > 1e-12) { cosphi /= r; sinphi /= r; }
    else            { cosphi = 1.0; sinphi = 0.0; }

    Tensor<1, 3> g_xi, g_theta, g_phi;
    g_xi[0] = d * std::cosh(xi) * std::sin(theta) * cosphi;
    g_xi[1] = d * std::cosh(xi) * std::sin(theta) * sinphi;
    g_xi[2] = d * std::sinh(xi) * std::cos(theta);

    g_theta[0] = d * std::sinh(xi) * std::cos(theta) * cosphi;
    g_theta[1] = d * std::sinh(xi) * std::cos(theta) * sinphi;
    g_theta[2] = -d * std::cosh(xi) * std::sin(theta);

    g_phi[0] = -d * std::sinh(xi) * std::sin(theta) * sinphi;
    g_phi[1] =  d * std::sinh(xi) * std::sin(theta) * cosphi;
    g_phi[2] = 0.0;

    const Tensor<1, 3> e_xi    = g_xi    / g_xi.norm();
    const Tensor<1, 3> e_theta = g_theta / g_theta.norm();
    const Tensor<1, 3> e_phi   = g_phi   / g_phi.norm();

    const double alpha_endo_r = p_.alpha_endo * M_PI / 180.0;
    const double alpha_epi_r  = p_.alpha_epi  * M_PI / 180.0;
    double wall_fraction = (xi - p_.xi_endo) / (p_.xi_epi - p_.xi_endo);
    wall_fraction = std::max(0.0, std::min(1.0, wall_fraction));
    const double alpha = alpha_endo_r + (alpha_epi_r - alpha_endo_r) * wall_fraction;

    n0 = e_xi;
    f0 = std::cos(alpha) * e_phi + std::sin(alpha) * e_theta;
    f0 /= f0.norm();
    s0 = cross_product_3d(n0, f0);
    s0 /= s0.norm();
  }

private:
  Params p_;
};

} // namespace cardiac
