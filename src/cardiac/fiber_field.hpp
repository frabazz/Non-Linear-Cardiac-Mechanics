#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>

#include "../constants.hpp"

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_vector.h>

#include <map>
#include <vector>
#include <cmath>


namespace cardiac {

class FiberField {
public:
  virtual ~FiberField() = default;

  virtual void frame_at(const dealii::DoFHandler<3>::active_cell_iterator &cell,
                         unsigned int                                       q_point,
                         const dealii::Point<3>                            &x,
                         dealii::Tensor<1, 3>                              &f0,
                         dealii::Tensor<1, 3>                              &s0,
                         dealii::Tensor<1, 3>                              &n0) const = 0;
};

  

class LaplaceFibers : public FiberField {
public:
  LaplaceFibers(const DoFHandler<3>                 &dof_handler_poisson,
                const TrilinosWrappers::MPI::Vector &lambda_ghost,
                const FiniteElement<3>              &fe_poisson,
                const Quadrature<3>                 &quadrature)
  {
    FEValues<3> fev(fe_poisson, quadrature, update_values | update_gradients);
    const unsigned int n_q = quadrature.size();

    for (const auto &cell : dof_handler_poisson.active_cell_iterators()) {
      if (!cell->is_locally_owned()) continue;
      fev.reinit(cell);
      PerCellData d;
      d.lam.resize(n_q);
      d.grad_lam.resize(n_q);
      fev.get_function_values(lambda_ghost, d.lam);
      fev.get_function_gradients(lambda_ghost, d.grad_lam);
      cell_data_[cell->id()] = std::move(d);
    }
  }

  void frame_at(const DoFHandler<3>::active_cell_iterator &cell,
                unsigned int                               q,
                const Point<3>                            & /*x*/,
                Tensor<1, 3>                              &f0,
                Tensor<1, 3>                              &s0,
                Tensor<1, 3>                              &n0) const override
  {
    namespace cc  = cardiac::constants;
    namespace cch = cardiac::constants::holzapfel;

    const auto  &d   = cell_data_.at(cell->id());
    const double lam = d.lam[q];

    Tensor<1, 3> e2 = d.grad_lam[q];
    e2 /= e2.norm();

    Tensor<1, 3> k_axis;
    k_axis[2] = -1.0;
    Tensor<1, 3> e1 = k_axis - (k_axis * e2) * e2;
    e1 /= e1.norm();
    Tensor<1, 3> e0 = cross_product_3d(e1, e2);

    
    const double alpha = (cc::FIBER_ALPHA_ENDO  * (1.0 - lam) + cc::FIBER_ALPHA_EPI  * lam) * M_PI / 180.0;
    const double beta  = (cch::FIBER_BETA_ENDO * (1.0 - lam) + cch::FIBER_BETA_EPI * lam) * M_PI / 180.0;

    f0 = std::cos(alpha) * e0 + std::sin(alpha) * e1;
    const Tensor<1, 3> g_hat = -std::sin(alpha) * e0 + std::cos(alpha) * e1;
    s0 = std::cos(beta) * g_hat + std::sin(beta) * e2;
    n0 = cross_product_3d(f0, s0);
  }

private:
  struct PerCellData {
    std::vector<double>       lam;
    std::vector<Tensor<1, 3>> grad_lam;
  };

  std::map<CellId, PerCellData> cell_data_;
};




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
    const Tensor<1, 3> e_phi   = -g_phi   / g_phi.norm();

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


class TrivialFibers : public FiberField {
public:
  void frame_at(const dealii::DoFHandler<3>::active_cell_iterator & /*cell*/,
                 unsigned int                                        /*q_point*/,
                 const dealii::Point<3>                            & /*x*/,
                 dealii::Tensor<1, 3>                              &f0,
                 dealii::Tensor<1, 3>                              &s0,
                 dealii::Tensor<1, 3>                              &n0) const override
  {
    f0 = dealii::Tensor<1, 3>();  f0[0] = 1.0;
    s0 = dealii::Tensor<1, 3>();  s0[1] = 1.0;
    n0 = dealii::Tensor<1, 3>();  n0[2] = 1.0;
  }
};

  
} // namespace cardiac

