#pragma once

#include "fiber_field.hpp"
#include "../constants.hpp"

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_vector.h>

#include <map>
#include <vector>
#include <cmath>

using namespace dealii;

namespace cardiac {

class LaplaceFibers : public FiberField {
public:
  LaplaceFibers(const DoFHandler<3>                   &dof_handler_poisson,
                const TrilinosWrappers::MPI::Vector   &lambda_ghost,
                const FiniteElement<3>                &fe_poisson,
                const Quadrature<3>                   &quadrature)
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
                 unsigned int                              q,
                 const Point<3>                           & /*x*/,
                 Tensor<1, 3>                             &f0,
                 Tensor<1, 3>                             &s0,
                 Tensor<1, 3>                             &n0) const override
  {
    namespace cc  = cardiac::constants;
    namespace cch = cardiac::constants::holzapfel;

    const auto &d  = cell_data_.at(cell->id());
    const double lam = d.lam[q];

    Tensor<1, 3> e2 = d.grad_lam[q];
    e2 /= e2.norm();

    Tensor<1, 3> ref;
    {
      unsigned int min_idx = 0;
      for (unsigned int dd = 1; dd < 3; ++dd)
        if (std::abs(e2[dd]) < std::abs(e2[min_idx]))
          min_idx = dd;
      ref[min_idx] = 1.0;
    }
    Tensor<1, 3> e1 = ref - (ref * e2) * e2;
    e1 /= e1.norm();

    Tensor<1, 3> e0 = cross_product_3d(e1, e2);

    const double alpha = (cc::FIBER_ALPHA_ENDO * (1.0 - lam) + cc::FIBER_ALPHA_EPI  * lam) * M_PI / 180.0;
    const double beta  = (cch::FIBER_BETA_ENDO * (1.0 - lam) + cch::FIBER_BETA_EPI  * lam) * M_PI / 180.0;

    f0 = std::cos(alpha) * e0 + std::sin(alpha) * e1;
    const Tensor<1, 3> g_hat = -std::sin(alpha) * e0 + std::cos(alpha) * e1;
    s0 = std::cos(beta) * g_hat + std::sin(beta) * e2;
    n0 = cross_product_3d(f0, s0);
  }

private:
  struct PerCellData {
    std::vector<double>         lam;
    std::vector<Tensor<1, 3>>   grad_lam;
  };
  std::map<CellId, PerCellData> cell_data_;
};

} // namespace cardiac
