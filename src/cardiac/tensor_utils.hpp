#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/differentiation/ad/ad_drivers.h>
#include <deal.II/differentiation/ad/ad_helpers.h>
#include <deal.II/fe/fe_values_extractors.h>

#include "../common.hpp"
#include "strain_energy.hpp"

using namespace dealii;

namespace cardiac {

class TensorUtils {
  constexpr static unsigned int dim = 3;
  using ADHelper     = AutoDiff::ScalarFunction<dim, ADTypeCode, double>;
  using ADNumberType = typename ADHelper::ad_type;

public:
  explicit TensorUtils(const StrainEnergy &energy);

  void compute_tensors(Tensor<2, dim>        F,
                       Tensor<2, dim>        &P,
                       Tensor<4, dim>        &C,
                       const Tensor<1, dim>  &f0,
                       const Tensor<1, dim>  &s0,
                       const Tensor<1, dim>  &n0);

private:
  const StrainEnergy &energy_;
  ADHelper            ad_helper;
  Vector<double>      P_flat;
  FullMatrix<double>  C_flat;
};

} // namespace cardiac
