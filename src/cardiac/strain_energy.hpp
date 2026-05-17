#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/differentiation/ad/ad_drivers.h>
#include <deal.II/differentiation/ad/ad_helpers.h>

#include "../common.hpp"

namespace cardiac {

class StrainEnergy {
public:
  using ADHelper     = AutoDiff::ScalarFunction<3, ADTypeCode, double>;
  using ADNumberType = typename ADHelper::ad_type;

  virtual ~StrainEnergy() = default;

  virtual ADNumberType compute_W(const dealii::Tensor<2, 3, ADNumberType> &F,
                                  const dealii::Tensor<1, 3>               &f0,
                                  const dealii::Tensor<1, 3>               &s0,
                                  const dealii::Tensor<1, 3>               &n0) const = 0;
};

} // namespace cardiac
