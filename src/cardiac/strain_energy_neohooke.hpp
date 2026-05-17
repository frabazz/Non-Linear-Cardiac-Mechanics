#pragma once

#include "strain_energy.hpp"
#include <deal.II/base/tensor.h>

using namespace dealii;

namespace cardiac {

class NeoHookeEnergy : public StrainEnergy {
public:
  struct Params {
    double mu    = 4.0;
    double kappa = 50.0;
  };

  NeoHookeEnergy() = default;
  explicit NeoHookeEnergy(const Params &p) : mu(p.mu), kappa(p.kappa) {}

  ADNumberType compute_W(const Tensor<2, 3, ADNumberType> &F,
                          const Tensor<1, 3>               & /*f0*/,
                          const Tensor<1, 3>               & /*s0*/,
                          const Tensor<1, 3>               & /*n0*/) const override
  {
    const ADNumberType J_ad  = determinant(F);
    const ADNumberType I1_ad = trace(transpose(F) * F);

    ADNumberType psi_ad = (mu / 2.0) * (I1_ad * std::pow(J_ad, -2.0 / 3.0) - 3.0);
    psi_ad += (kappa / 2.0) * std::pow(J_ad - 1.0, 2.0);
    return psi_ad;
  }

private:
  double mu    = 4.0;
  double kappa = 50.0;
};

} // namespace cardiac
