#pragma once

#include "strain_energy.hpp"
#include "../constants.hpp"
#include <deal.II/base/tensor.h>

using namespace dealii;

namespace cardiac {

class HolzapfelOgdenEnergy : public StrainEnergy {
public:
  struct Params {
    double a     = constants::holzapfel::A;
    double b     = constants::holzapfel::B;
    double a_f   = constants::holzapfel::A_F;
    double b_f   = constants::holzapfel::B_F;
    double a_s   = constants::holzapfel::A_S;
    double b_s   = constants::holzapfel::B_S;
    double a_fs  = constants::holzapfel::A_FS;
    double b_fs  = constants::holzapfel::B_FS;
    double kappa = constants::holzapfel::KAPPA;
  };

  HolzapfelOgdenEnergy() = default;
  explicit HolzapfelOgdenEnergy(const Params &p)
  {
    a = p.a; b = p.b;
    a_f = p.a_f; b_f = p.b_f;
    a_s = p.a_s; b_s = p.b_s;
    a_fs = p.a_fs; b_fs = p.b_fs;
    kappa = p.kappa;
  }

  ADNumberType compute_W(const Tensor<2, 3, ADNumberType> &F,
                          const Tensor<1, 3>               &f0,
                          const Tensor<1, 3>               &s0,
                          const Tensor<1, 3>               & /*n0*/) const override
  {
    constexpr unsigned int dim = 3;
    using ADTensor2 = Tensor<2, dim, ADNumberType>;

    const ADNumberType J_ad  = determinant(F);
    const ADTensor2    C_ad  = transpose(F) * F;
    const ADNumberType I1_ad = trace(C_ad);

    ADNumberType I4f_ad  = 0.0;
    ADNumberType I4s_ad  = 0.0;
    ADNumberType I8fs_ad = 0.0;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j) {
        I4f_ad  += f0[i] * C_ad[i][j] * f0[j];
        I4s_ad  += s0[i] * C_ad[i][j] * s0[j];
        I8fs_ad += f0[i] * C_ad[i][j] * s0[j];
      }

    ADNumberType psi_ad = (a / (2.0 * b)) * (std::exp(b * (I1_ad - 3.0)) - 1.0);

    if (I4f_ad > 1.0)
      psi_ad += (a_f / (2.0 * b_f)) * (std::exp(b_f * std::pow(I4f_ad - 1.0, 2.0)) - 1.0);
    if (I4s_ad > 1.0)
      psi_ad += (a_s / (2.0 * b_s)) * (std::exp(b_s * std::pow(I4s_ad - 1.0, 2.0)) - 1.0);

    psi_ad += (a_fs / (2.0 * b_fs)) * (std::exp(b_fs * std::pow(I8fs_ad, 2.0)) - 1.0);
    psi_ad += (kappa / 2.0) * std::pow(J_ad - 1.0, 2.0);

    return psi_ad;
  }

private:
  double a     = constants::holzapfel::A;
  double b     = constants::holzapfel::B;
  double a_f   = constants::holzapfel::A_F;
  double b_f   = constants::holzapfel::B_F;
  double a_s   = constants::holzapfel::A_S;
  double b_s   = constants::holzapfel::B_S;
  double a_fs  = constants::holzapfel::A_FS;
  double b_fs  = constants::holzapfel::B_FS;
  double kappa = constants::holzapfel::KAPPA;
};

} // namespace cardiac
